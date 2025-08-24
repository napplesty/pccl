#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define BLOCK_SIZE 128
#define WARPSIZE 32
#define MAX_BLOCK_SIZE 256

// 前向传播kernel
__global__ void flash_attn_v2_forward_kernel(
    const half* Q,    // [batch_size, seq_len, num_heads, head_dim]
    const half* K,    // [batch_size, seq_len, num_heads, head_dim]
    const half* V,    // [batch_size, seq_len, num_heads, head_dim]
    half* O,          // [batch_size, seq_len, num_heads, head_dim]
    half* l,          // [batch_size, num_heads, seq_len] - 对数求和
    half* m,          // [batch_size, num_heads, seq_len] - 最大值
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale
) {
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;
    
    extern __shared__ half shared_mem[];
    half* K_shared = shared_mem;
    half* V_shared = &shared_mem[head_dim * BLOCK_SIZE];
    
    int q_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    int o_offset = q_offset;
    
    // 每个线程处理一个head_dim元素
    if (tid < head_dim) {
        half max_val = -1e4;
        half sum_val = 0.0;
        
        for (int j = 0; j < seq_len; j += BLOCK_SIZE) {
            // 加载K, V到共享内存
            int load_idx = j + tid;
            if (load_idx < seq_len) {
                int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + load_idx) * head_dim;
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    if (tid < BLOCK_SIZE) {
                        K_shared[tid * head_dim + d] = K[k_offset + d];
                        V_shared[tid * head_dim + d] = V[k_offset + d];
                    }
                }
            }
            __syncthreads();
            
            // 计算QK^T
            for (int i = 0; i < min(BLOCK_SIZE, seq_len - j); i++) {
                half dot_product = 0.0;
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    dot_product += Q[q_offset + tid * head_dim + d] * 
                                  K_shared[i * head_dim + d];
                }
                dot_product *= __float2half(scale);
                
                // Online softmax update
                half old_max = max_val;
                half new_val = dot_product;
                max_val = fmaxf(__half2float(max_val), __half2float(new_val));
                
                half exp_old = hexp(__hmul(__hsub(old_max, max_val), sum_val));
                half exp_new = hexp(__hsub(new_val, max_val));
                
                sum_val = __hadd(__hmul(exp_old, sum_val), exp_new);
            }
            __syncthreads();
        }
        
        // 计算输出
        half output_val = 0.0;
        for (int j = 0; j < seq_len; j += BLOCK_SIZE) {
            // 重新加载K, V
            int load_idx = j + tid;
            if (load_idx < seq_len) {
                int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + load_idx) * head_dim;
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    if (tid < BLOCK_SIZE) {
                        K_shared[tid * head_dim + d] = K[k_offset + d];
                        V_shared[tid * head_dim + d] = V[k_offset + d];
                    }
                }
            }
            __syncthreads();
            
            for (int i = 0; i < min(BLOCK_SIZE, seq_len - j); i++) {
                half dot_product = 0.0;
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    dot_product += Q[q_offset + tid * head_dim + d] * 
                                  K_shared[i * head_dim + d];
                }
                dot_product *= __float2half(scale);
                
                half attention = hexp(__hsub(dot_product, max_val));
                attention = __hdiv(attention, sum_val);
                
                output_val += attention * V_shared[i * head_dim + tid];
            }
            __syncthreads();
        }
        
        O[o_offset + tid] = output_val;
        
        // 保存softmax统计量用于反向传播
        if (tid == 0) {
            int stats_idx = (batch_idx * num_heads + head_idx) * seq_len + blockIdx.x;
            l[stats_idx] = sum_val;
            m[stats_idx] = max_val;
        }
    }
}

// 反向传播kernel
__global__ void flash_attn_v2_backward_kernel(
    const half* dO,       // 上游梯度 [batch_size, seq_len, num_heads, head_dim]
    const half* Q,
    const half* K,
    const half* V,
    const half* O,
    const half* l,        // 前向传播的统计量
    const half* m,        // 前向传播的统计量
    half* dQ,
    half* dK,
    half* dV,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale
) {
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;
    
    extern __shared__ half shared_mem[];
    half* K_shared = shared_mem;
    half* V_shared = &shared_mem[head_dim * BLOCK_SIZE];
    half* dV_shared = &shared_mem[2 * head_dim * BLOCK_SIZE];
    
    int q_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    int stats_idx = (batch_idx * num_heads + head_idx) * seq_len;
    
    if (tid < head_dim) {
        // 首先计算dV
        for (int j = 0; j < seq_len; j += BLOCK_SIZE) {
            // 加载K, V到共享内存
            int load_idx = j + tid;
            if (load_idx < seq_len) {
                int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + load_idx) * head_dim;
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    if (tid < BLOCK_SIZE) {
                        K_shared[tid * head_dim + d] = K[k_offset + d];
                        V_shared[tid * head_dim + d] = V[k_offset + d];
                    }
                }
            }
            __syncthreads();
            
            // 计算attention权重并累加到dV
            for (int i = 0; i < min(BLOCK_SIZE, seq_len - j); i++) {
                half dot_product = 0.0;
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    dot_product += Q[q_offset + tid * head_dim + d] * 
                                  K_shared[i * head_dim + d];
                }
                dot_product *= __float2half(scale);
                
                half max_val = m[stats_idx + blockIdx.x];
                half sum_val = l[stats_idx + blockIdx.x];
                half attention = hexp(__hsub(dot_product, max_val));
                attention = __hdiv(attention, sum_val);
                
                half dO_val = dO[q_offset + tid];
                atomicAdd(&dV_shared[i * head_dim + tid], attention * dO_val);
            }
            __syncthreads();
            
            // 写回dV
            if (tid < BLOCK_SIZE) {
                int v_offset = ((batch_idx * num_heads + head_idx) * seq_len + j + tid) * head_dim;
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    atomicAdd(&dV[v_offset + d], dV_shared[tid * head_dim + d]);
                }
            }
            __syncthreads();
        }
        
        // 类似地计算dQ和dK（简化显示，实际实现需要更复杂的计算）
        // ... 这里需要实现完整的反向传播逻辑
    }
}

// 包装函数
void flash_attn_v2_forward(
    const half* Q, const half* K, const half* V,
    half* O, half* l, half* m,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float scale, cudaStream_t stream
) {
    dim3 grid_size((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);
    int shared_mem_size = 2 * head_dim * BLOCK_SIZE * sizeof(half);
    
    flash_attn_v2_forward_kernel<<<grid_size, head_dim, shared_mem_size, stream>>>(
        Q, K, V, O, l, m, batch_size, seq_len, num_heads, head_dim, scale
    );
}

void flash_attn_v2_backward(
    const half* dO, const half* Q, const half* K, const half* V, const half* O,
    const half* l, const half* m, half* dQ, half* dK, half* dV,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float scale, cudaStream_t stream
) {
    dim3 grid_size((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);
    int shared_mem_size = 3 * head_dim * BLOCK_SIZE * sizeof(half);
    
    flash_attn_v2_backward_kernel<<<grid_size, head_dim, shared_mem_size, stream>>>(
        dO, Q, K, V, O, l, m, dQ, dK, dV,
        batch_size, seq_len, num_heads, head_dim, scale
    );
}