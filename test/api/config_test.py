import torch
import pccl

def format_bytes(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def print_config():
    config = pccl.Config
    max_name_length = max(len(name) for name in config.__dict__ if not name.startswith('__'))
    
    print("PCCL Configuration Values:")
    print("-" * (max_name_length + 20))
    
    # 按分类分组打印
    categories = {
        "Kernel Settings": [
            'WARP_SIZE', 'WARP_PER_SM', 'WARP_FOR_SCHEDULE',
            'WARP_FOR_PROXY', 'WARP_FOR_MEMORY', 'MAX_SM_COUNT'
        ],
        "Buffer Configuration": [
            'LIB_BUFFER_SIZE', 'DEVICE_BUFFER_SIZE',
            'HOST_BUFFER_SIZE', 'NUM_SLOT'
        ],
        "Network Parameters": [
            'MAX_CHANNEL_PER_OPERATION', 'MAX_ACTIVE_CONNECTIONS',
            'MAX_CQ_SIZE', 'MAX_CQ_POLL_NUM'
        ],
        "Proxy Settings": [
            'PROXY_FLUSH_PERIOD', 'PROXY_CHECK_STOP_PERIOD'
        ]
    }

    for category, keys in categories.items():
        print(f"\n[{category}]")
        for key in keys:
            value = getattr(config, key)
            # 特殊处理缓冲区大小
            if 'BUFFER_SIZE' in key:
                value_str = f"{format_bytes(value)} ({value:,} bytes)"
            else:
                value_str = f"{value:,}"
                
            print(f"  {key:<{max_name_length}} : {value_str}")

if __name__ == "__main__":
    print_config()