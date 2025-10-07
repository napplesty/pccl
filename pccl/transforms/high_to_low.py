from typing import Dict, List, Set, Optional, Any
import logging

from ..ir.core import CollectiveIR, PrimitiveOp, Chunk, Device, PrimitiveOpType, DeviceType
from ..ir.types import PrimitiveOpType as HighPrimitiveOpType
from ..low_ir.core import LowLevelIR, PrimitiveConfig, BufferConfig, ExecutorConfig, TensorInfo
from ..low_ir.types import PrimitiveType, DataType, ComputeType, ExecutorType

logger = logging.getLogger(__name__)

class HighToLowConverter:
    """高级IR到低级IR转换器"""
    
    def __init__(self):
        self._device_type_map: Dict[DeviceType, ExecutorType] = {
            DeviceType.CUDA: ExecutorType.CUDA,
            DeviceType.CPU: ExecutorType.CPU,
            DeviceType.ROCM: ExecutorType.CUDA,  # 映射到CUDA作为临时方案
        }
        
        self._op_type_map: Dict[HighPrimitiveOpType, PrimitiveType] = {
            HighPrimitiveOpType.COPY: PrimitiveType.COPY,
            HighPrimitiveOpType.REDUCE: PrimitiveType.COMPUTE,
            HighPrimitiveOpType.NOTIFY: PrimitiveType.SIGNAL,
            HighPrimitiveOpType.GET_NOTIFIED: PrimitiveType.WAITSIGNAL,
        }
        
        self._compute_op_map: Dict[str, ComputeType] = {
            'sum': ComputeType.SUM,
            'max': ComputeType.MAX,
            'min': ComputeType.MIN,
            'prod': ComputeType.PROD,
        }
    
    def convert(self, high_ir: CollectiveIR) -> LowLevelIR:
        """将高级IR转换为低级IR
        
        Args:
            high_ir: 高级集合通信IR
            
        Returns:
            转换后的低级IR
        """
        logger.info(f"开始转换高级IR: {high_ir.name}")
        
        # 创建低级IR
        low_ir = LowLevelIR(high_ir.name)
        
        # 转换设备拓扑
        self._convert_devices(high_ir, low_ir)
        
        # 转换缓冲区
        buffer_mapping = self._convert_buffers(high_ir, low_ir)
        
        # 转换操作序列
        op_mapping = self._convert_operations(high_ir, low_ir, buffer_mapping)
        
        # 转换依赖关系
        self._convert_dependencies(high_ir, low_ir, op_mapping)
        
        # 转换输入输出张量
        self._convert_tensors(high_ir, low_ir, buffer_mapping)
        
        # 验证转换结果
        if not low_ir.validate():
            logger.warning("转换后的低级IR验证失败")
        
        logger.info(f"转换完成: {high_ir.name} -> {low_ir.name}")
        return low_ir
    
    def _convert_devices(self, high_ir: CollectiveIR, low_ir: LowLevelIR):
        """转换设备拓扑"""
        for device in high_ir.devices.values():
            executor_type = self._device_type_map.get(device.device_type, ExecutorType.CPU)
            low_ir.add_executor(executor_type, 1)  # 每个设备一个执行器
    
    def _convert_buffers(self, high_ir: CollectiveIR, low_ir: LowLevelIR) -> Dict[int, int]:
        """转换缓冲区配置
        
        Returns:
            缓冲区映射: 高级IR缓冲区ID -> 低级IR缓冲区ID
        """
        buffer_mapping = {}
        
        # 为每个设备的每个数据块创建缓冲区
        for op in high_ir.get_operation_sequence():
            # 处理源缓冲区
            if op.src_chunk and op.src_chunk.cur_device_id not in buffer_mapping:
                device_id = op.src_chunk.cur_device_id
                buffer_id = self._create_buffer_for_chunk(low_ir, op.src_chunk, device_id)
                buffer_mapping[device_id] = buffer_id
            
            # 处理目标缓冲区
            if op.tgt_chunk and op.tgt_chunk.cur_device_id not in buffer_mapping:
                device_id = op.tgt_chunk.cur_device_id
                buffer_id = self._create_buffer_for_chunk(low_ir, op.tgt_chunk, device_id)
                buffer_mapping[device_id] = buffer_id
        
        return buffer_mapping
    
    def _create_buffer_for_chunk(self, low_ir: LowLevelIR, chunk: Chunk, device_id: int) -> int:
        """为数据块创建缓冲区"""
        # 推断数据类型（默认为F32）
        dtype = DataType.F32
        
        # 推断执行器类型（默认为CUDA）
        executor_type = ExecutorType.CUDA
        
        # 计算缓冲区大小
        size = chunk.data_size
        
        return low_ir.add_buffer(dtype, size, executor_type)
    
    def _convert_operations(self, high_ir: CollectiveIR, low_ir: LowLevelIR, 
                           buffer_mapping: Dict[int, int]) -> Dict[int, int]:
        """转换操作序列
        
        Returns:
            操作映射: 高级IR操作ID -> 低级IR操作ID
        """
        op_mapping = {}
        
        for high_op in high_ir.get_operation_sequence():
            # 转换操作类型
            low_op_type = self._op_type_map.get(high_op.op_type, PrimitiveType.COPY)
            
            # 创建低级操作配置
            op_config = PrimitiveConfig(
                type=low_op_type,
                dtype=DataType.F32,  # 默认为F32
                target_rank=-1,
                src_buffer_idx=-1,
                dst_buffer_idx=-1,
                compute_op=ComputeType.SUM,
                executor_type=ExecutorType.CUDA,
                num_executors=1,
                data_size=0,
                signal_value=0,
                num_dependencies=0,
                num_followers=0
            )
            
            # 设置缓冲区索引
            if high_op.src_chunk:
                device_id = high_op.src_chunk.cur_device_id
                op_config.src_buffer_idx = buffer_mapping.get(device_id, -1)
            
            if high_op.tgt_chunk:
                device_id = high_op.tgt_chunk.cur_device_id
                op_config.dst_buffer_idx = buffer_mapping.get(device_id, -1)
            
            # 设置数据大小
            if high_op.src_chunk:
                op_config.data_size = high_op.src_chunk.data_size
            elif high_op.tgt_chunk:
                op_config.data_size = high_op.tgt_chunk.data_size
            
            # 设置计算操作类型
            if high_op.op_type == HighPrimitiveOpType.REDUCE:
                compute_op = high_op.metadata.get('compute_op', 'sum')
                op_config.compute_op = self._compute_op_map.get(compute_op, ComputeType.SUM)
            
            # 设置执行器类型
            if high_op.src_device and high_op.src_device in high_ir.devices:
                device_type = high_ir.devices[high_op.src_device].device_type
                op_config.executor_type = self._device_type_map.get(device_type, ExecutorType.CPU)
            
            # 添加操作到低级IR
            low_op_id = low_ir.add_operator(op_config)
            op_mapping[high_op.op_id] = low_op_id
        
        return op_mapping
    
    def _convert_dependencies(self, high_ir: CollectiveIR, low_ir: LowLevelIR, 
                             op_mapping: Dict[int, int]):
        """转换依赖关系"""
        for high_op in high_ir.get_operation_sequence():
            low_op_id = op_mapping.get(high_op.op_id)
            if low_op_id is None:
                continue
            
            # 转换依赖关系
            for dep_id in high_op.dependencies:
                low_dep_id = op_mapping.get(dep_id)
                if low_dep_id is not None:
                    low_ir.add_dependency(low_dep_id, low_op_id)
    
    def _convert_tensors(self, high_ir: CollectiveIR, low_ir: LowLevelIR, 
                        buffer_mapping: Dict[int, int]):
        """转换输入输出张量信息"""
        # 转换输入张量（前置条件）
        for chunk in high_ir.precondition:
            tensor_info = TensorInfo(
                shape=(chunk.data_size // 4,),  # 假设F32类型
                dtype=DataType.F32,
                device_type=ExecutorType.CUDA,
                device_id=chunk.cur_device_id
            )
            low_ir.add_input_tensor(tensor_info)
        
        # 转换输出张量（后置条件）
        for chunk in high_ir.postcondition:
            tensor_info = TensorInfo(
                shape=(chunk.data_size // 4,),  # 假设F32类型
                dtype=DataType.F32,
                device_type=ExecutorType.CUDA,
                device_id=chunk.cur_device_id
            )
            low_ir.add_output_tensor(tensor_info)
