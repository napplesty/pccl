from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum
import json

from .types import PrimitiveType, DataType, ComputeType, ExecutorType, TensorInfo

@dataclass
class BufferConfig:
    buffer_idx: int
    dtype: DataType
    size: int
    executor_type: ExecutorType
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'buffer_idx': self.buffer_idx,
            'dtype': self.dtype.name,
            'size': self.size,
            'executor_type': self.executor_type.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BufferConfig':
        return cls(
            buffer_idx=data['buffer_idx'],
            dtype=DataType[data['dtype']],
            size=data['size'],
            executor_type=ExecutorType[data['executor_type']]
        )

@dataclass
class ExecutorConfig:
    executor_type: ExecutorType
    num_total_executors: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'executor_type': self.executor_type.name,
            'num_total_executors': self.num_total_executors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutorConfig':
        return cls(
            executor_type=ExecutorType[data['executor_type']],
            num_total_executors=data['num_total_executors']
        )

@dataclass
class PrimitiveConfig:
    type: PrimitiveType
    dtype: DataType
    target_rank: int = -1
    src_buffer_idx: int = -1
    dst_buffer_idx: int = -1
    compute_op: ComputeType = ComputeType.SUM
    executor_type: ExecutorType = ExecutorType.CPU
    num_executors: int = 1
    data_size: int = 0
    signal_value: int = 0
    num_dependencies: int = 0
    num_followers: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.name,
            'dtype': self.dtype.name,
            'target_rank': self.target_rank,
            'src_buffer_idx': self.src_buffer_idx,
            'dst_buffer_idx': self.dst_buffer_idx,
            'compute_op': self.compute_op.name,
            'executor_type': self.executor_type.name,
            'num_executors': self.num_executors,
            'data_size': self.data_size,
            'signal_value': self.signal_value,
            'num_dependencies': self.num_dependencies,
            'num_followers': self.num_followers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PrimitiveConfig':
        return cls(
            type=PrimitiveType[data['type']],
            dtype=DataType[data['dtype']],
            target_rank=data.get('target_rank', -1),
            src_buffer_idx=data.get('src_buffer_idx', -1),
            dst_buffer_idx=data.get('dst_buffer_idx', -1),
            compute_op=ComputeType[data.get('compute_op', 'SUM')],
            executor_type=ExecutorType[data.get('executor_type', 'CPU')],
            num_executors=data.get('num_executors', 1),
            data_size=data.get('data_size', 0),
            signal_value=data.get('signal_value', 0),
            num_dependencies=data.get('num_dependencies', 0),
            num_followers=data.get('num_followers', 0)
        )

class LowLevelIR:
    def __init__(self, name: str = ""):
        self.name = name
        self.buffers: Dict[int, BufferConfig] = {}
        self.executors: Dict[int, ExecutorConfig] = {}
        self.operators: Dict[int, PrimitiveConfig] = {}
        self.dependencies: Dict[int, Set[int]] = {}  # op_id -> 依赖的op_ids
        self.input_tensors: List[TensorInfo] = []
        self.output_tensors: List[TensorInfo] = []
        self._next_buffer_id = 0
        self._next_executor_id = 0
        self._next_operator_id = 0
    
    def add_buffer(self, dtype: DataType, size: int, executor_type: ExecutorType) -> int:
        buffer_id = self._next_buffer_id
        self._next_buffer_id += 1
        buffer = BufferConfig(buffer_id, dtype, size, executor_type)
        self.buffers[buffer_id] = buffer
        return buffer_id
    
    def add_executor(self, executor_type: ExecutorType, num_total_executors: int) -> int:
        executor_id = self._next_executor_id
        self._next_executor_id += 1
        executor = ExecutorConfig(executor_type, num_total_executors)
        self.executors[executor_id] = executor
        return executor_id
    
    def add_operator(self, op_config: PrimitiveConfig, dependencies: List[int] = None) -> int:
        op_id = self._next_operator_id
        self._next_operator_id += 1
        
        op_config_copy = PrimitiveConfig(
            type=op_config.type,
            dtype=op_config.dtype,
            target_rank=op_config.target_rank,
            src_buffer_idx=op_config.src_buffer_idx,
            dst_buffer_idx=op_config.dst_buffer_idx,
            compute_op=op_config.compute_op,
            executor_type=op_config.executor_type,
            num_executors=op_config.num_executors,
            data_size=op_config.data_size,
            signal_value=op_config.signal_value,
            num_dependencies=op_config.num_dependencies,
            num_followers=op_config.num_followers
        )
        
        self.operators[op_id] = op_config_copy
        self.dependencies[op_id] = set(dependencies) if dependencies else set()
        
        return op_id
    
    def add_dependency(self, from_op_id: int, to_op_id: int):
        if to_op_id not in self.dependencies:
            self.dependencies[to_op_id] = set()
        self.dependencies[to_op_id].add(from_op_id)
    
    def add_input_tensor(self, tensor_info: TensorInfo):
        self.input_tensors.append(tensor_info)
    
    def add_output_tensor(self, tensor_info: TensorInfo):
        self.output_tensors.append(tensor_info)
    
    def to_json(self) -> str:
        data = {
            'name': self.name,
            'buffers': [buffer.to_dict() for buffer in self.buffers.values()],
            'executors': [executor.to_dict() for executor in self.executors.values()],
            'operators': [op.to_dict() for op in self.operators.values()],
            'dependencies': {str(k): list(v) for k, v in self.dependencies.items()},
            'input_tensors': [
                {
                    'shape': tensor.shape,
                    'dtype': tensor.dtype.name,
                    'device_type': tensor.device_type.name,
                    'device_id': tensor.device_id
                }
                for tensor in self.input_tensors
            ],
            'output_tensors': [
                {
                    'shape': tensor.shape,
                    'dtype': tensor.dtype.name,
                    'device_type': tensor.device_type.name,
                    'device_id': tensor.device_id
                }
                for tensor in self.output_tensors
            ]
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LowLevelIR':
        """从JSON创建LowLevelIR"""
        data = json.loads(json_str)
        low_ir = cls(data.get('name', ''))
        
        # 重建缓冲区
        for buffer_data in data.get('buffers', []):
            buffer = BufferConfig.from_dict(buffer_data)
            low_ir.buffers[buffer.buffer_idx] = buffer
            low_ir._next_buffer_id = max(low_ir._next_buffer_id, buffer.buffer_idx + 1)
        
        # 重建执行器
        for executor_data in data.get('executors', []):
            executor = ExecutorConfig.from_dict(executor_data)
            low_ir.executors[executor.executor_type] = executor
        
        # 重建操作
        for op_data in data.get('operators', []):
            op = PrimitiveConfig.from_dict(op_data)
            low_ir.operators[op.op_id] = op
            low_ir._next_operator_id = max(low_ir._next_operator_id, op.op_id + 1)
        
        # 重建依赖
        for op_id_str, deps in data.get('dependencies', {}).items():
            op_id = int(op_id_str)
            low_ir.dependencies[op_id] = set(deps)
        
        # 重建输入输出张量
        for tensor_data in data.get('input_tensors', []):
            tensor = TensorInfo(
                shape=tuple(tensor_data['shape']),
                dtype=DataType[tensor_data['dtype']],
                device_type=ExecutorType[tensor_data['device_type']],
                device_id=tensor_data.get('device_id', 0)
            )
            low_ir.add_input_tensor(tensor)
        
        for tensor_data in data.get('output_tensors', []):
            tensor = TensorInfo(
                shape=tuple(tensor_data['shape']),
                dtype=DataType[tensor_data['dtype']],
                device_type=ExecutorType[tensor_data['device_type']],
                device_id=tensor_data.get('device_id', 0)
            )
            low_ir.add_output_tensor(tensor)
        
        return low_ir
    
    def validate(self) -> bool:
        """验证低级IR的完整性"""
        # 检查所有依赖的操作都存在
        for op_id, deps in self.dependencies.items():
            if op_id not in self.operators:
                return False
            for dep_id in deps:
                if dep_id not in self.operators:
                    return False
        
        # 检查缓冲区索引有效
        for op in self.operators.values():
            if op.src_buffer_idx != -1 and op.src_buffer_idx not in self.buffers:
                return False
            if op.dst_buffer_idx != -1 and op.dst_buffer_idx not in self.buffers:
                return False
        
        return True
