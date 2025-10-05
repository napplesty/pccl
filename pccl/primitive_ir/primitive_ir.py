# pccl/primitive_ir/primitive_ir.py
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Union
from enum import Enum, auto
import json

class PrimitiveType(Enum):
    WRITE = auto()
    COMPUTE = auto()
    COPY = auto()
    SIGNAL = auto()
    WAITSIGNAL = auto()

class DataType(Enum):
    F32 = auto()
    F16 = auto()
    BF16 = auto()

class ComputeType(Enum):
    SUM = auto()
    MAX = auto()
    MIN = auto()
    PROD = auto()

class ExecutorType(Enum):
    CPU = auto()
    CUDA = auto()
    LAST = auto()

@dataclass
class BufferConfig:
    buffer_idx: int
    dtype: DataType
    size: int
    executor_type: ExecutorType
    
    def to_dict(self) -> Dict:
        return {
            'buffer_idx': self.buffer_idx,
            'dtype': self.dtype.name,
            'size': self.size,
            'executor_type': self.executor_type.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BufferConfig':
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
    
    def to_dict(self) -> Dict:
        return {
            'executor_type': self.executor_type.name,
            'num_total_executors': self.num_total_executors
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExecutorConfig':
        return cls(
            executor_type=ExecutorType[data['executor_type']],
            num_total_executors=data['num_total_executors']
        )

@dataclass
class PrimitiveConfig:
    type: PrimitiveType
    dtype: DataType
    target_rank: int
    src_buffer_idx: int
    dst_buffer_idx: int
    compute_op: ComputeType
    executor_type: ExecutorType
    num_executors: int
    data_size: int
    signal_value: int
    num_dependencies: int
    followers: List[int] = field(default_factory=list)
    num_followers: int = 0
    
    def to_dict(self) -> Dict:
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
            'followers': self.followers,
            'num_followers': self.num_followers
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PrimitiveConfig':
        return cls(
            type=PrimitiveType[data['type']],
            dtype=DataType[data['dtype']],
            target_rank=data['target_rank'],
            src_buffer_idx=data['src_buffer_idx'],
            dst_buffer_idx=data['dst_buffer_idx'],
            compute_op=ComputeType[data['compute_op']],
            executor_type=ExecutorType[data['executor_type']],
            num_executors=data['num_executors'],
            data_size=data['data_size'],
            signal_value=data['signal_value'],
            num_dependencies=data['num_dependencies'],
            followers=data.get('followers', []),
            num_followers=data.get('num_followers', 0)
        )

@dataclass
class PrimitiveGraph:
    rank: int
    buffers: List[BufferConfig]
    executors: List[ExecutorConfig]
    operators: List[PrimitiveConfig]
    dependencies: List[tuple] = field(default_factory=list)  # (from_op_id, to_op_id)
    
    def __init__(self, rank: int = 0):
        self.rank = rank
        self.buffers = []
        self.executors = []
        self.operators = []
        self.dependencies = []
    
    def add_buffer(self, idx: int, dtype: DataType, size: int, executor_type: ExecutorType = ExecutorType.CPU):
        buffer_config = BufferConfig(idx, dtype, size, executor_type)
        self.buffers.append(buffer_config)
        return buffer_config
    
    def add_operator(self, op_config: PrimitiveConfig) -> int:
        self.operators.append(op_config)
        return len(self.operators) - 1
    
    def add_dependency(self, from_op_id: int, to_op_id: int):
        if 0 <= from_op_id < len(self.operators) and 0 <= to_op_id < len(self.operators):
            self.dependencies.append((from_op_id, to_op_id))
            self.operators[to_op_id].num_dependencies += 1
            if self.operators[from_op_id].num_followers < 8:  # 最大8个followers
                self.operators[from_op_id].followers.append(to_op_id)
                self.operators[from_op_id].num_followers += 1
    
    def to_dict(self) -> Dict:
        return {
            'rank': self.rank,
            'buffers': [buf.to_dict() for buf in self.buffers],
            'executors': [exec.to_dict() for exec in self.executors],
            'operators': [op.to_dict() for op in self.operators],
            'dependencies': self.dependencies
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PrimitiveGraph':
        graph = cls(data['rank'])
        graph.buffers = [BufferConfig.from_dict(buf) for buf in data['buffers']]
        graph.executors = [ExecutorConfig.from_dict(exec) for exec in data['executors']]
        graph.operators = [PrimitiveConfig.from_dict(op) for op in data['operators']]
        graph.dependencies = data.get('dependencies', [])
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PrimitiveGraph':
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_executor_types(self) -> List[ExecutorType]:
        return list(set(exec.executor_type for exec in self.executors))
    
    def validate(self) -> bool:
        if not self.buffers:
            return False
        
        for from_id, to_id in self.dependencies:
            if from_id >= len(self.operators) or to_id >= len(self.operators):
                return False
        
        return True
