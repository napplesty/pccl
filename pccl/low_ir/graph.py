from typing import List, Dict, Optional, Any
import json

from .core import LowLevelIR, PrimitiveConfig, BufferConfig, ExecutorConfig
from .types import PrimitiveType, DataType, ComputeType, ExecutorType

class PrimitiveGraph:
    def __init__(self, rank: int = 0):
        self.rank = rank
        self.buffers: Dict[int, BufferConfig] = {}
        self.operators: List[PrimitiveConfig] = []
        self.dependencies: Dict[int, List[int]] = {}
    
    def add_buffer(self, idx: int, dtype: DataType, size: int, executor_type: ExecutorType):
        buffer = BufferConfig(idx, dtype, size, executor_type)
        self.buffers[idx] = buffer
    
    def add_operator(self, op: PrimitiveConfig):
        self.operators.append(op)
    
    def add_dependency(self, from_op_id: int, to_op_id: int):
        if to_op_id not in self.dependencies:
            self.dependencies[to_op_id] = []
        self.dependencies[to_op_id].append(from_op_id)
    
    def get_rank(self) -> int:
        return self.rank
    
    def get_buffers(self) -> List[BufferConfig]:
        return list(self.buffers.values())
    
    def get_operators(self) -> List[PrimitiveConfig]:
        return self.operators
    
    def get_executors(self) -> List[ExecutorConfig]:
        return [ExecutorConfig(ExecutorType.CUDA, 8), ExecutorConfig(ExecutorType.CPU, 1)]
    
    def to_json(self) -> str:
        data = {
            'rank': self.rank,
            'buffers': [buffer.to_dict() for buffer in self.buffers.values()],
            'operators': [op.to_dict() for op in self.operators],
            'dependencies': self.dependencies
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_data: str) -> 'PrimitiveGraph':
        data = json.loads(json_data)
        graph = cls(data.get('rank', 0))
        
        for buffer_data in data.get('buffers', []):
            buffer = BufferConfig.from_dict(buffer_data)
            graph.buffers[buffer.buffer_idx] = buffer
        
        for op_data in data.get('operators', []):
            op = PrimitiveConfig.from_dict(op_data)
            graph.operators.append(op)
        
        graph.dependencies = data.get('dependencies', {})
        
        return graph
    
    @classmethod
    def load_from_file(cls, filename: str, rank: int) -> 'PrimitiveGraph':
        with open(filename, 'r') as f:
            json_data = f.read()
        return cls.from_json(json_data)
    
    @classmethod
    def load_from_json(cls, json_data: str) -> 'PrimitiveGraph':
        return cls.from_json(json_data)
