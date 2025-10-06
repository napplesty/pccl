from enum import Enum, IntEnum
from typing import Dict, Any

class OptimizationLevel(IntEnum):
    BASIC = 0
    STANDARD = 1
    ADVANCED = 2
    AGGRESSIVE = 3

class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINED = "pipelined"

class PipelineConfig:
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                 execution_strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL,
                 enable_verification: bool = True,
                 enable_profiling: bool = False,
                 timeout_ms: int = 30000):
        self.optimization_level = optimization_level
        self.execution_strategy = execution_strategy
        self.enable_verification = enable_verification
        self.enable_profiling = enable_profiling
        self.timeout_ms = timeout_ms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimization_level': self.optimization_level.value,
            'execution_strategy': self.execution_strategy.value,
            'enable_verification': self.enable_verification,
            'enable_profiling': self.enable_profiling,
            'timeout_ms': self.timeout_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        return cls(
            optimization_level=OptimizationLevel(data.get('optimization_level', 1)),
            execution_strategy=ExecutionStrategy(data.get('execution_strategy', 'parallel')),
            enable_verification=data.get('enable_verification', True),
            enable_profiling=data.get('enable_profiling', False),
            timeout_ms=data.get('timeout_ms', 30000)
        )
