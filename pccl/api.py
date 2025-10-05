from typing import List, Dict, Any, Optional
from .collective_ir.specs.factories import (
    create_allreduce_spec, create_broadcast_spec, create_allgather_spec,
    create_alltoall_spec, create_reducescatter_spec,
    create_simple_allreduce_ir, create_simple_broadcast_ir,
    create_simple_allgather_ir, create_simple_alltoall_ir,
    create_simple_reducescatter_ir
)
from .collective_ir.core.enums import CollectiveOpType
from .pipeline import CollectiveExecutionPipeline, PipelineConfig, ExecutionResult

class CollectiveAPI:
    def __init__(self, pipeline: Optional[CollectiveExecutionPipeline] = None):
        self.pipeline = pipeline or CollectiveExecutionPipeline()
    
    def allreduce(self, device_ids: List[int], data_size_gb: float = 1.0) -> ExecutionResult:
        collective_ir = create_simple_allreduce_ir(device_ids, data_size_gb)
        return self.pipeline.execute(collective_ir)
    
    def broadcast(self, root_device: int, device_ids: List[int], data_size_gb: float = 1.0) -> ExecutionResult:
        collective_ir = create_simple_broadcast_ir(root_device, device_ids, data_size_gb)
        return self.pipeline.execute(collective_ir)
    
    def allgather(self, device_ids: List[int], data_size_gb: float = 1.0) -> ExecutionResult:
        collective_ir = create_simple_allgather_ir(device_ids, data_size_gb)
        return self.pipeline.execute(collective_ir)
    
    def alltoall(self, device_ids: List[int], data_size_gb: float = 1.0) -> ExecutionResult:
        collective_ir = create_simple_alltoall_ir(device_ids, data_size_gb)
        return self.pipeline.execute(collective_ir)
    
    def reducescatter(self, device_ids: List[int], data_size_gb: float = 1.0) -> ExecutionResult:
        collective_ir = create_simple_reducescatter_ir(device_ids, data_size_gb)
        return self.pipeline.execute(collective_ir)
    
    def execute_custom(self, collective_ir) -> ExecutionResult:
        return self.pipeline.execute(collective_ir)
    
    def shutdown(self):
        self.pipeline.shutdown()

def create_collective_api(optimization_level: str = "standard") -> CollectiveAPI:
    from .pipeline import PipelineConfig, OptimizationLevel
    
    level_map = {
        "basic": OptimizationLevel.BASIC,
        "standard": OptimizationLevel.STANDARD,
        "advanced": OptimizationLevel.ADVANCED,
        "aggressive": OptimizationLevel.AGGRESSIVE
    }
    
    config = PipelineConfig(optimization_level=level_map.get(optimization_level, OptimizationLevel.STANDARD))
    pipeline = CollectiveExecutionPipeline(config)
    return CollectiveAPI(pipeline)
