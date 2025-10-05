from typing import Dict
from .base import IRPass
from ..core.ir import CollectiveIR, CommunicationPrimitive, LocalMemory, RemoteMemory, Task
from ..core.enums import PrimitiveOpType
from ..performance_modeling import (
    GPUBandwidthModel, SimpleLatencyModel, 
    ComputationCostModel, MemoryBandwidthModel,
    PerformanceModel
)

class AccuratePerformanceModel(PerformanceModel):
    def __init__(self):
        bandwidth_model = GPUBandwidthModel()
        latency_model = SimpleLatencyModel()
        super().__init__(bandwidth_model, latency_model)
        self.computation_model = ComputationCostModel()
        self.memory_model = MemoryBandwidthModel()
    
    def estimate_primitive_time(self, primitive: CommunicationPrimitive) -> float:
        total_time = 0.0

        local_memories = [m for m in primitive.memory_regions if isinstance(m, LocalMemory)]
        remote_memories = [m for m in primitive.memory_regions if isinstance(m, RemoteMemory)]
        
        if primitive.op_type == PrimitiveOpType.COPY:
            if local_memories and remote_memories:
                local_mem = local_memories[0]
                remote_mem = remote_memories[0]

                comm_time = self.estimate_communication_time(
                    primitive.initiator, remote_mem.device, local_mem.size
                )
                total_time += comm_time

                local_mem_time = self.memory_model.estimate_memory_time(
                    local_mem.memory_type, local_mem.size
                )
                remote_mem_time = self.memory_model.estimate_memory_time(
                    remote_mem.memory_type, remote_mem.size
                )
                total_time += max(local_mem_time, remote_mem_time)
        
        elif primitive.op_type == PrimitiveOpType.REDUCE:
            if local_memories and remote_memories:
                local_mem = local_memories[0]
                remote_mem = remote_memories[0]

                comm_time = self.estimate_communication_time(
                    primitive.initiator, remote_mem.device, remote_mem.size
                )
                total_time += comm_time
                comp_time = self.computation_model.estimate_computation_time(
                    PrimitiveOpType.REDUCE, remote_mem.size
                )
                total_time += comp_time

                mem_time = self.memory_model.estimate_memory_time(
                    local_mem.memory_type, local_mem.size
                )
                total_time += mem_time
        
        elif primitive.op_type in [PrimitiveOpType.NOTIFY, PrimitiveOpType.GET_NOTIFIED]:
            if remote_memories:
                remote_mem = remote_memories[0]
                latency = self.latency_model.get_latency(primitive.initiator, remote_mem.device)
                total_time += latency
        
        else:
            total_mem_size = sum(m.size for m in primitive.memory_regions)
            if total_mem_size > 0:
                max_mem_time = max(
                    self.memory_model.estimate_memory_time(m.memory_type, m.size)
                    for m in primitive.memory_regions
                )
                total_time += max_mem_time
        
        return max(total_time, 0.1)  # 最小0.1ms
    
    def estimate_task_time(self, task: Task) -> float:
        total_time = 0.0
        for primitive in task.primitives:
            primitive_time = self.estimate_primitive_time(primitive)
            total_time += primitive_time
        return total_time

class PerformanceModelingPass(IRPass):
    def __init__(self):
        self.performance_model = AccuratePerformanceModel()
    
    @property
    def name(self) -> str:
        return "PerformanceModelingPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                estimated_time = self.performance_model.estimate_primitive_time(primitive)
                primitive.estimated_duration_ms = estimated_time
            task_total_time = self.performance_model.estimate_task_time(task)
        return ir
