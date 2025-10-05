from .base import PerformanceModel
from ..core.ir import CommunicationPrimitive, Task, LocalMemory, RemoteMemory
from ..core.enums import PrimitiveOpType, MemoryType

class MemoryBandwidthModel:
    def __init__(self):
        self.memory_bandwidths = {
            MemoryType.HBM: 1500.0,
            MemoryType.DRAM: 50.0,
            MemoryType.SHARED: 50.0,
        }
    
    def get_memory_bandwidth(self, memory_type: MemoryType) -> float:
        return self.memory_bandwidths.get(memory_type, 50.0)
    
    def estimate_memory_time(self, memory_type: MemoryType, data_size_bytes: int) -> float:
        bandwidth_gbs = self.get_memory_bandwidth(memory_type)
        
        if bandwidth_gbs <= 0:
            return float('inf')
        
        memory_time_ms = (data_size_bytes / (bandwidth_gbs * 1024 * 1024 * 1024)) * 1000
        return memory_time_ms
