from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from ..core.ir import CommunicationPrimitive, Task, Device
from ..core.enums import PrimitiveOpType

class BandwidthModel(ABC):
    @abstractmethod
    def get_bandwidth(self, src_device: Device, dst_device: Device, message_size: int) -> float:
        pass

class LatencyModel(ABC):
    @abstractmethod
    def get_latency(self, src_device: Device, dst_device: Device) -> float:
        pass

class PerformanceModel(ABC):
    def __init__(self, bandwidth_model: BandwidthModel, latency_model: LatencyModel):
        self.bandwidth_model = bandwidth_model
        self.latency_model = latency_model
    
    @abstractmethod
    def estimate_primitive_time(self, primitive: CommunicationPrimitive) -> float:
        pass
    
    @abstractmethod
    def estimate_task_time(self, task: Task) -> float:
        pass
    
    def estimate_communication_time(self, src_device: Device, dst_device: Device, 
                                  data_size_bytes: int) -> float:
        if data_size_bytes <= 0:
            return 0.0
        
        bandwidth_gbs = self.bandwidth_model.get_bandwidth(src_device, dst_device, data_size_bytes)
        latency_ms = self.latency_model.get_latency(src_device, dst_device)
        
        if bandwidth_gbs <= 0:
            return float('inf')

        communication_time_ms = latency_ms + (data_size_bytes / (bandwidth_gbs * 1024 * 1024 * 1024)) * 1000
        return communication_time_ms
