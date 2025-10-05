from .base import LatencyModel
from ..core.ir import Device

class SimpleLatencyModel(LatencyModel):
    def __init__(self):
        self.base_latencies = {
            "NVLink": 0.1,    # NVLink延迟 (us)
            "PCIe": 0.1,      # PCIe延迟
            "Network": 1.0,  # 网络延迟
        }
    
    def _get_connection_type(self, src_device: Device, dst_device: Device) -> str:
        if (src_device.type in ["A100", "H100", "V100"] and 
            dst_device.type in ["A100", "H100", "V100"]):
            return "NVLink"
        elif src_device.properties.get("host_id") == dst_device.properties.get("host_id"):
            return "PCIe"
        else:
            return "Network"
    
    def get_latency(self, src_device: Device, dst_device: Device) -> float:
        connection_type = self._get_connection_type(src_device, dst_device)
        base_latency_us = self.base_latencies.get(connection_type, 5.0)

        return base_latency_us / 1000.0

class TopologyAwareLatencyModel(LatencyModel):
    def __init__(self, cluster_topology):
        self.topology = cluster_topology
        self.hop_latency = 0.001
    
    def get_latency(self, src_device: Device, dst_device: Device) -> float:
        src_host = src_device.properties.get("host_id", -1)
        dst_host = dst_device.properties.get("host_id", -1)
        
        if src_host == dst_host:
            return 0.0001
        else:
            return 0.001
