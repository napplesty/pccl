from typing import Dict, Tuple
from .base import BandwidthModel
from ..core.ir import Device

class GPUBandwidthModel(BandwidthModel):
    def __init__(self):
        self.nominal_bandwidths = {
            ("A100", "A100"): 600.0,  # NVLink 3.0
            ("H100", "H100"): 900.0,  # NVLink 4.0
            ("H20", "H20"): 1200.0,   # NVLink 4.0
            ("V100", "V100"): 300.0,  # NVLink 2.0
            
            ("PCIe5", "PCIe5"): 64.0,  # PCIe 5.0 x16
            ("PCIe4", "PCIe4"): 32.0,  # PCIe 4.0 x16
            ("PCIe3", "PCIe3"): 16.0,  # PCIe 3.0 x16
            
            ("default", "default"): 25.0
        }
        
        self.efficiency_factors = {
            "NVLink": 0.99,
            "PCIe": 0.75,  
            "Network": 0.99, 
        }
        
        self.size_efficiency = {
            (0, 1024): 0.9,
            (1024, 64*1024): 0.9,
            (64*1024, 1*1024*1024): 0.9,
            (1*1024*1024, float('inf')): 0.99
        }
    
    def _get_device_pair_key(self, src_device: Device, dst_device: Device) -> Tuple[str, str]:
        src_type = src_device.type if src_device.type else "default"
        dst_type = dst_device.type if dst_device.type else "default"
        return (src_type, dst_type)
    
    def _get_connection_type(self, src_device: Device, dst_device: Device) -> str:
        if (src_device.type in ["A100", "H100", "V100"] and 
            dst_device.type in ["A100", "H100", "V100"]):
            return "NVLink"
        elif src_device.properties.get("host_id") == dst_device.properties.get("host_id"):
            return "PCIe"
        else:
            return "Network"
    
    def _get_size_efficiency(self, message_size: int) -> float:
        for (min_size, max_size), efficiency in self.size_efficiency.items():
            if min_size <= message_size < max_size:
                return efficiency
        return 0.8  # 默认效率
    
    def get_bandwidth(self, src_device: Device, dst_device: Device, message_size: int) -> float:
        # 获取名义带宽
        pair_key = self._get_device_pair_key(src_device, dst_device)
        nominal_bw = self.nominal_bandwidths.get(pair_key, 25.0)
        if hasattr(src_device, 'bandwidth_gbs') and src_device.bandwidth_gbs > 0:
            nominal_bw = min(nominal_bw, src_device.bandwidth_gbs)
        if hasattr(dst_device, 'bandwidth_gbs') and dst_device.bandwidth_gbs > 0:
            nominal_bw = min(nominal_bw, dst_device.bandwidth_gbs)
        connection_type = self._get_connection_type(src_device, dst_device)
        efficiency = self.efficiency_factors.get(connection_type, 0.7)
        size_efficiency = self._get_size_efficiency(message_size)
        effective_bandwidth = nominal_bw * efficiency * size_efficiency
        return effective_bandwidth
