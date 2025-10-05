from typing import Dict, List, Set, Tuple, Optional
from ..core.ir import Device, LocalMemory, RemoteMemory, MemoryRegion
from ..core.enums import MemoryType, AccessPermission

class MemoryManager:
    def __init__(self):
        self.device_memory_pools = {}  # device_id -> MemoryPool
        self.memory_registry = {}  # (device_id, address) -> MemoryRegion
        self.allocated_regions = set()  # 已分配的内存区域
    
    def initialize_device_memory(self, device: Device, total_size: int):
        from .pool import MemoryPool
        self.device_memory_pools[device.device_id] = MemoryPool(device, total_size)
    
    def allocate_memory(self, device: Device, size: int, alignment: int = 64,
                       memory_type: MemoryType = MemoryType.DRAM,
                       access: AccessPermission = AccessPermission.READ_WRITE) -> Optional[LocalMemory]:
        if device.device_id not in self.device_memory_pools:
            self.initialize_device_memory(device, 16 * 1024 * 1024 * 1024)  # 16GB默认
        
        pool = self.device_memory_pools[device.device_id]
        address = pool.allocate(size, alignment)
        
        if address is None:
            return None
        
        memory_region = LocalMemory(device, address, size, memory_type, access)
        
        self.memory_registry[(device.device_id, address)] = memory_region
        self.allocated_regions.add((device.device_id, address))
        
        return memory_region
    
    def deallocate_memory(self, memory_region: MemoryRegion):
        key = (memory_region.device.device_id, memory_region.address)
        
        if key in self.memory_registry:
            pool = self.device_memory_pools.get(memory_region.device.device_id)
            if pool:
                pool.deallocate(memory_region.address, memory_region.size)
            
            del self.memory_registry[key]
            self.allocated_regions.discard(key)
    
    def find_reusable_memory(self, device: Device, size: int, alignment: int = 64) -> Optional[LocalMemory]:
        if device.device_id not in self.device_memory_pools:
            return None
        
        pool = self.device_memory_pools[device.device_id]
        return pool.find_reusable_region(size, alignment)
    
    def get_memory_usage(self, device: Device) -> Tuple[int, int]:
        if device.device_id not in self.device_memory_pools:
            return (0, 0)
        
        pool = self.device_memory_pools[device.device_id]
        return pool.get_usage()
    
    def optimize_memory_layout(self):
        for pool in self.device_memory_pools.values():
            pool.defragment()
    
    def clear(self):
        for pool in self.device_memory_pools.values():
            pool.clear()
        self.memory_registry.clear()
        self.allocated_regions.clear()
