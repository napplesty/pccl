from typing import Dict, List, Optional, Tuple
from ..core.ir import Device, LocalMemory
from .allocator import SmartMemoryAllocator

class MemoryPool:
    def __init__(self, device: Device, total_size: int):
        self.device = device
        self.total_size = total_size
        self.allocator = SmartMemoryAllocator(total_size)
        self.allocated_regions = {} 
    
    def allocate(self, size: int, alignment: int = 64) -> Optional[int]:
        reusable = self.allocator.find_reusable_region(size, alignment)
        if reusable:
            address = reusable.address
            if address in self.allocated_regions:
                old_size, ref_count = self.allocated_regions[address]
                self.allocated_regions[address] = (size, ref_count + 1)
            else:
                self.allocated_regions[address] = (size, 1)
            return address

        address = self.allocator.allocate(size, alignment)
        if address is not None:
            self.allocated_regions[address] = (size, 1)
        
        return address
    
    def deallocate(self, address: int, size: int):
        if address in self.allocated_regions:
            current_size, ref_count = self.allocated_regions[address]
            
            if ref_count > 1:
                self.allocated_regions[address] = (current_size, ref_count - 1)
            else:
                self.allocator.deallocate(address, current_size)
                del self.allocated_regions[address]
    
    def find_reusable_region(self, size: int, alignment: int = 64) -> Optional[LocalMemory]:
        address = self.allocator.find_reusable_region(size, alignment)
        if address:
            return LocalMemory(self.device, address, size)
        return None
    
    def defragment(self):
        if self.allocator.get_fragmentation() > 0.5:  # 碎片率超过50%

            current_allocations = list(self.allocated_regions.items())

            self.allocator = SmartMemoryAllocator(self.total_size)
            self.allocated_regions.clear()

            for address, (size, ref_count) in current_allocations:
                new_address = self.allocate(size, 64)
                if new_address is None:
                    self.allocator = SmartMemoryAllocator(self.total_size)
                    for addr, (sz, rc) in current_allocations:
                        self.allocated_regions[addr] = (sz, rc)
                    break
    
    def clear(self):
        self.allocator = SmartMemoryAllocator(self.total_size)
        self.allocated_regions.clear()
    
    def get_usage(self) -> Tuple[int, int]:
        return self.allocator.get_usage()
    
    def get_fragmentation(self) -> float:
        return self.allocator.get_fragmentation()
