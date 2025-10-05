from typing import Dict, List, Tuple, Optional
from ..core.ir import Device, LocalMemory
from ..core.enums import MemoryType, AccessPermission

class MemoryBlock:
    def __init__(self, start: int, size: int, allocated: bool = False):
        self.start = start
        self.size = size
        self.allocated = allocated
        self.next = None

class SmartMemoryAllocator:

    def __init__(self, total_size: int):
        self.total_size = total_size
        self.free_blocks = [MemoryBlock(0, total_size)]
        self.allocated_blocks = {} 
        self.allocations = 0
        self.deallocations = 0
    
    def allocate(self, size: int, alignment: int = 64) -> Optional[int]:
        if size <= 0:
            return None
        
        for i, block in enumerate(self.free_blocks):
            if not block.allocated and block.size >= size:
                aligned_start = self._align_address(block.start, alignment)
                aligned_end = aligned_start + size
                
                if aligned_end <= block.start + block.size:
                    remaining_size = block.size - (aligned_start - block.start) - size
                    if aligned_start > block.start:
                        front_block = MemoryBlock(block.start, aligned_start - block.start)
                        allocated_block = MemoryBlock(aligned_start, size, True)
                        
                        if remaining_size > 0:
                            back_block = MemoryBlock(aligned_end, remaining_size)
                            self.free_blocks[i] = front_block
                            self.free_blocks.insert(i + 1, allocated_block)
                            self.free_blocks.insert(i + 2, back_block)
                        else:
                            self.free_blocks[i] = front_block
                            self.free_blocks.insert(i + 1, allocated_block)
                    else:
                        allocated_block = MemoryBlock(aligned_start, size, True)
                        
                        if remaining_size > 0:
                            back_block = MemoryBlock(aligned_end, remaining_size)
                            self.free_blocks[i] = allocated_block
                            self.free_blocks.insert(i + 1, back_block)
                        else:
                            self.free_blocks[i] = allocated_block
                    
                    self.allocated_blocks[aligned_start] = allocated_block
                    self.allocations += 1

                    self._merge_free_blocks()
                    
                    return aligned_start
        
        return None
    
    def deallocate(self, address: int, size: int):
        if address in self.allocated_blocks:
            block = self.allocated_blocks[address]
            block.allocated = False

            for i, free_block in enumerate(self.free_blocks):
                if free_block.start > block.start:
                    self.free_blocks.insert(i, block)
                    break
            else:
                self.free_blocks.append(block)
            
            del self.allocated_blocks[address]
            self.deallocations += 1

            self._merge_free_blocks()
    
    def _align_address(self, address: int, alignment: int) -> int:
        if address % alignment == 0:
            return address
        return address + (alignment - (address % alignment))
    
    def _merge_free_blocks(self):
        if not self.free_blocks:
            return

        self.free_blocks.sort(key=lambda b: b.start)
        
        i = 0
        while i < len(self.free_blocks) - 1:
            current = self.free_blocks[i]
            next_block = self.free_blocks[i + 1]
            
            if not current.allocated and not next_block.allocated:
                if current.start + current.size == next_block.start:
                    current.size += next_block.size
                    self.free_blocks.pop(i + 1)
                else:
                    i += 1
            else:
                i += 1
    
    def find_reusable_region(self, size: int, alignment: int = 64) -> Optional[LocalMemory]:
        for block in self.free_blocks:
            if not block.allocated and block.size >= size:
                aligned_start = self._align_address(block.start, alignment)
                if aligned_start + size <= block.start + block.size:
                    return LocalMemory(None, aligned_start, size)
        
        return None
    
    def get_fragmentation(self) -> float:
        if not self.free_blocks:
            return 0.0
        
        total_free = sum(block.size for block in self.free_blocks if not block.allocated)
        if total_free == 0:
            return 0.0
        
        largest_free = max((block.size for block in self.free_blocks if not block.allocated), default=0)
        return 1.0 - (largest_free / total_free) if total_free > 0 else 0.0
    
    def get_usage(self) -> Tuple[int, int]:
        allocated_size = sum(block.size for block in self.allocated_blocks.values())
        return (allocated_size, self.total_size)
