from typing import Dict, List
from ...core.ir import CollectiveIR, CommunicationPrimitive, LocalMemory
from ...core.enums import CollectiveOpType, PrimitiveOpType
from ..base import IRPass

class MemoryOptimizationPass(IRPass):
    @property
    def name(self) -> str:
        return "MemoryOptimizationPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        ir = self._optimize_memory_layout(ir)
        ir = self._reduce_memory_footprint(ir)
        return ir
    
    def _optimize_memory_layout(self, ir: CollectiveIR) -> CollectiveIR:
        device_memory_usage = {}
        
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                for mem_region in primitive.memory_regions:
                    device_id = mem_region.device.device_id
                    if device_id not in device_memory_usage:
                        device_memory_usage[device_id] = []
                    device_memory_usage[device_id].append(mem_region)
        
        for device_id, regions in device_memory_usage.items():
            regions.sort(key=lambda r: r.address)
            current_address = 0
            
            for region in regions:
                if isinstance(region, LocalMemory):
                    region.address = current_address
                    current_address += region.size
        
        return ir
    
    def _reduce_memory_footprint(self, ir: CollectiveIR) -> CollectiveIR:
        memory_reuse_map = {}
        
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                for mem_region in primitive.memory_regions:
                    if mem_region.size > 0:
                        key = (mem_region.device.device_id, mem_region.size, id(primitive))
                        if key in memory_reuse_map:
                            existing_region = memory_reuse_map[key]
                            if existing_region.address + existing_region.size <= mem_region.address:
                                mem_region.address = existing_region.address
        
        return ir
  