from typing import Dict, List, Set
from ..core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory, RemoteMemory
from ..core.enums import PrimitiveOpType, MemoryType
from .manager import MemoryManager
from ..passes.base import IRPass

class AdvancedMemoryOptimizer(IRPass):
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.memory_reuse_strategies = {
            PrimitiveOpType.COPY: self._optimize_copy_memory,
            PrimitiveOpType.REDUCE: self._optimize_reduce_memory,
            PrimitiveOpType.WRITE: self._optimize_write_memory,
        }
    
    @property
    def name(self) -> str:
        return "AdvancedMemoryOptimizer"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        self._initialize_device_memory(ir)
        ir = self._optimize_memory_layout(ir)
        ir = self._apply_memory_reuse(ir)
        self.memory_manager.optimize_memory_layout()
        
        return ir
    
    def _initialize_device_memory(self, ir: CollectiveIR):
        for device in ir.cluster.devices_by_id.values():
            memory_capacity = getattr(device, 'memory_capacity_gb', 16.0)  # 默认16GB
            pool_size = int(memory_capacity * 1024 * 1024 * 1024)  # 转换为字节
            self.memory_manager.initialize_device_memory(device, pool_size)
    
    def _optimize_memory_layout(self, ir: CollectiveIR) -> CollectiveIR:
        device_memory_regions = {}
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                for mem_region in primitive.memory_regions:
                    device_id = mem_region.device.device_id
                    if device_id not in device_memory_regions:
                        device_memory_regions[device_id] = []
                    device_memory_regions[device_id].append(mem_region)
        for device_id, regions in device_memory_regions.items():
            device = ir.cluster.get_device(device_id)
            self._optimize_device_memory_layout(device, regions)
        
        return ir
    
    def _optimize_device_memory_layout(self, device, regions: List):
        regions.sort(key=lambda r: r.size, reverse=True)
        
        current_address = 0
        alignment = 64
        
        for region in regions:
            if isinstance(region, LocalMemory):
                if current_address % alignment != 0:
                    current_address += alignment - (current_address % alignment)
                
                allocated_region = self.memory_manager.allocate_memory(
                    device, region.size, alignment, region.memory_type, region.access
                )
                
                if allocated_region:
                    region.address = allocated_region.address
                    current_address = region.address + region.size
    
    def _apply_memory_reuse(self, ir: CollectiveIR) -> CollectiveIR:
        memory_access_patterns = self._analyze_memory_access_patterns(ir)
        
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                strategy = self.memory_reuse_strategies.get(primitive.op_type)
                if strategy:
                    strategy(primitive, memory_access_patterns)
        
        return ir
    
    def _analyze_memory_access_patterns(self, ir: CollectiveIR) -> Dict:
        patterns = {
            'read_after_write': set(),
            'write_after_read': set(),
            'concurrent_access': set(),
        }
        
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                for mem_region in primitive.memory_regions:
                    key = (mem_region.device.device_id, mem_region.address)
                    
                    if primitive.op_type in [PrimitiveOpType.WRITE, PrimitiveOpType.REDUCE]:
                        patterns['write_after_read'].add(key)
                    elif primitive.op_type == PrimitiveOpType.COPY:
                        patterns['read_after_write'].add(key)
        
        return patterns
    
    def _optimize_copy_memory(self, primitive: CommunicationPrimitive, patterns: Dict):
        local_memories = [m for m in primitive.memory_regions if isinstance(m, LocalMemory)]
        remote_memories = [m for m in primitive.memory_regions if isinstance(m, RemoteMemory)]
        
        if local_memories and remote_memories:
            local_mem = local_memories[0]
            remote_mem = remote_memories[0]

            reusable_local = self.memory_manager.find_reusable_memory(
                local_mem.device, local_mem.size
            )
            if reusable_local:
                local_mem.address = reusable_local.address

    
    def _optimize_reduce_memory(self, primitive: CommunicationPrimitive, patterns: Dict):
        local_memories = [m for m in primitive.memory_regions if isinstance(m, LocalMemory)]
        
        for local_mem in local_memories:
            if local_mem.size > 1024 * 1024:
                reusable_mem = self.memory_manager.find_reusable_memory(
                    local_mem.device, local_mem.size
                )
                if reusable_mem:
                    local_mem.address = reusable_mem.address
                    local_mem.memory_type = MemoryType.HBM 
    
    def _optimize_write_memory(self, primitive: CommunicationPrimitive, patterns: Dict):
        local_memories = [m for m in primitive.memory_regions if isinstance(m, LocalMemory)]
        
        for local_mem in local_memories:
            if local_mem.size <= 64 * 1024: 
                reusable_mem = self.memory_manager.find_reusable_memory(
                    local_mem.device, local_mem.size
                )
                if reusable_mem:
                    local_mem.address = reusable_mem.address
