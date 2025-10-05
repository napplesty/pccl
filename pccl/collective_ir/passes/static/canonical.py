from typing import Dict, List, Set
from ...core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory
from ...core.enums import PrimitiveOpType
from ..base import IRPass

class CanonicalPass(IRPass):
    @property
    def name(self) -> str:
        return "CanonicalPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        ir = self._remove_empty_tasks(ir)
        ir = self._sort_tasks_by_id(ir)
        ir = self._merge_redundant_primitives(ir)
        ir = self._optimize_dependencies(ir)
        return ir
    
    def _remove_empty_tasks(self, ir: CollectiveIR) -> CollectiveIR:
        tasks_to_remove = []
        
        for task_id, task in ir.task_map.tasks.items():
            if not task.primitives:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del ir.task_map.tasks[task_id]
            
            for other_task in ir.task_map.tasks.values():
                other_task.dependencies = [dep for dep in other_task.dependencies if dep.task_id != task_id]
        
        return ir
    
    def _sort_tasks_by_id(self, ir: CollectiveIR) -> CollectiveIR:
        sorted_tasks = dict(sorted(ir.task_map.tasks.items()))
        ir.task_map.tasks = sorted_tasks
        return ir
    
    def _merge_redundant_primitives(self, ir: CollectiveIR) -> CollectiveIR:
        for task in ir.task_map.tasks.values():
            i = 0
            while i < len(task.primitives) - 1:
                current = task.primitives[i]
                next_prim = task.primitives[i + 1]
                
                if self._can_merge_primitives(current, next_prim):
                    merged = self._merge_primitives(current, next_prim)
                    task.primitives[i] = merged
                    task.primitives.pop(i + 1)
                else:
                    i += 1
        
        return ir
    
    def _can_merge_primitives(self, prim1: CommunicationPrimitive, prim2: CommunicationPrimitive) -> bool:
        if prim1.initiator != prim2.initiator:
            return False
        
        if prim1.op_type != prim2.op_type:
            return False
        
        if len(prim1.memory_regions) != len(prim2.memory_regions):
            return False
        
        for reg1, reg2 in zip(prim1.memory_regions, prim2.memory_regions):
            if reg1.device != reg2.device:
                return False
            if reg1.address + reg1.size != reg2.address:
                return False
        
        return True
    
    def _merge_primitives(self, prim1: CommunicationPrimitive, prim2: CommunicationPrimitive) -> CommunicationPrimitive:
        merged_regions = []
        
        for reg1, reg2 in zip(prim1.memory_regions, prim2.memory_regions):
            merged_size = reg1.size + reg2.size
            merged_region = type(reg1)(
                device=reg1.device,
                address=reg1.address,
                size=merged_size,
                memory_type=reg1.memory_type,
                access=reg1.access
            )
            merged_regions.append(merged_region)
        
        merged_primitive = CommunicationPrimitive(
            initiator=prim1.initiator,
            op_type=prim1.op_type,
            memory_regions=merged_regions,
            estimated_duration_ms=prim1.estimated_duration_ms + prim2.estimated_duration_ms
        )
        
        merged_primitive.chunk_updates = {**prim1.chunk_updates, **prim2.chunk_updates}
        
        return merged_primitive
    
    def _optimize_dependencies(self, ir: CollectiveIR) -> CollectiveIR:
        for task in ir.task_map.tasks.values():
            direct_deps = set(dep.task_id for dep in task.dependencies)
            transitive_deps = set()
            
            for dep in task.dependencies:
                transitive_deps.update(self._get_transitive_dependencies(dep))
            
            task.dependencies = [
                dep for dep in task.dependencies 
                if dep.task_id not in transitive_deps or dep.task_id in direct_deps
            ]
        
        return ir
    
    def _get_transitive_dependencies(self, task: Task) -> Set[int]:
        transitive = set()
        visited = set()
        
        def collect_deps(t: Task):
            if t.task_id in visited:
                return
            visited.add(t.task_id)
            
            for dep in t.dependencies:
                transitive.add(dep.task_id)
                collect_deps(dep)
        
        collect_deps(task)
        return transitive
