from ir.pipeline import Pass, Diagnostic, DiagnosticLevel
from ir import *
from typing import List, Set, Tuple


class CanonicPass(Pass):
    def run(self, ir: CollectiveIR, diags: List[Diagnostic]) -> CollectiveIR:
        # 1. 重新生成规范化的TaskMap
        new_task_map = TaskMap(ir.collective_op, ir.data_size_gb)
        task_id_map = self._reindex_tasks(ir, new_task_map, diags)
        self._normalize_primitives(new_task_map, diags)
        self._fix_dependencies(new_task_map, task_id_map, diags)
        
        return CollectiveIR(ir.cluster, ir.collective_op, ir.data_size_gb, new_task_map)

    def _reindex_tasks(self, ir: CollectiveIR, new_task_map: TaskMap, diags: List[Diagnostic]) -> Dict[int, int]:
        """重新编号任务ID（连续、无间隙），删除空任务"""
        task_id_map = {}
        new_id = 0
        
        for old_task in sorted(ir.task_map.tasks.values(), key=lambda t: t.task_id):
            if not old_task.primitives:
                diags.append(Diagnostic(DiagnosticLevel.WARNING, f"删除空任务: {old_task.task_id}", loc="CanonicPass"))
                continue
            
            new_task = Task(new_id, old_task.primitives.copy(), old_task.dependencies.copy(), old_task.status)
            new_task_map.add_task(new_task)
            task_id_map[old_task.task_id] = new_id
            new_id += 1
        
        return task_id_map

    def _normalize_primitives(self, task_map: TaskMap, diags: List[Diagnostic]):
        """统一原语格式：合并重复项、排序内存区域"""
        for task in task_map.tasks.values():
            seen: Set[Tuple] = set()
            unique_primitives = []
            
            for prim in task.primitives:
                # 生成原语唯一标识（避免重复）
                key = (
                    prim.initiator.device_id,
                    prim.op_type.name,
                    tuple((type(m), m.device.device_id, m.address, m.size) for m in prim.memory_regions)
                )
                
                if key not in seen:
                    seen.add(key)
                    # 强制内存区域顺序：LocalMemory在前，RemoteMemory在后
                    local_mem = next(m for m in prim.memory_regions if isinstance(m, LocalMemory))
                    remote_mem = next(m for m in prim.memory_regions if isinstance(m, RemoteMemory))
                    normalized_prim = CommunicationPrimitive(prim.initiator, prim.op_type, [local_mem, remote_mem])
                    unique_primitives.append(normalized_prim)
            
            if len(unique_primitives) < len(task.primitives):
                diags.append(Diagnostic(DiagnosticLevel.NOTE, f"任务{task.task_id}合并了{len(task.primitives)-len(unique_primitives)}个重复原语", loc="CanonicPass"))
            
            task.primitives = unique_primitives

    def _fix_dependencies(self, task_map: TaskMap, task_id_map: Dict[int, int], diags: List[Diagnostic]):
        """修复依赖关系（替换为新任务ID、删除无效依赖）"""
        for task in task_map.tasks.values():
            valid_deps = []
            for dep in task.dependencies:
                if dep.task_id in task_id_map:
                    new_dep_id = task_id_map[dep.task_id]
                    valid_deps.append(task_map.tasks[new_dep_id])
                else:
                    diags.append(Diagnostic(DiagnosticLevel.WARNING, f"任务{task.task_id}的依赖{dep.task_id}无效，已删除", loc="CanonicPass"))
            task.dependencies = valid_deps
