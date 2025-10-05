from typing import Dict, Set, List
from ..core.ir import Task, TaskMap, CollectiveIR
from .analyzer import DependencyAnalyzer

class DependencyOptimizer:
    def __init__(self, ir: CollectiveIR):
        self.ir = ir
        self.analyzer = DependencyAnalyzer(ir.task_map)
    
    def optimize_task_dependencies(self) -> CollectiveIR:
        optimized_deps = self.analyzer.optimize_dependencies()

        for task_id, dep_ids in optimized_deps.items():
            task = self.ir.task_map.tasks[task_id]

            task.dependencies.clear()

            for dep_id in dep_ids:
                if dep_id in self.ir.task_map.tasks:
                    task.dependencies.append(self.ir.task_map.tasks[dep_id])
        
        return self.ir
    
    def identify_parallel_execution_groups(self) -> List[Set[int]]:
        return self.analyzer.parallel_groups
    
    def get_critical_path_tasks(self) -> Set[int]:
        return self.analyzer.critical_path
    
    def optimize_for_parallelism(self) -> CollectiveIR:
        parallel_groups = self.identify_parallel_execution_groups()

        for group in parallel_groups:
            self._optimize_parallel_group(group)
        
        return self.ir
    
    def _optimize_parallel_group(self, group: Set[int]):
        for task_id in group:
            task = self.ir.task_map.tasks[task_id]

            dependencies_to_remove = []
            for dep in task.dependencies:
                if dep.task_id in group:
                    if not self._is_dependency_necessary(task_id, dep.task_id):
                        dependencies_to_remove.append(dep)
            
            for dep in dependencies_to_remove:
                task.dependencies.remove(dep)
    
    def _is_dependency_necessary(self, task_id: int, dep_id: int) -> bool:

        other_paths = False
        
        def has_alternative_path(current: int, target: int, visited: Set[int]) -> bool:
            if current == target:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            for next_dep in self.analyzer.dependency_graph.get(current, set()):
                if next_dep != dep_id and has_alternative_path(next_dep, target, visited.copy()):
                    return True
            return False

        for indirect_dep in self.analyzer.dependency_graph.get(dep_id, set()):
            if indirect_dep != task_id and has_alternative_path(indirect_dep, task_id, set()):
                other_paths = True
                break
        
        return not other_paths
