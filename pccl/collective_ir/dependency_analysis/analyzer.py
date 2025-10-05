from typing import Dict, Set, List, Tuple
from ..core.ir import Task, TaskMap, CollectiveIR
from ..core.enums import TaskStatus

class DependencyAnalyzer:
    def __init__(self, task_map: TaskMap):
        self.task_map = task_map
        self._analyze_dependencies()
    
    def _analyze_dependencies(self):
        self.dependency_graph = {} 
        self.reverse_dependency_graph = {} 
        self.critical_path = set()
        self.parallel_groups = []

        for task_id, task in self.task_map.tasks.items():
            self.dependency_graph[task_id] = set(dep.task_id for dep in task.dependencies)
            for dep in task.dependencies:
                if dep.task_id not in self.reverse_dependency_graph:
                    self.reverse_dependency_graph[dep.task_id] = set()
                self.reverse_dependency_graph[dep.task_id].add(task_id)

        self._find_critical_path()

        self._find_parallel_groups()
    
    def _find_critical_path(self):
        task_times = {}
        
        def calculate_critical_time(task_id: int) -> float:
            if task_id in task_times:
                return task_times[task_id]
            
            task = self.task_map.tasks[task_id]
            max_dep_time = 0.0
            
            for dep_id in self.dependency_graph[task_id]:
                dep_time = calculate_critical_time(dep_id)
                max_dep_time = max(max_dep_time, dep_time)
            
            total_time = max_dep_time + task.total_estimated_duration
            task_times[task_id] = total_time
            return total_time

        for task_id in self.task_map.tasks:
            calculate_critical_time(task_id)

        current_task = max(task_times, key=task_times.get)
        while current_task is not None:
            self.critical_path.add(current_task)
            next_task = None
            max_time = 0
            
            for dep_id in self.dependency_graph[current_task]:
                if task_times[dep_id] > max_time:
                    max_time = task_times[dep_id]
                    next_task = dep_id
            
            current_task = next_task
    
    def _find_parallel_groups(self):
        visited = set()
        
        for task_id in self.task_map.tasks:
            if task_id not in visited:
                independent_group = self._find_independent_tasks(task_id, visited)
                if len(independent_group) > 1:
                    self.parallel_groups.append(independent_group)
    
    def _find_independent_tasks(self, start_task_id: int, visited: Set[int]) -> Set[int]:
        independent_tasks = {start_task_id}
        visited.add(start_task_id)
        queue = [start_task_id]
        
        while queue:
            current_id = queue.pop(0)
            
            for other_id in self.task_map.tasks:
                if (other_id not in visited and 
                    other_id not in self.dependency_graph.get(current_id, set()) and
                    current_id not in self.dependency_graph.get(other_id, set())):
                    
                    independent_tasks.add(other_id)
                    visited.add(other_id)
                    queue.append(other_id)
        
        return independent_tasks
    
    def get_ready_tasks(self, completed_tasks: Set[int]) -> List[int]:
        ready_tasks = []
        
        for task_id, task in self.task_map.tasks.items():
            if (task_id not in completed_tasks and
                all(dep_id in completed_tasks for dep_id in self.dependency_graph.get(task_id, set()))):
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def get_task_parallelism(self) -> float:
        total_tasks = len(self.task_map.tasks)
        if total_tasks == 0:
            return 0.0

        critical_path_length = len(self.critical_path)
        return 1.0 - (critical_path_length / total_tasks)
    
    def get_critical_path_tasks(self) -> Set[int]:
        return self.critical_path
    
    def optimize_dependencies(self) -> Dict[int, Set[int]]:
        optimized_deps = {}
        
        for task_id, dependencies in self.dependency_graph.items():
            direct_deps = set(dependencies)

            transitive_deps = set()
            for dep_id in dependencies:
                transitive_deps.update(self._get_transitive_dependencies(dep_id))

            optimized_deps[task_id] = direct_deps - transitive_deps
        
        return optimized_deps
    
    def _get_transitive_dependencies(self, task_id: int) -> Set[int]:
        transitive = set()
        visited = set()
        
        def collect_transitive(current_id: int):
            if current_id in visited:
                return
            visited.add(current_id)
            
            for dep_id in self.dependency_graph.get(current_id, set()):
                transitive.add(dep_id)
                collect_transitive(dep_id)
        
        collect_transitive(task_id)
        return transitive
