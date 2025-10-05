from typing import Dict, List, Set
from ...core.ir import CollectiveIR, Task
from ..base import IRPass

class TaskSchedulingPass(IRPass):
    def __init__(self, scheduling_strategy: str = "critical_path"):
        self.scheduling_strategy = scheduling_strategy
    
    @property
    def name(self) -> str:
        return f"TaskSchedulingPass({self.scheduling_strategy})"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if self.scheduling_strategy == "critical_path":
            return self._critical_path_scheduling(ir)
        elif self.scheduling_strategy == "earliest_deadline":
            return self._earliest_deadline_scheduling(ir)
        else:
            return self._basic_scheduling(ir)
    
    def _critical_path_scheduling(self, ir: CollectiveIR) -> CollectiveIR:
        task_times = self._calculate_critical_path(ir)
        
        current_time = 0.0
        scheduled_tasks = set()
        
        while len(scheduled_tasks) < len(ir.task_map.tasks):
            ready_tasks = [
                task for task in ir.task_map.tasks.values()
                if all(dep.task_id in scheduled_tasks for dep in task.dependencies)
                and task.task_id not in scheduled_tasks
            ]
            
            if not ready_tasks:
                break
            
            ready_tasks.sort(key=lambda t: task_times[t.task_id], reverse=True)
            
            for task in ready_tasks:
                task.estimated_start_time = current_time
                task.estimated_end_time = current_time + task.total_estimated_duration
                scheduled_tasks.add(task.task_id)
                current_time += task.total_estimated_duration
        
        return ir
    
    def _calculate_critical_path(self, ir: CollectiveIR) -> Dict[int, float]:
        task_times = {}
        
        def calculate_time(task: Task) -> float:
            if task.task_id in task_times:
                return task_times[task.task_id]
            
            max_dep_time = 0.0
            for dep in task.dependencies:
                dep_time = calculate_time(dep)
                max_dep_time = max(max_dep_time, dep_time)
            
            total_time = max_dep_time + task.total_estimated_duration
            task_times[task.task_id] = total_time
            return total_time
        
        for task in ir.task_map.tasks.values():
            calculate_time(task)
        
        return task_times
    
    def _earliest_deadline_scheduling(self, ir: CollectiveIR) -> CollectiveIR:
        for task in ir.task_map.tasks.values():
            task.estimated_start_time = 0.0
            task.estimated_end_time = task.total_estimated_duration
        
        return ir
    
    def _basic_scheduling(self, ir: CollectiveIR) -> CollectiveIR:
        current_time = 0.0
        for task in ir.task_map.tasks.values():
            task.estimated_start_time = current_time
            task.estimated_end_time = current_time + task.total_estimated_duration
            current_time += task.total_estimated_duration
        
        return ir
