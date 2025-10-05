from typing import Dict, List, Set, Tuple
from ..core.ir import Task, TaskMap, CollectiveIR, Device
from ..core.enums import TaskStatus
from .analyzer import DependencyAnalyzer

class AdvancedTaskScheduler:
    def __init__(self, ir: CollectiveIR):
        self.ir = ir
        self.analyzer = DependencyAnalyzer(ir.task_map)
        self.device_timelines = {}
    
    def schedule_tasks(self, strategy: str = "critical_path") -> CollectiveIR:
        if strategy == "critical_path":
            return self._critical_path_scheduling()
        elif strategy == "earliest_finish":
            return self._earliest_finish_scheduling()
        elif strategy == "load_balanced":
            return self._load_balanced_scheduling()
        else:
            return self._critical_path_scheduling()
    
    def _critical_path_scheduling(self) -> CollectiveIR:
        self._initialize_device_timelines()

        critical_times = self._calculate_critical_times()

        sorted_tasks = sorted(
            self.ir.task_map.tasks.values(),
            key=lambda t: critical_times[t.task_id],
            reverse=True
        )

        scheduled_tasks = set()
        
        while len(scheduled_tasks) < len(sorted_tasks):
            ready_tasks = [
                t for t in sorted_tasks 
                if t.task_id not in scheduled_tasks and
                all(dep.task_id in scheduled_tasks for dep in t.dependencies)
            ]
            
            if not ready_tasks:
                break

            critical_ready = [t for t in ready_tasks if t.task_id in self.analyzer.critical_path]
            if critical_ready:
                tasks_to_schedule = critical_ready
            else:
                tasks_to_schedule = ready_tasks
            
            for task in tasks_to_schedule:
                self._schedule_task(task)
                scheduled_tasks.add(task.task_id)
        
        return self.ir
    
    def _earliest_finish_scheduling(self) -> CollectiveIR:
        self._initialize_device_timelines()
        
        scheduled_tasks = set()
        
        while len(scheduled_tasks) < len(self.ir.task_map.tasks):
            ready_tasks = [
                t for t in self.ir.task_map.tasks.values()
                if t.task_id not in scheduled_tasks and
                all(dep.task_id in scheduled_tasks for dep in t.dependencies)
            ]
            
            if not ready_tasks:
                break

            best_task = None
            best_finish_time = float('inf')
            
            for task in ready_tasks:
                devices = self._get_task_devices(task)

                earliest_start = max(
                    max((self.device_timelines[dev.device_id] for dev in devices), default=0),
                    max((dep.estimated_end_time for dep in task.dependencies), default=0)
                )
                
                finish_time = earliest_start + task.total_estimated_duration
                
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_task = task
            
            if best_task:
                self._schedule_task(best_task)
                scheduled_tasks.add(best_task.task_id)
        
        return self.ir
    
    def _load_balanced_scheduling(self) -> CollectiveIR:
        self._initialize_device_timelines()
        
        scheduled_tasks = set()
        device_loads = {dev_id: 0.0 for dev_id in self.device_timelines.keys()}
        
        while len(scheduled_tasks) < len(self.ir.task_map.tasks):
            ready_tasks = [
                t for t in self.ir.task_map.tasks.values()
                if t.task_id not in scheduled_tasks and
                all(dep.task_id in scheduled_tasks for dep in t.dependencies)
            ]
            
            if not ready_tasks:
                break

            best_task = None
            min_max_load = float('inf')
            
            for task in ready_tasks:
                devices = self._get_task_devices(task)

                current_loads = [device_loads[dev.device_id] for dev in devices]
                max_load = max(current_loads) if current_loads else 0
                
                if max_load < min_max_load:
                    min_max_load = max_load
                    best_task = task
            
            if best_task:
                self._schedule_task(best_task)
                scheduled_tasks.add(best_task.task_id)

                devices = self._get_task_devices(best_task)
                for dev in devices:
                    device_loads[dev.device_id] += best_task.total_estimated_duration
        
        return self.ir
    
    def _initialize_device_timelines(self):
        all_devices = set()
        for task in self.ir.task_map.tasks.values():
            for primitive in task.primitives:
                all_devices.add(primitive.initiator.device_id)
                for mem_region in primitive.memory_regions:
                    all_devices.add(mem_region.device.device_id)
        
        self.device_timelines = {dev_id: 0.0 for dev_id in all_devices}
    
    def _calculate_critical_times(self) -> Dict[int, float]:
        task_times = {}
        
        def calculate_time(task_id: int) -> float:
            if task_id in task_times:
                return task_times[task_id]
            
            task = self.ir.task_map.tasks[task_id]
            max_dep_time = 0.0
            
            for dep in task.dependencies:
                dep_time = calculate_time(dep.task_id)
                max_dep_time = max(max_dep_time, dep_time)
            
            total_time = max_dep_time + task.total_estimated_duration
            task_times[task_id] = total_time
            return total_time
        
        for task_id in self.ir.task_map.tasks:
            calculate_time(task_id)
        
        return task_times
    
    def _get_task_devices(self, task: Task) -> List[Device]:
        devices = set()
        for primitive in task.primitives:
            devices.add(primitive.initiator)
            for mem_region in primitive.memory_regions:
                devices.add(mem_region.device)
        return list(devices)
    
    def _schedule_task(self, task: Task):
        devices = self._get_task_devices(task)

        dependency_end = max((dep.estimated_end_time for dep in task.dependencies), default=0)
        device_availability = max((self.device_timelines[dev.device_id] for dev in devices), default=0)
        start_time = max(dependency_end, device_availability)

        task.estimated_start_time = start_time
        task.estimated_end_time = start_time + task.total_estimated_duration

        for dev in devices:
            self.device_timelines[dev.device_id] = task.estimated_end_time
