from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
from ..base import IRPass
from ...core.ir import CollectiveIR, Task

class SolverBasedOptimizationPass(IRPass, ABC):
    def __init__(self, time_limit: int = 60, optimization_level: int = 1):
        self.time_limit = time_limit
        self.optimization_level = optimization_level
        self.solution_metrics = {}
    
    @abstractmethod
    def _solve(self, ir: CollectiveIR) -> Dict[int, Task]:
        pass
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        start_time = time.time()
        
        try:
            new_tasks = self._solve(ir)
            execution_time = time.time() - start_time
            
            if execution_time > self.time_limit:
                print(f"Warning: {self.name} exceeded time limit ({execution_time:.2f}s)")
            
            self.solution_metrics['execution_time'] = execution_time
            self.solution_metrics['task_count'] = len(new_tasks) if new_tasks else 0
            
            if new_tasks:
                ir.task_map.tasks = new_tasks
                estimated_time = sum(task.total_estimated_duration for task in new_tasks.values())
                self.solution_metrics['estimated_completion_time'] = estimated_time
            
            return ir
            
        except Exception as e:
            print(f"{self.name} failed: {e}")
            self.solution_metrics['error'] = str(e)
            return ir
    
    def get_solution_metrics(self) -> Dict[str, Any]:
        return self.solution_metrics
