import time
import numpy as np
import pulp
from typing import Dict, Any, List, Tuple, Set
from scipy.optimize import linprog
import ortools.linear_solver.pywraplp as ortools_lp

from ...core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory, RemoteMemory, Device
from ...core.enums import CollectiveOpType, PrimitiveOpType
from .base import SolverBasedOptimizationPass
from ..static.scheduling import TaskSchedulingPass
from ..static.algorithms.collective import CollectiveOptimizationPass

class MILPOptimizationPass(SolverBasedOptimizationPass):
    def __init__(self, time_limit: int = 60, optimization_level: int = 1, 
                 solver_backend: str = "pulp", use_heuristic_fallback: bool = True):
        super().__init__(time_limit, optimization_level)
        self.solver_backend = solver_backend
        self.use_heuristic_fallback = use_heuristic_fallback
        self.solution_stats = {}
        self.model = None
    
    @property
    def name(self) -> str:
        return f"MILPOptimizationPass(level={self.optimization_level}, solver={self.solver_backend})"
    
    def _solve(self, ir: CollectiveIR) -> Dict[int, Task]:
        print(f"  MILP Solver: Starting optimization using {self.solver_backend}")
        start_time = time.time()
        
        try:
            if self.optimization_level >= 2:
                solution = self._solve_comprehensive_milp(ir)
            elif self.optimization_level >= 1:
                solution = self._solve_basic_milp(ir)
            else:
                solution = self._solve_simple_milp(ir)
            
            solve_time = time.time() - start_time
            self.solution_stats = {
                'solve_time': solve_time,
                'solver_backend': self.solver_backend,
                'optimization_level': self.optimization_level,
                'status': 'optimal' if solution else 'failed'
            }
            
            print(f"  MILP Solver: Completed in {solve_time:.2f}s")
            return solution
            
        except Exception as e:
            print(f"  MILP Solver failed: {e}")
            if self.use_heuristic_fallback:
                return self._heuristic_fallback(ir)
            return {}
    
    def _solve_comprehensive_milp(self, ir: CollectiveIR) -> Dict[int, Task]:
        device_ids = list(ir.cluster.devices_by_id.keys())
        task_ids = list(ir.task_map.tasks.keys())
        n_devices = len(device_ids)
        n_tasks = len(task_ids)
        
        if self.solver_backend == "pulp":
            return self._solve_with_pulp(ir, device_ids, task_ids, comprehensive=True)
        elif self.solver_backend == "ortools":
            return self._solve_with_ortools(ir, device_ids, task_ids, comprehensive=True)
        else:
            return self._solve_with_scipy(ir, device_ids, task_ids, comprehensive=True)
    
    def _solve_basic_milp(self, ir: CollectiveIR) -> Dict[int, Task]:
        device_ids = list(ir.cluster.devices_by_id.keys())
        task_ids = list(ir.task_map.tasks.keys())
        
        if self.solver_backend == "pulp":
            return self._solve_with_pulp(ir, device_ids, task_ids, comprehensive=False)
        elif self.solver_backend == "ortools":
            return self._solve_with_ortools(ir, device_ids, task_ids, comprehensive=False)
        else:
            return self._solve_with_scipy(ir, device_ids, task_ids, comprehensive=False)
    
    def _solve_simple_milp(self, ir: CollectiveIR) -> Dict[int, Task]:
        return self._solve_with_scipy(ir, 
                                    list(ir.cluster.devices_by_id.keys()),
                                    list(ir.task_map.tasks.keys()),
                                    comprehensive=False)
    
    def _solve_with_pulp(self, ir: CollectiveIR, device_ids: List[int], 
                        task_ids: List[int], comprehensive: bool = True) -> Dict[int, Task]:
        model = pulp.LpProblem("CollectiveCommunicationOptimization", pulp.LpMinimize)
        
        n_devices = len(device_ids)
        n_tasks = len(task_ids)
        
        start_times = {}
        end_times = {}
        device_assignments = {}
        task_orders = {}
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            start_times[task_id] = pulp.LpVariable(f"start_{task_id}", lowBound=0, cat='Continuous')
            end_times[task_id] = pulp.LpVariable(f"end_{task_id}", lowBound=0, cat='Continuous')
            
            model += end_times[task_id] == start_times[task_id] + task.total_estimated_duration
            
            for dev_id in device_ids:
                device_assignments[(task_id, dev_id)] = pulp.LpVariable(
                    f"assign_{task_id}_{dev_id}", cat='Binary'
                )
        
        for i, tid1 in enumerate(task_ids):
            for j, tid2 in enumerate(task_ids):
                if i < j:
                    task_orders[(tid1, tid2)] = pulp.LpVariable(
                        f"order_{tid1}_{tid2}", cat='Binary'
                    )
        
        makespan = pulp.LpVariable("makespan", lowBound=0, cat='Continuous')
        
        for task_id in task_ids:
            model += end_times[task_id] <= makespan
        
        for task_id in task_ids:
            model += pulp.lpSum([device_assignments[(task_id, dev_id)] for dev_id in device_ids]) == 1
        
        for task_id in task_ids:
            for dep in ir.task_map.tasks[task_id].dependencies:
                model += start_times[task_id] >= end_times[dep.task_id]
        
        for dev_id in device_ids:
            for i, tid1 in enumerate(task_ids):
                for j, tid2 in enumerate(task_ids):
                    if i < j:
                        M = 1000000
                        model += start_times[tid1] >= end_times[tid2] - M * (1 - task_orders[(tid1, tid2)])
                        model += start_times[tid2] >= end_times[tid1] - M * task_orders[(tid1, tid2)]
        
        if comprehensive:
            bandwidth_usage = self._add_bandwidth_constraints_pulp(ir, model, device_ids, task_ids)
            model += bandwidth_usage
        
        model += makespan
        
        solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit, msg=0)
        model.solve(solver)
        
        if model.status == pulp.LpStatusOptimal:
            return self._extract_solution_from_pulp(ir, start_times, device_assignments, task_ids, device_ids)
        else:
            print(f"  MILP Pulp: No optimal solution found, status: {pulp.LpStatus[model.status]}")
            return {}
    
    def _add_bandwidth_constraints_pulp(self, ir: CollectiveIR, model: pulp.LpProblem,
                                      device_ids: List[int], task_ids: List[int]) -> pulp.LpAffineExpression:
        bandwidth_usage = pulp.LpVariable("total_bandwidth_usage", lowBound=0, cat='Continuous')
        
        for step in range(10):
            time_point = pulp.LpVariable(f"time_point_{step}", lowBound=0, cat='Continuous')
            
            for dev1 in device_ids:
                for dev2 in device_ids:
                    if dev1 != dev2:
                        link_usage = pulp.LpVariable(f"link_usage_{dev1}_{dev2}_{step}", lowBound=0, cat='Continuous')
                        max_bandwidth = ir.cluster.network_topology.get_bandwidth(dev1, dev2)
                        
                        if max_bandwidth > 0:
                            model += link_usage <= max_bandwidth
        
        return bandwidth_usage
    
    def _extract_solution_from_pulp(self, ir: CollectiveIR, start_times: Dict[int, pulp.LpVariable],
                                  device_assignments: Dict[Tuple[int, int], pulp.LpVariable],
                                  task_ids: List[int], device_ids: List[int]) -> Dict[int, Task]:
        solution_tasks = {}
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            
            assigned_device_id = None
            for dev_id in device_ids:
                if pulp.value(device_assignments[(task_id, dev_id)]) > 0.5:
                    assigned_device_id = dev_id
                    break
            
            if assigned_device_id is not None:
                start_time = pulp.value(start_times[task_id])
                device = ir.cluster.get_device(assigned_device_id)
                
                optimized_primitives = []
                for primitive in task.primitives:
                    optimized_primitive = CommunicationPrimitive(
                        initiator=device,
                        op_type=primitive.op_type,
                        memory_regions=primitive.memory_regions,
                        estimated_duration_ms=primitive.estimated_duration_ms
                    )
                    optimized_primitives.append(optimized_primitive)
                
                optimized_task = Task(
                    task_id=task_id,
                    primitives=optimized_primitives,
                    dependencies=task.dependencies,
                    status=task.status,
                    estimated_start_time=start_time,
                    estimated_end_time=start_time + task.total_estimated_duration
                )
                solution_tasks[task_id] = optimized_task
        
        return solution_tasks
    
    def _solve_with_ortools(self, ir: CollectiveIR, device_ids: List[int],
                          task_ids: List[int], comprehensive: bool = True) -> Dict[int, Task]:
        solver = ortools_lp.Solver.CreateSolver('SCIP')
        if not solver:
            return {}
        
        n_devices = len(device_ids)
        n_tasks = len(task_ids)
        
        infinity = solver.infinity()
        start_times = {}
        end_times = {}
        device_assignments = {}
        task_orders = {}
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            start_times[task_id] = solver.NumVar(0, infinity, f'start_{task_id}')
            end_times[task_id] = solver.NumVar(0, infinity, f'end_{task_id}')
            
            solver.Add(end_times[task_id] == start_times[task_id] + task.total_estimated_duration)
            
            for dev_id in device_ids:
                device_assignments[(task_id, dev_id)] = solver.IntVar(0, 1, f'assign_{task_id}_{dev_id}')
        
        for i, tid1 in enumerate(task_ids):
            for j, tid2 in enumerate(task_ids):
                if i < j:
                    task_orders[(tid1, tid2)] = solver.IntVar(0, 1, f'order_{tid1}_{tid2}')
        
        makespan = solver.NumVar(0, infinity, 'makespan')
        
        for task_id in task_ids:
            solver.Add(end_times[task_id] <= makespan)
        
        for task_id in task_ids:
            solver.Add(sum(device_assignments[(task_id, dev_id)] for dev_id in device_ids) == 1)
        
        for task_id in task_ids:
            for dep in ir.task_map.tasks[task_id].dependencies:
                solver.Add(start_times[task_id] >= end_times[dep.task_id])
        
        M = 1000000
        for dev_id in device_ids:
            for i, tid1 in enumerate(task_ids):
                for j, tid2 in enumerate(task_ids):
                    if i < j:
                        solver.Add(start_times[tid1] >= end_times[tid2] - M * (1 - task_orders[(tid1, tid2)]))
                        solver.Add(start_times[tid2] >= end_times[tid1] - M * task_orders[(tid1, tid2)])
        
        solver.Minimize(makespan)
        
        solver.SetTimeLimit(self.time_limit * 1000)
        status = solver.Solve()
        
        if status == ortools_lp.Solver.OPTIMAL:
            return self._extract_solution_from_ortools(ir, solver, start_times, device_assignments, task_ids, device_ids)
        else:
            print(f"  MILP OR-Tools: No optimal solution found, status: {status}")
            return {}
    
    def _extract_solution_from_ortools(self, ir: CollectiveIR, solver: ortools_lp.Solver,
                                     start_times: Dict[int, ortools_lp.Variable],
                                     device_assignments: Dict[Tuple[int, int], ortools_lp.Variable],
                                     task_ids: List[int], device_ids: List[int]) -> Dict[int, Task]:
        solution_tasks = {}
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            
            assigned_device_id = None
            for dev_id in device_ids:
                if device_assignments[(task_id, dev_id)].solution_value() > 0.5:
                    assigned_device_id = dev_id
                    break
            
            if assigned_device_id is not None:
                start_time = start_times[task_id].solution_value()
                device = ir.cluster.get_device(assigned_device_id)
                
                optimized_primitives = []
                for primitive in task.primitives:
                    optimized_primitive = CommunicationPrimitive(
                        initiator=device,
                        op_type=primitive.op_type,
                        memory_regions=primitive.memory_regions,
                        estimated_duration_ms=primitive.estimated_duration_ms
                    )
                    optimized_primitives.append(optimized_primitive)
                
                optimized_task = Task(
                    task_id=task_id,
                    primitives=optimized_primitives,
                    dependencies=task.dependencies,
                    status=task.status,
                    estimated_start_time=start_time,
                    estimated_end_time=start_time + task.total_estimated_duration
                )
                solution_tasks[task_id] = optimized_task
        
        return solution_tasks
    
    def _solve_with_scipy(self, ir: CollectiveIR, device_ids: List[int],
                         task_ids: List[int], comprehensive: bool = True) -> Dict[int, Task]:
        n_tasks = len(task_ids)
        n_devices = len(device_ids)
        
        if n_tasks == 0:
            return {}
        
        c = np.zeros(n_tasks + 1)
        c[-1] = 1
        
        A_ub = []
        b_ub = []
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            
            for dep in task.dependencies:
                if dep.task_id in task_ids:
                    dep_index = task_ids.index(dep.task_id)
                    task_index = task_ids.index(task_id)
                    
                    row = np.zeros(n_tasks + 1)
                    row[task_index] = 1
                    row[dep_index] = -1
                    row[-1] = 0
                    
                    A_ub.append(row)
                    b_ub.append(-task.total_estimated_duration)
        
        if comprehensive and n_tasks > 1:
            for i in range(n_tasks):
                for j in range(i + 1, n_tasks):
                    row1 = np.zeros(n_tasks + 1)
                    row1[i] = 1
                    row1[j] = -1
                    row1[-1] = 0
                    
                    row2 = np.zeros(n_tasks + 1)
                    row2[i] = -1
                    row2[j] = 1
                    row2[-1] = 0
                    
                    A_ub.append(row1)
                    b_ub.append(-ir.task_map.tasks[task_ids[j]].total_estimated_duration)
                    
                    A_ub.append(row2)
                    b_ub.append(-ir.task_map.tasks[task_ids[i]].total_estimated_duration)
        
        A_ub = np.array(A_ub) if A_ub else np.zeros((0, n_tasks + 1))
        b_ub = np.array(b_ub) if b_ub else np.zeros(0)
        
        bounds = [(0, None) for _ in range(n_tasks)] + [(0, None)]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                return self._extract_solution_from_scipy(ir, result.x, task_ids, device_ids)
        except:
            pass
        
        return {}
    
    def _extract_solution_from_scipy(self, ir: CollectiveIR, solution: np.ndarray,
                                   task_ids: List[int], device_ids: List[int]) -> Dict[int, Task]:
        solution_tasks = {}
        
        for i, task_id in enumerate(task_ids):
            if i < len(solution) - 1:
                start_time = solution[i]
                task = ir.task_map.tasks[task_id]
                
                device = self._select_best_device(ir, task, device_ids)
                
                optimized_primitives = []
                for primitive in task.primitives:
                    optimized_primitive = CommunicationPrimitive(
                        initiator=device,
                        op_type=primitive.op_type,
                        memory_regions=primitive.memory_regions,
                        estimated_duration_ms=primitive.estimated_duration_ms
                    )
                    optimized_primitives.append(optimized_primitive)
                
                optimized_task = Task(
                    task_id=task_id,
                    primitives=optimized_primitives,
                    dependencies=task.dependencies,
                    status=task.status,
                    estimated_start_time=start_time,
                    estimated_end_time=start_time + task.total_estimated_duration
                )
                solution_tasks[task_id] = optimized_task
        
        return solution_tasks
    
    def _select_best_device(self, ir: CollectiveIR, task: Task, device_ids: List[int]) -> Device:
        best_device = None
        best_score = float('inf')
        
        for dev_id in device_ids:
            device = ir.cluster.get_device(dev_id)
            score = 0
            
            for primitive in task.primitives:
                for mem_region in primitive.memory_regions:
                    if mem_region.device.device_id == dev_id:
                        score += 1
                    else:
                        latency = ir.cluster.network_topology.get_latency(dev_id, mem_region.device.device_id)
                        score += latency
            
            if score < best_score:
                best_score = score
                best_device = device
        
        return best_device or ir.cluster.get_device(device_ids[0])
    
    def _heuristic_fallback(self, ir: CollectiveIR) -> Dict[int, Task]:
        print("  MILP: Using heuristic fallback")
        scheduler = TaskSchedulingPass("critical_path")
        scheduled_ir = scheduler.run(ir)
        return scheduled_ir.task_map.tasks
    
    def get_solution_stats(self) -> Dict[str, Any]:
        return self.solution_stats
