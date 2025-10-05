import time
from typing import Dict, Any, List, Tuple, Set
import z3

from ...core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory, RemoteMemory, Device
from ...core.enums import CollectiveOpType, PrimitiveOpType
from .base import SolverBasedOptimizationPass
from ..static.scheduling import TaskSchedulingPass
from ..static.algorithms.collective import CollectiveOptimizationPass

class SMTBasedOptimizationPass(SolverBasedOptimizationPass):
    def __init__(self, time_limit: int = 60, optimization_level: int = 1,
                 use_symmetry_breaking: bool = True, use_parallel_solving: bool = False):
        super().__init__(time_limit, optimization_level)
        self.use_symmetry_breaking = use_symmetry_breaking
        self.use_parallel_solving = use_parallel_solving
        self.constraint_stats = {}
        self.solver = None
    
    @property
    def name(self) -> str:
        return f"SMTBasedOptimizationPass(level={self.optimization_level}, symmetry={self.use_symmetry_breaking})"
    
    def _solve(self, ir: CollectiveIR) -> Dict[int, Task]:
        print(f"  SMT Solver: Starting Z3-based optimization")
        start_time = time.time()
        
        try:
            if self.optimization_level >= 2:
                solution = self._solve_comprehensive_smt(ir)
            elif self.optimization_level >= 1:
                solution = self._solve_basic_smt(ir)
            else:
                solution = self._solve_simple_smt(ir)
            
            solve_time = time.time() - start_time
            self.constraint_stats = {
                'solve_time': solve_time,
                'optimization_level': self.optimization_level,
                'symmetry_breaking': self.use_symmetry_breaking,
                'status': 'sat' if solution else 'unsat'
            }
            
            print(f"  SMT Solver: Completed in {solve_time:.2f}s")
            return solution
            
        except Exception as e:
            print(f"  SMT Solver failed: {e}")
            return self._heuristic_fallback(ir)
    
    def _solve_comprehensive_smt(self, ir: CollectiveIR) -> Dict[int, Task]:
        device_ids = list(ir.cluster.devices_by_id.keys())
        task_ids = list(ir.task_map.tasks.keys())
        
        solver = z3.Solver()
        solver.set("timeout", int(self.time_limit * 1000))
        
        # 移除不支持的参数设置
        # solver.set("verbose", 0)  # 这个参数在Z3中不存在
        
        if self.use_parallel_solving:
            solver.set("parallel.enable", True)
        
        start_times, end_times, device_assignments, task_orders = self._create_smt_variables(
            ir, device_ids, task_ids
        )
        
        self._add_basic_constraints(ir, solver, start_times, end_times, device_assignments, task_orders, device_ids, task_ids)
        self._add_dependency_constraints(ir, solver, start_times, end_times, task_ids)
        self._add_resource_constraints(ir, solver, start_times, end_times, device_assignments, task_orders, device_ids, task_ids)
        
        if self.optimization_level >= 2:
            self._add_bandwidth_constraints(ir, solver, device_ids, task_ids)
        
        if self.use_symmetry_breaking:
            self._add_symmetry_breaking_constraints(ir, solver, device_assignments, task_orders, device_ids, task_ids)
        
        makespan = self._define_makespan_objective(ir, solver, end_times, task_ids)
        
        if solver.check() == z3.sat:
            model = solver.model()
            return self._extract_solution_from_smt(ir, model, start_times, device_assignments, task_ids, device_ids)
        else:
            print("  SMT: No solution found")
            return {}
    
    def _solve_basic_smt(self, ir: CollectiveIR) -> Dict[int, Task]:
        device_ids = list(ir.cluster.devices_by_id.keys())
        task_ids = list(ir.task_map.tasks.keys())
        
        solver = z3.Solver()
        # 移除不支持的参数设置
        # solver.set("verbose", 0)  # 这个参数在Z3中不存在
        solver.set("timeout", int(self.time_limit * 1000))
        
        start_times, end_times, device_assignments, task_orders = self._create_smt_variables(
            ir, device_ids, task_ids
        )
        
        self._add_basic_constraints(ir, solver, start_times, end_times, device_assignments, task_orders, device_ids, task_ids)
        self._add_dependency_constraints(ir, solver, start_times, end_times, task_ids)
        
        makespan = self._define_makespan_objective(ir, solver, end_times, task_ids)
        
        if solver.check() == z3.sat:
            model = solver.model()
            return self._extract_solution_from_smt(ir, model, start_times, device_assignments, task_ids, device_ids)
        else:
            return self._heuristic_fallback(ir)
    
    def _solve_simple_smt(self, ir: CollectiveIR) -> Dict[int, Task]:
        return self._heuristic_fallback(ir)
    
    def _create_smt_variables(self, ir: CollectiveIR, device_ids: List[int], task_ids: List[int]):
        start_times = {}
        end_times = {}
        device_assignments = {}
        task_orders = {}
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            start_times[task_id] = z3.Real(f"start_{task_id}")
            end_times[task_id] = z3.Real(f"end_{task_id}")
            
            for dev_id in device_ids:
                device_assignments[(task_id, dev_id)] = z3.Bool(f"assign_{task_id}_{dev_id}")
        
        for i, tid1 in enumerate(task_ids):
            for j, tid2 in enumerate(task_ids):
                if i < j:
                    task_orders[(tid1, tid2)] = z3.Bool(f"order_{tid1}_{tid2}")
        
        return start_times, end_times, device_assignments, task_orders
    
    def _add_basic_constraints(self, ir: CollectiveIR, solver: z3.Solver,
                             start_times: Dict[int, z3.ExprRef], end_times: Dict[int, z3.ExprRef],
                             device_assignments: Dict[Tuple[int, int], z3.ExprRef],
                             task_orders: Dict[Tuple[int, int], z3.ExprRef],
                             device_ids: List[int], task_ids: List[int]):
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            solver.add(start_times[task_id] >= 0)
            solver.add(end_times[task_id] == start_times[task_id] + z3.RealVal(task.total_estimated_duration))
            
            solver.add(z3.Or([device_assignments[(task_id, dev_id)] for dev_id in device_ids]))
            
            for dev_id in device_ids:
                for other_dev in device_ids:
                    if dev_id != other_dev:
                        solver.add(z3.Implies(
                            device_assignments[(task_id, dev_id)],
                            z3.Not(device_assignments[(task_id, other_dev)])
                        ))
    
    def _add_dependency_constraints(self, ir: CollectiveIR, solver: z3.Solver,
                                  start_times: Dict[int, z3.ExprRef], end_times: Dict[int, z3.ExprRef],
                                  task_ids: List[int]):
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            for dep in task.dependencies:
                if dep.task_id in task_ids:
                    solver.add(start_times[task_id] >= end_times[dep.task_id])
    
    def _add_resource_constraints(self, ir: CollectiveIR, solver: z3.Solver,
                                start_times: Dict[int, z3.ExprRef], end_times: Dict[int, z3.ExprRef],
                                device_assignments: Dict[Tuple[int, int], z3.ExprRef],
                                task_orders: Dict[Tuple[int, int], z3.ExprRef],
                                device_ids: List[int], task_ids: List[int]):
        
        for dev_id in device_ids:
            for i, tid1 in enumerate(task_ids):
                for j, tid2 in enumerate(task_ids):
                    if i < j:
                        both_assigned = z3.And(
                            device_assignments[(tid1, dev_id)],
                            device_assignments[(tid2, dev_id)]
                        )
                        
                        tid1_before_tid2 = z3.And(
                            both_assigned,
                            task_orders[(tid1, tid2)]
                        )
                        
                        tid2_before_tid1 = z3.And(
                            both_assigned,
                            z3.Not(task_orders[(tid1, tid2)])
                        )
                        
                        solver.add(z3.Implies(
                            tid1_before_tid2,
                            start_times[tid2] >= end_times[tid1]
                        ))
                        
                        solver.add(z3.Implies(
                            tid2_before_tid1,
                            start_times[tid1] >= end_times[tid2]
                        ))
    
    def _add_bandwidth_constraints(self, ir: CollectiveIR, solver: z3.Solver,
                                 device_ids: List[int], task_ids: List[int]):
        
        for dev1 in device_ids:
            for dev2 in device_ids:
                if dev1 != dev2:
                    bandwidth = ir.cluster.network_topology.get_bandwidth(dev1, dev2)
                    if bandwidth > 0:
                        total_comm = z3.Real(f"total_comm_{dev1}_{dev2}")
                        solver.add(total_comm >= 0)
                        solver.add(total_comm <= bandwidth)
    
    def _add_symmetry_breaking_constraints(self, ir: CollectiveIR, solver: z3.Solver,
                                         device_assignments: Dict[Tuple[int, int], z3.ExprRef],
                                         task_orders: Dict[Tuple[int, int], z3.ExprRef],
                                         device_ids: List[int], task_ids: List[int]):
        
        similar_devices = self._find_similar_devices(ir, device_ids)
        for group in similar_devices:
            if len(group) > 1:
                for i in range(len(group) - 1):
                    dev1, dev2 = group[i], group[i+1]
                    
                    dev1_tasks = [device_assignments[(tid, dev1)] for tid in task_ids]
                    dev2_tasks = [device_assignments[(tid, dev2)] for tid in task_ids]
                    
                    solver.add(z3.Sum([z3.If(dev1_tasks[i], 1, 0) for i in range(len(dev1_tasks))]) >=
                              z3.Sum([z3.If(dev2_tasks[i], 1, 0) for i in range(len(dev2_tasks))]))
    
    def _find_similar_devices(self, ir: CollectiveIR, device_ids: List[int]) -> List[List[int]]:
        device_types = {}
        for dev_id in device_ids:
            device = ir.cluster.get_device(dev_id)
            dev_type = device.type
            if dev_type not in device_types:
                device_types[dev_type] = []
            device_types[dev_type].append(dev_id)
        
        return [group for group in device_types.values() if len(group) > 1]
    
    def _define_makespan_objective(self, ir: CollectiveIR, solver: z3.Solver,
                                 end_times: Dict[int, z3.ExprRef], task_ids: List[int]) -> z3.ExprRef:
        
        makespan = z3.Real("makespan")
        
        for task_id in task_ids:
            solver.add(makespan >= end_times[task_id])
        
        objective = z3.Optimize()
        for constraint in solver.assertions():
            objective.add(constraint)
        
        objective.minimize(makespan)
        
        return objective
    
    def _extract_solution_from_smt(self, ir: CollectiveIR, model: z3.ModelRef,
                                 start_times: Dict[int, z3.ExprRef],
                                 device_assignments: Dict[Tuple[int, int], z3.ExprRef],
                                 task_ids: List[int], device_ids: List[int]) -> Dict[int, Task]:
        
        solution_tasks = {}
        
        for task_id in task_ids:
            task = ir.task_map.tasks[task_id]
            
            assigned_device_id = None
            for dev_id in device_ids:
                if z3.is_true(model[device_assignments[(task_id, dev_id)]]):
                    assigned_device_id = dev_id
                    break
            
            if assigned_device_id is not None:
                start_time = float(model[start_times[task_id]].as_fraction())
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
    
    def _heuristic_fallback(self, ir: CollectiveIR) -> Dict[int, Task]:
        print("  SMT: Using heuristic fallback")
        collective_optimizer = CollectiveOptimizationPass()
        fallback_ir = collective_optimizer.run(ir)
        return fallback_ir.task_map.tasks
    
    def get_constraint_stats(self) -> Dict[str, Any]:
        return self.constraint_stats
