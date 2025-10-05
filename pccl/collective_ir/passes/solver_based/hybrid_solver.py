import time
import numpy as np
from typing import List, Dict, Any, Tuple, Set
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...core.ir import CollectiveIR, Task
from .base import SolverBasedOptimizationPass
from .smt_solver import SMTBasedOptimizationPass
from .milp_solver import MILPOptimizationPass
from .mst_solver import BottleneckAwareMSTPass
from ..static.algorithms.collective import CollectiveOptimizationPass
from ...dependency_analysis.scheduler import AdvancedTaskScheduler

class HybridSolverPass(SolverBasedOptimizationPass):
    def __init__(self, solvers: List[SolverBasedOptimizationPass] = None, 
                 selection_strategy: str = "adaptive_ensemble",
                 ensemble_weighting: bool = True,
                 parallel_execution: bool = True,
                 quality_threshold: float = 0.1):
        self.solvers = solvers or self._create_adaptive_solvers()
        self.selection_strategy = selection_strategy
        self.ensemble_weighting = ensemble_weighting
        self.parallel_execution = parallel_execution
        self.quality_threshold = quality_threshold
        self.solver_results = {}
        self.best_solution = None
        self.best_score = float('inf')
        super().__init__(time_limit=180, optimization_level=3)
    
    @property
    def name(self) -> str:
        return f"HybridSolverPass({self.selection_strategy}, ensemble={self.ensemble_weighting}, parallel={self.parallel_execution})"
    
    def _create_adaptive_solvers(self) -> List[SolverBasedOptimizationPass]:
        return [
            BottleneckAwareMSTPass(optimization_level=2),
            SMTBasedOptimizationPass(optimization_level=1, time_limit=45, use_symmetry_breaking=True),
            MILPOptimizationPass(optimization_level=2, time_limit=60, solver_backend="pulp"),
            MILPOptimizationPass(optimization_level=1, time_limit=30, solver_backend="ortools"),
            CollectiveOptimizationPass(algorithm_selection_strategy="auto"),
        ]
    
    def _solve(self, ir: CollectiveIR) -> Dict[int, Task]:
        print(f"Hybrid Solver: Starting with {len(self.solvers)} solvers")
        print(f"  Strategy: {self.selection_strategy}, Ensemble: {self.ensemble_weighting}, Parallel: {self.parallel_execution}")
        
        start_time = time.time()
        self.solver_results = {}
        self.best_solution = None
        self.best_score = float('inf')
        
        problem_characteristics = self._analyze_problem_characteristics(ir)
        print(f"  Problem characteristics: {problem_characteristics}")
        
        filtered_solvers = self._filter_solvers_by_problem(ir, problem_characteristics)
        print(f"  Selected {len(filtered_solvers)} solvers for this problem")
        
        candidate_solutions = self._run_solvers_parallel(ir, filtered_solvers) if self.parallel_execution else self._run_solvers_sequential(ir, filtered_solvers)
        
        if not candidate_solutions:
            print("  Hybrid Solver: All solvers failed, using comprehensive fallback")
            return self._comprehensive_fallback(ir)
        
        best_solution = self._select_best_solution(ir, candidate_solutions, problem_characteristics)
        
        if self.ensemble_weighting and len(candidate_solutions) > 1:
            refined_solution = self._ensemble_refinement(ir, best_solution, candidate_solutions)
            if refined_solution:
                best_solution = refined_solution
        
        total_time = time.time() - start_time
        print(f"Hybrid Solver: Completed in {total_time:.2f}s")
        print(f"  Best score: {self.best_score:.2f}")
        
        return best_solution
    
    def _analyze_problem_characteristics(self, ir: CollectiveIR) -> Dict[str, Any]:
        device_ids = list(ir.cluster.devices_by_id.keys())
        task_ids = list(ir.task_map.tasks.keys())
        
        characteristics = {
            'num_devices': len(device_ids),
            'num_tasks': len(task_ids),
            'collective_op': ir.collective_op.name,
            'data_size_gb': ir.data_size_gb,
            'graph_density': self._calculate_graph_density(ir),
            'heterogeneity': self._calculate_heterogeneity(ir),
            'critical_path_length': self._estimate_critical_path_length(ir),
            'parallelism_potential': self._estimate_parallelism_potential(ir),
        }
        
        return characteristics
    
    def _calculate_graph_density(self, ir: CollectiveIR) -> float:
        task_ids = list(ir.task_map.tasks.keys())
        n = len(task_ids)
        if n <= 1:
            return 0.0
        
        edge_count = 0
        for task in ir.task_map.tasks.values():
            edge_count += len(task.dependencies)
        
        max_edges = n * (n - 1)
        return edge_count / max_edges if max_edges > 0 else 0.0
    
    def _calculate_heterogeneity(self, ir: CollectiveIR) -> float:
        device_bandwidths = []
        for device in ir.cluster.devices_by_id.values():
            if hasattr(device, 'bandwidth_gbs'):
                device_bandwidths.append(device.bandwidth_gbs)
        
        if len(device_bandwidths) <= 1:
            return 0.0
        
        return np.std(device_bandwidths) / np.mean(device_bandwidths)
    
    def _estimate_critical_path_length(self, ir: CollectiveIR) -> float:
        from ...dependency_analysis.analyzer import DependencyAnalyzer
        analyzer = DependencyAnalyzer(ir.task_map)
        critical_path_tasks = analyzer.get_critical_path_tasks()
        
        total_duration = 0.0
        for task_id in critical_path_tasks:
            if task_id in ir.task_map.tasks:
                total_duration += ir.task_map.tasks[task_id].total_estimated_duration
        
        return total_duration
    
    def _estimate_parallelism_potential(self, ir: CollectiveIR) -> float:
        from ...dependency_analysis.analyzer import DependencyAnalyzer
        analyzer = DependencyAnalyzer(ir.task_map)
        return analyzer.get_task_parallelism()
    
    def _filter_solvers_by_problem(self, ir: CollectiveIR, characteristics: Dict[str, Any]) -> List[SolverBasedOptimizationPass]:
        filtered_solvers = []
        
        for solver in self.solvers:
            if self._is_solver_suitable(solver, characteristics):
                filtered_solvers.append(solver)
        
        return filtered_solvers
    
    def _is_solver_suitable(self, solver: SolverBasedOptimizationPass, characteristics: Dict[str, Any]) -> bool:
        num_devices = characteristics['num_devices']
        num_tasks = characteristics['num_tasks']
        graph_density = characteristics['graph_density']
        heterogeneity = characteristics['heterogeneity']
        
        if isinstance(solver, BottleneckAwareMSTPass):
            return num_devices >= 2 and heterogeneity > 0.1
        
        elif isinstance(solver, SMTBasedOptimizationPass):
            return num_tasks <= 50 and graph_density < 0.7
        
        elif isinstance(solver, MILPOptimizationPass):
            if solver.optimization_level >= 2:
                return num_tasks <= 100
            else:
                return num_tasks <= 200
        
        elif isinstance(solver, CollectiveOptimizationPass):
            return True
        
        return True
    
    def _run_solvers_parallel(self, ir: CollectiveIR, solvers: List[SolverBasedOptimizationPass]) -> List[Tuple[str, Dict[int, Task], float]]:
        candidate_solutions = []
        
        def run_solver(solver):
            solver_name = solver.name
            try:
                solver_start = time.time()
                candidate_ir = solver.run(ir.copy() if hasattr(ir, 'copy') else ir)
                solver_time = time.time() - solver_start
                
                if candidate_ir and candidate_ir.task_map.tasks:
                    candidate_time = self._evaluate_solution_quality(candidate_ir)
                    
                    self.solver_results[solver_name] = {
                        'success': True,
                        'execution_time': solver_time,
                        'solution_quality': candidate_time,
                        'task_count': len(candidate_ir.task_map.tasks)
                    }
                    
                    return (solver_name, candidate_ir.task_map.tasks, candidate_time, solver_time)
                else:
                    self.solver_results[solver_name] = {
                        'success': False,
                        'error': 'No valid solution generated'
                    }
                    
            except Exception as e:
                self.solver_results[solver_name] = {
                    'success': False,
                    'error': str(e)
                }
            
            return None
        
        with ThreadPoolExecutor(max_workers=min(len(solvers), 4)) as executor:
            future_to_solver = {executor.submit(run_solver, solver): solver for solver in solvers}
            
            for future in as_completed(future_to_solver):
                result = future.result()
                if result:
                    solver_name, solution, quality, exec_time = result
                    candidate_solutions.append((solver_name, solution, quality))
                    print(f"    {solver_name}: {quality:.2f}ms, {len(solution)} tasks, {exec_time:.2f}s")
                    
                    if quality < self.best_score:
                        self.best_score = quality
                        self.best_solution = solution
        
        return candidate_solutions
    
    def _run_solvers_sequential(self, ir: CollectiveIR, solvers: List[SolverBasedOptimizationPass]) -> List[Tuple[str, Dict[int, Task], float]]:
        candidate_solutions = []
        
        for solver in solvers:
            solver_name = solver.name
            print(f"  Running solver: {solver_name}")
            
            try:
                solver_start = time.time()
                candidate_ir = solver.run(ir)
                solver_time = time.time() - solver_start
                
                if candidate_ir and candidate_ir.task_map.tasks:
                    candidate_time = self._evaluate_solution_quality(candidate_ir)
                    
                    candidate_solutions.append((solver_name, candidate_ir.task_map.tasks, candidate_time))
                    
                    self.solver_results[solver_name] = {
                        'success': True,
                        'execution_time': solver_time,
                        'solution_quality': candidate_time,
                        'task_count': len(candidate_ir.task_map.tasks)
                    }
                    
                    print(f"    {solver_name}: {candidate_time:.2f}ms, {len(candidate_ir.task_map.tasks)} tasks, {solver_time:.2f}s")
                    
                    if candidate_time < self.best_score:
                        self.best_score = candidate_time
                        self.best_solution = candidate_ir.task_map.tasks
                else:
                    self.solver_results[solver_name] = {
                        'success': False,
                        'error': 'No valid solution generated'
                    }
                    print(f"    {solver_name}: Failed - No valid solution")
                    
            except Exception as e:
                self.solver_results[solver_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"    {solver_name}: Failed - {e}")
        
        return candidate_solutions
    
    def _evaluate_solution_quality(self, ir: CollectiveIR) -> float:
        try:
            completion_time = ir.estimate_completion_time()
            
            task_count = len(ir.task_map.tasks)
            device_count = len(ir.cluster.devices_by_id)
            
            efficiency_penalty = 0.0
            if device_count > 0:
                avg_device_utilization = completion_time / (device_count * max(1, completion_time))
                efficiency_penalty = (1 - avg_device_utilization) * 10
            
            complexity_penalty = task_count * 0.001
            
            return completion_time + efficiency_penalty + complexity_penalty
            
        except:
            return float('inf')
    
    def _select_best_solution(self, ir: CollectiveIR, candidate_solutions: List[Tuple[str, Dict[int, Task], float]], 
                            characteristics: Dict[str, Any]) -> Dict[int, Task]:
        
        if self.selection_strategy == "adaptive_ensemble":
            return self._adaptive_ensemble_selection(ir, candidate_solutions, characteristics)
        elif self.selection_strategy == "quality_based":
            return self._quality_based_selection(candidate_solutions)
        elif self.selection_strategy == "robustness_based":
            return self._robustness_based_selection(candidate_solutions)
        elif self.selection_strategy == "fastest_valid":
            return self._fastest_valid_selection(candidate_solutions)
        else:
            return self._quality_based_selection(candidate_solutions)
    
    def _adaptive_ensemble_selection(self, ir: CollectiveIR, candidate_solutions: List[Tuple[str, Dict[int, Task], float]],
                                   characteristics: Dict[str, Any]) -> Dict[int, Task]:
        print("  Using adaptive ensemble selection")
        
        weights = self._calculate_adaptive_weights(candidate_solutions, characteristics)
        
        if self._should_use_solution_fusion(characteristics):
            return self._solution_fusion(ir, candidate_solutions, weights)
        else:
            return self._weighted_ensemble_selection(candidate_solutions, weights)
    
    def _calculate_adaptive_weights(self, candidate_solutions: List[Tuple[str, Dict[int, Task], float]],
                                  characteristics: Dict[str, Any]) -> Dict[str, float]:
        weights = {}
        performances = {name: quality for name, _, quality in candidate_solutions}
        
        if not performances:
            return weights
        
        min_perf = min(performances.values())
        max_perf = max(performances.values())
        
        if max_perf == min_perf:
            equal_weight = 1.0 / len(performances)
            return {name: equal_weight for name in performances.keys()}
        
        base_weights = {}
        for name, perf in performances.items():
            normalized_perf = (max_perf - perf) / (max_perf - min_perf)
            base_weights[name] = normalized_perf
        
        characteristics_weights = self._adjust_weights_by_characteristics(base_weights, characteristics)
        
        total_weight = sum(characteristics_weights.values())
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in characteristics_weights.items()}
        
        print(f"  Solver weights: {weights}")
        return weights
    
    def _adjust_weights_by_characteristics(self, base_weights: Dict[str, float], 
                                         characteristics: Dict[str, Any]) -> Dict[str, float]:
        adjusted_weights = base_weights.copy()
        
        num_tasks = characteristics['num_tasks']
        graph_density = characteristics['graph_density']
        heterogeneity = characteristics['heterogeneity']
        
        for solver_name in adjusted_weights.keys():
            weight_multiplier = 1.0
            
            if "MST" in solver_name and heterogeneity > 0.2:
                weight_multiplier *= 1.5
            
            if "SMT" in solver_name and num_tasks <= 30 and graph_density < 0.5:
                weight_multiplier *= 1.3
            
            if "MILP" in solver_name and num_tasks <= 100:
                weight_multiplier *= 1.2
            
            if "Collective" in solver_name and num_tasks > 50:
                weight_multiplier *= 1.4
            
            adjusted_weights[solver_name] *= weight_multiplier
        
        return adjusted_weights
    
    def _should_use_solution_fusion(self, characteristics: Dict[str, Any]) -> bool:
        num_tasks = characteristics['num_tasks']
        graph_density = characteristics['graph_density']
        
        return num_tasks <= 100 and graph_density < 0.8
    
    def _solution_fusion(self, ir: CollectiveIR, candidate_solutions: List[Tuple[str, Dict[int, Task], float]],
                       weights: Dict[str, float]) -> Dict[int, Task]:
        print("  Using solution fusion technique")
        
        base_solution = self.best_solution
        
        for solver_name, candidate_solution, candidate_quality in candidate_solutions:
            weight = weights.get(solver_name, 0)
            
            if weight > 0.1 and candidate_quality < self.best_score * 1.2:
                base_solution = self._fuse_solutions(base_solution, candidate_solution, ir)
        
        return base_solution
    
    def _fuse_solutions(self, base_solution: Dict[int, Task], candidate_solution: Dict[int, Task],
                       ir: CollectiveIR) -> Dict[int, Task]:
        fused_solution = base_solution.copy()
        
        base_tasks = set(base_solution.keys())
        candidate_tasks = set(candidate_solution.keys())
        
        common_tasks = base_tasks.intersection(candidate_tasks)
        
        for task_id in common_tasks:
            base_task = base_solution[task_id]
            candidate_task = candidate_solution[task_id]
            
            if candidate_task.total_estimated_duration < base_task.total_estimated_duration * 0.9:
                fused_solution[task_id] = candidate_task
        
        new_tasks = candidate_tasks - base_tasks
        for task_id in new_tasks:
            if task_id not in fused_solution:
                fused_solution[task_id] = candidate_solution[task_id]
        
        return fused_solution
    
    def _weighted_ensemble_selection(self, candidate_solutions: List[Tuple[str, Dict[int, Task], float]],
                                   weights: Dict[str, float]) -> Dict[int, Task]:
        best_weighted_score = float('inf')
        best_solution = None
        
        for solver_name, solution, quality in candidate_solutions:
            weight = weights.get(solver_name, 0)
            weighted_score = quality * (1.0 / weight) if weight > 0 else float('inf')
            
            if weighted_score < best_weighted_score:
                best_weighted_score = weighted_score
                best_solution = solution
        
        return best_solution or self.best_solution
    
    def _quality_based_selection(self, candidate_solutions: List[Tuple[str, Dict[int, Task], float]]) -> Dict[int, Task]:
        return min(candidate_solutions, key=lambda x: x[2])[1]
    
    def _robustness_based_selection(self, candidate_solutions: List[Tuple[str, Dict[int, Task], float]]) -> Dict[int, Task]:
        best_robustness_score = float('inf')
        best_solution = None
        
        for solver_name, solution, quality in candidate_solutions:
            task_count = len(solution)
            robustness_score = quality * (1 + 0.05 * np.log(task_count + 1))
            
            if robustness_score < best_robustness_score:
                best_robustness_score = robustness_score
                best_solution = solution
        
        return best_solution
    
    def _fastest_valid_selection(self, candidate_solutions: List[Tuple[str, Dict[int, Task], float]]) -> Dict[int, Task]:
        valid_solutions = [(name, sol, qual) for name, sol, qual in candidate_solutions if qual < float('inf')]
        
        if not valid_solutions:
            return self.best_solution
        
        fastest_time = float('inf')
        fastest_solution = None
        
        for solver_name, solution, quality in valid_solutions:
            solver_time = self.solver_results.get(solver_name, {}).get('execution_time', float('inf'))
            
            if solver_time < fastest_time and quality <= self.best_score * (1 + self.quality_threshold):
                fastest_time = solver_time
                fastest_solution = solution
        
        return fastest_solution or self.best_solution
    
    def _ensemble_refinement(self, ir: CollectiveIR, base_solution: Dict[int, Task],
                           candidate_solutions: List[Tuple[str, Dict[int, Task], float]]) -> Dict[int, Task]:
        print("  Applying ensemble refinement")
        
        refined_solution = base_solution.copy()
        
        scheduler = AdvancedTaskScheduler(ir)
        scheduled_ir = scheduler.schedule_tasks("critical_path")
        
        makespan_before = self._evaluate_solution_quality(scheduled_ir)
        
        for solver_name, candidate_solution, candidate_quality in candidate_solutions:
            if candidate_quality < makespan_before * 0.95:
                candidate_ir = ir.copy() if hasattr(ir, 'copy') else ir
                candidate_ir.task_map.tasks = candidate_solution
                scheduled_candidate = scheduler.schedule_tasks("critical_path")
                
                makespan_candidate = self._evaluate_solution_quality(scheduled_candidate)
                
                if makespan_candidate < makespan_before * 0.98:
                    refined_solution = candidate_solution
                    break
        
        return refined_solution
    
    def _comprehensive_fallback(self, ir: CollectiveIR) -> Dict[int, Task]:
        print("  Using comprehensive fallback strategy")
        
        from ...passes.comprehensive_optimization import ComprehensiveOptimizationPass
        
        comprehensive_pass = ComprehensiveOptimizationPass(optimization_level=2)
        fallback_ir = comprehensive_pass.run(ir)
        
        scheduler = AdvancedTaskScheduler(fallback_ir)
        scheduled_ir = scheduler.schedule_tasks("critical_path")
        
        return scheduled_ir.task_map.tasks
    
    def get_hybrid_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_solvers': len(self.solvers),
            'successful_solvers': sum(1 for r in self.solver_results.values() if r.get('success', False)),
            'best_solution_score': self.best_score,
            'selection_strategy': self.selection_strategy,
            'ensemble_used': self.ensemble_weighting,
            'parallel_execution': self.parallel_execution,
            'solver_details': self.solver_results
        }
        
        if self.best_solution:
            stats['best_solution_task_count'] = len(self.best_solution)
        
        return stats
    
    def set_solver_timeout(self, timeout: int):
        self.time_limit = timeout
        for solver in self.solvers:
            if hasattr(solver, 'time_limit'):
                solver.time_limit = min(timeout // 2, solver.time_limit)
