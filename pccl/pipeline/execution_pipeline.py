import time
import json
from typing import Dict, List, Optional, Any
from ..collective_ir.core.ir import CollectiveIR
from ..collective_ir.management.manager import PassManager, create_default_pass_manager
from ..collective_ir.passes.comprehensive_optimization import ComprehensiveOptimizationPass
from ..primitive_ir.transformer import convert_collective_to_primitive
from ..primitive_ir.primitive_ir import PrimitiveGraph
from ..cccl import executeGraph, initializeRuntime, shutdownRuntime, get_global_config
from .config import PipelineConfig, OptimizationLevel, ExecutionStrategy

class ExecutionResult:
    def __init__(self, success: bool, execution_time_ms: float, 
                 results: Optional[Dict] = None, error: Optional[str] = None):
        self.success = success
        self.execution_time_ms = execution_time_ms
        self.results = results or {}
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'execution_time_ms': self.execution_time_ms,
            'results': self.results,
            'error': self.error
        }

class CollectiveExecutionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.pass_manager = self._create_pass_manager()
        self.runtime_initialized = False
        self.execution_stats = {}
    
    def _create_pass_manager(self) -> PassManager:
        manager = PassManager()
        
        if self.config.optimization_level.value >= OptimizationLevel.BASIC.value:
            from ..collective_ir.passes.static.canonical import CanonicalPass
            manager.add_pass(CanonicalPass())
        
        if self.config.optimization_level.value >= OptimizationLevel.STANDARD.value:
            from ..collective_ir.passes.static.memory import MemoryOptimizationPass
            from ..collective_ir.passes.static.algorithms.collective import CollectiveOptimizationPass
            manager.add_pass(MemoryOptimizationPass())
            manager.add_pass(CollectiveOptimizationPass())
        
        if self.config.optimization_level.value >= OptimizationLevel.ADVANCED.value:
            from ..collective_ir.passes.performance_modeling import PerformanceModelingPass
            manager.add_pass(PerformanceModelingPass())
            manager.add_pass(ComprehensiveOptimizationPass(optimization_level=2))
        
        if self.config.optimization_level.value >= OptimizationLevel.AGGRESSIVE.value:
            from ..collective_ir.passes.solver_based.hybrid_solver import HybridSolverPass
            manager.add_pass(HybridSolverPass())
        
        return manager
    
    def execute(self, collective_ir: CollectiveIR) -> ExecutionResult:
        start_time = time.time()
        
        try:
            if not self.runtime_initialized:
                initializeRuntime()
                self.runtime_initialized = True
            
            if self.config.enable_verification:
                self._verify_collective_ir(collective_ir)
            
            optimized_ir = self._optimize_collective_ir(collective_ir)
            primitive_graphs = self._convert_to_primitive_ir(optimized_ir)
            execution_results = self._execute_primitive_graphs(primitive_graphs)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                execution_time_ms=execution_time,
                results=execution_results
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=False,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def _verify_collective_ir(self, collective_ir: CollectiveIR):
        from ..collective_ir.verification.verifiers import CompositeVerifier, BasicIRVerifier
        verifier = CompositeVerifier([BasicIRVerifier()])
        result = verifier.verify(collective_ir)
        
        if not result.is_valid:
            raise ValueError(f"CollectiveIR verification failed: {result.errors}")
    
    def _optimize_collective_ir(self, collective_ir: CollectiveIR) -> CollectiveIR:
        print(f"Optimizing CollectiveIR (level: {self.config.optimization_level.name})...")
        return self.pass_manager.run_passes(collective_ir)
    
    def _convert_to_primitive_ir(self, collective_ir: CollectiveIR) -> Dict[int, PrimitiveGraph]:
        device_ids = list(collective_ir.cluster.devices_by_id.keys())
        primitive_graphs = {}
        
        for rank in device_ids:
            primitive_graph = convert_collective_to_primitive(collective_ir, rank)
            primitive_graphs[rank] = primitive_graph
        
        print(f"Converted to {len(primitive_graphs)} PrimitiveGraphs")
        return primitive_graphs
    
    def _execute_primitive_graphs(self, primitive_graphs: Dict[int, PrimitiveGraph]) -> Dict[int, Any]:
        results = {}
        
        if self.config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            results = self._execute_sequential(primitive_graphs)
        elif self.config.execution_strategy == ExecutionStrategy.PARALLEL:
            results = self._execute_parallel(primitive_graphs)
        elif self.config.execution_strategy == ExecutionStrategy.PIPELINED:
            results = self._execute_pipelined(primitive_graphs)
        
        return results
    
    def _execute_sequential(self, primitive_graphs: Dict[int, PrimitiveGraph]) -> Dict[int, Any]:
        results = {}
        for rank, graph in primitive_graphs.items():
            graph_json = graph.to_json()
            result = executeGraph(graph_json, self.config.timeout_ms)
            results[rank] = result
        return results
    
    def _execute_parallel(self, primitive_graphs: Dict[int, PrimitiveGraph]) -> Dict[int, Any]:
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        results = {}
        lock = threading.Lock()
        
        def execute_single_graph(rank: int, graph: PrimitiveGraph):
            try:
                graph_json = graph.to_json()
                result = executeGraph(graph_json, self.config.timeout_ms)
                with lock:
                    results[rank] = result
            except Exception as e:
                with lock:
                    results[rank] = {'error': str(e)}
        
        with ThreadPoolExecutor(max_workers=len(primitive_graphs)) as executor:
            futures = []
            for rank, graph in primitive_graphs.items():
                future = executor.submit(execute_single_graph, rank, graph)
                futures.append(future)
            
            for future in futures:
                future.result()
        
        return results
    
    def _execute_pipelined(self, primitive_graphs: Dict[int, PrimitiveGraph]) -> Dict[int, Any]:
        results = {}
        
        graph_jsons = {rank: graph.to_json() for rank, graph in primitive_graphs.items()}
        combined_json = json.dumps(graph_jsons)
        
        result = executeGraph(combined_json, self.config.timeout_ms)
        
        for rank in primitive_graphs.keys():
            results[rank] = result.get(str(rank), {})
        
        return results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'runtime_initialized': self.runtime_initialized,
            'pass_manager_stats': self.pass_manager.get_execution_summary(),
            'execution_stats': self.execution_stats
        }
    
    def shutdown(self):
        if self.runtime_initialized:
            shutdownRuntime()
            self.runtime_initialized = False

def create_default_pipeline() -> CollectiveExecutionPipeline:
    config = PipelineConfig(
        optimization_level=OptimizationLevel.STANDARD,
        execution_strategy=ExecutionStrategy.PARALLEL,
        enable_verification=True,
        enable_profiling=False
    )
    return CollectiveExecutionPipeline(config)

def create_high_performance_pipeline() -> CollectiveExecutionPipeline:
    config = PipelineConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        execution_strategy=ExecutionStrategy.PIPELINED,
        enable_verification=True,
        enable_profiling=True
    )
    return CollectiveExecutionPipeline(config)
