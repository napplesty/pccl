from .base import IRPass
from .performance_modeling import PerformanceModelingPass
from .static.canonical import CanonicalPass
from .static.memory import MemoryOptimizationPass
from .static.scheduling import TaskSchedulingPass
from .static.algorithms.collective import CollectiveOptimizationPass
from ..dependency_analysis.optimizer import DependencyOptimizer
from ..dependency_analysis.scheduler import AdvancedTaskScheduler
from ..memory_optimization.optimizer import AdvancedMemoryOptimizer
from ..core.ir import CollectiveIR

class ComprehensiveOptimizationPass(IRPass):

    def __init__(self, optimization_level: int = 2):
        self.optimization_level = optimization_level
        self.performance_pass = PerformanceModelingPass()
        self.canonical_pass = CanonicalPass()
        self.memory_pass = AdvancedMemoryOptimizer()
        self.collective_pass = CollectiveOptimizationPass()
    
    @property
    def name(self) -> str:
        return f"ComprehensiveOptimizationPass(level={self.optimization_level})"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        print(f"Running comprehensive optimization (level {self.optimization_level})...")
        print("  Phase 1: Performance modeling...")
        ir = self.performance_pass.run(ir)

        print("  Phase 2: Canonical optimization...")
        ir = self.canonical_pass.run(ir)
        
        if self.optimization_level >= 1:
            print("  Phase 3: Dependency optimization...")
            dependency_optimizer = DependencyOptimizer(ir)
            ir = dependency_optimizer.optimize_task_dependencies()
            
            print("  Phase 4: Memory optimization...")
            ir = self.memory_pass.run(ir)
        
        if self.optimization_level >= 2:
            print("  Phase 5: Collective algorithm optimization...")
            ir = self.collective_pass.run(ir)

            print("  Phase 6: Advanced scheduling...")
            scheduler = AdvancedTaskScheduler(ir)
            ir = scheduler.schedule_tasks("critical_path")

        print("  Final phase: Final canonical optimization...")
        ir = self.canonical_pass.run(ir)
        
        print("Comprehensive optimization completed!")
        return ir
