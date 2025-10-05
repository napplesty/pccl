from typing import List, Dict, Any, Optional
import time
from ..core.ir import CollectiveIR, IRVerificationError, PassExecutionError
from ..passes.base import IRPass
from ..verification.base import IRVerifier
from ..verification.verifiers import CompositeVerifier, BasicIRVerifier, ChunkStateVerifier, PerformanceVerifier
from .results import PassExecutionResult

class PassManager:
    def __init__(self, verifier: Optional[IRVerifier] = None):
        self.passes: List[IRPass] = []
        self.verifier = verifier or CompositeVerifier([
            BasicIRVerifier(),
            ChunkStateVerifier(),
            PerformanceVerifier()
        ])
        self.execution_history: List[PassExecutionResult] = []
    
    def add_pass(self, pass_obj: IRPass):
        self.passes.append(pass_obj)
    
    def add_verifier(self, verifier: IRVerifier):
        if isinstance(self.verifier, CompositeVerifier):
            self.verifier.verifiers.append(verifier)
        else:
            self.verifier = CompositeVerifier([self.verifier, verifier])
    
    def run_passes(self, ir: CollectiveIR, verify_each_pass: bool = True) -> CollectiveIR:
        current_ir = ir
        
        for pass_obj in self.passes:
            start_time = time.time()
            
            try:
                if verify_each_pass:
                    verification_result = self.verifier.verify(current_ir)
                    if not verification_result.is_valid:
                        raise IRVerificationError(f"IR verification failed before {pass_obj.name}: {verification_result.errors}")
                
                new_ir = pass_obj.run(current_ir)
                execution_time = time.time() - start_time
                
                result = PassExecutionResult(
                    success=True,
                    execution_time=execution_time,
                    input_ir=current_ir,
                    output_ir=new_ir,
                    metrics=self._collect_metrics(current_ir, new_ir)
                )
                
                self.execution_history.append(result)
                current_ir = new_ir
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = PassExecutionResult(
                    success=False,
                    execution_time=execution_time,
                    input_ir=current_ir,
                    output_ir=current_ir,
                    errors=[str(e)]
                )
                self.execution_history.append(result)
                raise PassExecutionError(f"Pass {pass_obj.name} failed: {e}")
        
        final_verification = self.verifier.verify(current_ir)
        if not final_verification.is_valid:
            raise IRVerificationError(f"Final IR verification failed: {final_verification.errors}")
        
        return current_ir
    
    def _collect_metrics(self, input_ir: CollectiveIR, output_ir: CollectiveIR) -> Dict[str, Any]:
        return {
            "input_task_count": len(input_ir.task_map.tasks),
            "output_task_count": len(output_ir.task_map.tasks),
            "input_estimated_time": input_ir.estimate_completion_time(),
            "output_estimated_time": output_ir.estimate_completion_time(),
            "task_reduction": len(input_ir.task_map.tasks) - len(output_ir.task_map.tasks),
            "time_improvement": input_ir.estimate_completion_time() - output_ir.estimate_completion_time()
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        summary = {
            "total_passes": len(self.passes),
            "successful_passes": sum(1 for r in self.execution_history if r.success),
            "failed_passes": sum(1 for r in self.execution_history if not r.success),
            "total_execution_time": sum(r.execution_time for r in self.execution_history),
            "pass_details": []
        }
        
        for i, result in enumerate(self.execution_history):
            pass_detail = {
                "pass_index": i,
                "success": result.success,
                "execution_time": result.execution_time,
                "metrics": result.metrics,
                "errors": result.errors,
                "warnings": result.warnings
            }
            summary["pass_details"].append(pass_detail)
        
        return summary
    
    def clear_history(self):
        self.execution_history.clear()

def create_default_pass_manager() -> PassManager:
    from ..passes.static.canonical import CanonicalPass
    from ..passes.static.memory import MemoryOptimizationPass
    from ..passes.static.algorithms.collective import CollectiveOptimizationPass
    from ..passes.static.scheduling import TaskSchedulingPass
    
    manager = PassManager()
    
    manager.add_pass(CanonicalPass())
    manager.add_pass(MemoryOptimizationPass())
    manager.add_pass(CollectiveOptimizationPass())
    manager.add_pass(TaskSchedulingPass("critical_path"))
    manager.add_pass(CanonicalPass())
    
    return manager
def create_optimization_pass_manager() -> PassManager:
    from ..passes.static.canonical import CanonicalPass
    from ..passes.static.memory import MemoryOptimizationPass
    from ..passes.static.algorithms.collective import CollectiveOptimizationPass
    from ..passes.static.scheduling import TaskSchedulingPass
    
    manager = PassManager()
    
    manager.add_pass(CanonicalPass())
    manager.add_pass(MemoryOptimizationPass())
    manager.add_pass(CollectiveOptimizationPass())
    manager.add_pass(TaskSchedulingPass("critical_path"))
    manager.add_pass(CanonicalPass())
    
    return manager

def create_verification_pass_manager() -> PassManager:
    manager = PassManager()
    
    verifier = CompositeVerifier([
        BasicIRVerifier(),
        ChunkStateVerifier(),
        PerformanceVerifier()
    ])
    manager.verifier = verifier
    
    return manager
