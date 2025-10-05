from typing import Set, List
from ..core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory
from ..core.enums import PrimitiveOpType, ChunkState
from .base import IRVerifier
from .results import VerificationResult

class BasicIRVerifier(IRVerifier):
    def verify(self, ir: CollectiveIR) -> VerificationResult:
        result = VerificationResult(is_valid=True)
        
        self._verify_devices(ir, result)
        self._verify_tasks(ir, result)
        self._verify_dependencies(ir, result)
        self._verify_collective_spec(ir, result)
        self._verify_memory_regions(ir, result)
        
        return result
    
    def _verify_devices(self, ir: CollectiveIR, result: VerificationResult):
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                if primitive.initiator.device_id not in ir.cluster.devices_by_id:
                    result.add_error(f"Task {task.task_id} uses device {primitive.initiator.device_id} not in cluster")
                
                for mem_region in primitive.memory_regions:
                    if mem_region.device.device_id not in ir.cluster.devices_by_id:
                        result.add_error(f"Memory region references device {mem_region.device.device_id} not in cluster")
    
    def _verify_tasks(self, ir: CollectiveIR, result: VerificationResult):
        for task_id, task in ir.task_map.tasks.items():
            if not task.primitives:
                result.add_warning(f"Task {task_id} has no primitives")
            
            for primitive in task.primitives:
                if not primitive.memory_regions:
                    result.add_error(f"Task {task_id} has primitive with no memory regions")
    
    def _verify_dependencies(self, ir: CollectiveIR, result: VerificationResult):
        visited = set()
        
        def check_cycle(task: Task, path: List[int]):
            if task.task_id in path:
                cycle = " -> ".join(str(t) for t in path[path.index(task.task_id):] + [task.task_id])
                result.add_error(f"Circular dependency detected: {cycle}")
                return
            
            if task.task_id in visited:
                return
            
            visited.add(task.task_id)
            path.append(task.task_id)
            
            for dep in task.dependencies:
                if dep.task_id not in ir.task_map.tasks:
                    result.add_error(f"Task {task.task_id} depends on non-existent task {dep.task_id}")
                else:
                    check_cycle(dep, path.copy())
        
        for task in ir.task_map.tasks.values():
            if task.task_id not in visited:
                check_cycle(task, [])
    
    def _verify_collective_spec(self, ir: CollectiveIR, result: VerificationResult):
        if ir.collective_spec.op_type != ir.collective_op:
            result.add_error(f"CollectiveSpec op_type {ir.collective_spec.op_type} doesn't match IR op_type {ir.collective_op}")
        
        if ir.collective_spec.data_size_gb != ir.data_size_gb:
            result.add_warning(f"CollectiveSpec data_size {ir.collective_spec.data_size_gb} doesn't match IR data_size {ir.data_size_gb}")
    
    def _verify_memory_regions(self, ir: CollectiveIR, result: VerificationResult):
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                for mem_region in primitive.memory_regions:
                    device = ir.cluster.get_device(mem_region.device.device_id)
                    if mem_region.size <= 0:
                        result.add_error(f"Memory region has invalid size {mem_region.size}")
                    
                    if mem_region.address < 0:
                        result.add_error(f"Memory region has negative address {mem_region.address}")

class ChunkStateVerifier(IRVerifier):
    def verify(self, ir: CollectiveIR) -> VerificationResult:
        result = VerificationResult(is_valid=True)
        
        current_states = ir.get_current_chunk_states()
        
        for precondition in ir.collective_spec.preconditions:
            if not precondition.is_satisfied(current_states):
                result.add_warning(f"Precondition not satisfied: {precondition}")
        
        return result

class PerformanceVerifier(IRVerifier):
    def verify(self, ir: CollectiveIR) -> VerificationResult:
        result = VerificationResult(is_valid=True)
        
        total_duration = ir.estimate_completion_time()
        if total_duration <= 0:
            result.add_warning("Estimated completion time is zero or negative")
        
        bandwidth_utilization = self._check_bandwidth_utilization(ir)
        if bandwidth_utilization < 0.1:
            result.add_warning(f"Low estimated bandwidth utilization: {bandwidth_utilization:.2f}")
        
        return result
    
    def _check_bandwidth_utilization(self, ir: CollectiveIR) -> float:
        total_comm_size = 0
        max_possible_comm = 0
        
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                if primitive.op_type in [PrimitiveOpType.COPY, PrimitiveOpType.REDUCE]:
                    total_comm_size += sum(region.size for region in primitive.memory_regions)
        
        for device in ir.cluster.devices_by_id.values():
            max_possible_comm += device.bandwidth_gbs * 1024 * 1024 * 1024
        
        if max_possible_comm == 0:
            return 0.0
        
        return total_comm_size / max_possible_comm

class CompositeVerifier(IRVerifier):
    def __init__(self, verifiers: List[IRVerifier]):
        self.verifiers = verifiers
    
    def verify(self, ir: CollectiveIR) -> VerificationResult:
        result = VerificationResult(is_valid=True)
        
        for verifier in self.verifiers:
            verifier_result = verifier.verify(ir)
            result.errors.extend(verifier_result.errors)
            result.warnings.extend(verifier_result.warnings)
            result.is_valid = result.is_valid and verifier_result.is_valid
        
        return result
