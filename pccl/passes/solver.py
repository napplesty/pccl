from .base import Pass
from ..ir.core import CollectiveIR, PrimitiveOpType, Chunk, PrimitiveOp
from typing import List, Dict, Set, Tuple
import itertools

class SolverPass(Pass):
    def __init__(self, max_iterations: int = 1000):
        super().__init__("solver")
        self.max_iterations = max_iterations
    
    def run(self, ir: CollectiveIR) -> bool:
        return self._solve_collective(ir)
    
    def _solve_collective(self, ir: CollectiveIR) -> bool:
        current_state = {i: chunk.copy() for i, chunk in enumerate(ir.precondition)}
        target_state = {i: chunk for i, chunk in enumerate(ir.postcondition)}
        
        operations = []
        iteration = 0
        
        while iteration < self.max_iterations:
            if self._check_state_matches_target(current_state, target_state):
                break
            
            operation_found = self._find_next_operation(ir, current_state, target_state, operations)
            if not operation_found:
                return False
            
            iteration += 1
        
        if iteration >= self.max_iterations:
            return False
        
        self._build_operation_dag(ir, operations)
        return True
    
    def _check_state_matches_target(self, current_state: Dict, target_state: Dict) -> bool:
        for rank, target_chunk in target_state.items():
            if rank not in current_state:
                return False
            
            current_chunk = current_state[rank]
            if (current_chunk.reduced_ranks != target_chunk.reduced_ranks or
                current_chunk.cur_device_id != target_chunk.cur_device_id):
                return False
        
        return True
    
    def _find_next_operation(self, ir: CollectiveIR, current_state: Dict, 
                           target_state: Dict, operations: List) -> bool:
        for src_rank, src_chunk in current_state.items():
            for tgt_rank, tgt_chunk in current_state.items():
                if src_rank == tgt_rank:
                    continue
                
                if self._is_valid_copy(src_chunk, tgt_chunk, target_state):
                    new_op = self._create_copy_operation(src_chunk, tgt_chunk)
                    operations.append(new_op)
                    
                    current_state[tgt_rank] = tgt_chunk.copy()
                    current_state[tgt_rank].cur_device_id = src_chunk.cur_device_id
                    return True
                
                if self._is_valid_reduce(src_chunk, tgt_chunk, target_state):
                    new_op = self._create_reduce_operation(src_chunk, tgt_chunk)
                    operations.append(new_op)
                    
                    current_state[tgt_rank] = tgt_chunk.copy()
                    current_state[tgt_rank].reduced_ranks = (
                        src_chunk.reduced_ranks.union(tgt_chunk.reduced_ranks)
                    )
                    return True
        
        return False
    
    def _is_valid_copy(self, src_chunk: Chunk, tgt_chunk: Chunk, target_state: Dict) -> bool:
        return (src_chunk.cur_device_id != tgt_chunk.cur_device_id and
                src_chunk.data_size == tgt_chunk.data_size and
                src_chunk.offset == tgt_chunk.offset)
    
    def _is_valid_reduce(self, src_chunk: Chunk, tgt_chunk: Chunk, target_state: Dict) -> bool:
        return (src_chunk.cur_device_id == tgt_chunk.cur_device_id and
                src_chunk.data_size == tgt_chunk.data_size and
                src_chunk.offset == tgt_chunk.offset and
                not src_chunk.reduced_ranks.issubset(tgt_chunk.reduced_ranks))
    
    def _create_copy_operation(self, src_chunk: Chunk, tgt_chunk: Chunk) -> Dict:
        return {
            "type": PrimitiveOpType.COPY,
            "src_chunk": src_chunk.copy(),
            "tgt_chunk": tgt_chunk.copy()
        }
    
    def _create_reduce_operation(self, src_chunk: Chunk, tgt_chunk: Chunk) -> Dict:
        return {
            "type": PrimitiveOpType.REDUCE,
            "src_chunk": src_chunk.copy(),
            "tgt_chunk": tgt_chunk.copy()
        }
    
    def _build_operation_dag(self, ir: CollectiveIR, operations: List):
        op_dependencies = {}
        
        for i, op in enumerate(operations):
            dependencies = []
            
            for j in range(i):
                prev_op = operations[j]
                if self._operations_conflict(op, prev_op):
                    dependencies.append(j)
            
            op_dependencies[i] = dependencies
        
        for i, op_dict in enumerate(operations):
            op_id = ir.create_operation(
                op_dict["type"],
                src_chunk=op_dict["src_chunk"],
                tgt_chunk=op_dict["tgt_chunk"],
                dependencies=op_dependencies[i]
            )
    
    def _operations_conflict(self, op1: Dict, op2: Dict) -> bool:
        chunks1 = {op1["src_chunk"], op1["tgt_chunk"]}
        chunks2 = {op2["src_chunk"], op2["tgt_chunk"]}
        
        return len(chunks1.intersection(chunks2)) > 0
