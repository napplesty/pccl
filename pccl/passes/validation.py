from .base import Pass
from ..ir.core import CollectiveIR, PrimitiveOpType, Chunk
from typing import Set, Dict, List

class ValidationPass(Pass):
    def __init__(self):
        super().__init__("validation")
    
    def run(self, ir: CollectiveIR) -> bool:
        if not self._validate_pre_post_conditions(ir):
            return False
        
        if not self._validate_device_topology(ir):
            return False
        
        if not self._validate_operation_dag(ir):
            return False
        
        if not self._validate_chunk_consistency(ir):
            return False
        
        return True
    
    def _validate_pre_post_conditions(self, ir: CollectiveIR) -> bool:
        if len(ir.precondition) != len(ir.postcondition):
            return False
        
        total_data_pre = sum(chunk.data_size for chunk in ir.precondition)
        total_data_post = sum(chunk.data_size for chunk in ir.postcondition)
        
        if total_data_pre != total_data_post:
            return False
        
        return True
    
    def _validate_device_topology(self, ir: CollectiveIR) -> bool:
        return ir.validate_topology()
    
    def _validate_operation_dag(self, ir: CollectiveIR) -> bool:
        return ir.validate_dag()
    
    def _validate_chunk_consistency(self, ir: CollectiveIR) -> bool:
        chunk_states = {}
        
        for rank, chunk in enumerate(ir.precondition):
            chunk_key = (chunk.cur_device_id, chunk.offset)
            chunk_states[chunk_key] = chunk.copy()
        
        operations = ir.get_operation_sequence()
        
        for op in operations:
            if op.op_type in [PrimitiveOpType.COPY, PrimitiveOpType.REDUCE]:
                if op.src_chunk and op.tgt_chunk:
                    src_key = (op.src_chunk.cur_device_id, op.src_chunk.offset)
                    tgt_key = (op.tgt_chunk.cur_device_id, op.tgt_chunk.offset)
                    
                    if src_key not in chunk_states:
                        return False
                    
                    if op.op_type == PrimitiveOpType.COPY:
                        chunk_states[tgt_key] = op.tgt_chunk.copy()
                    else:
                        if tgt_key not in chunk_states:
                            return False
                        
                        new_reduced = chunk_states[src_key].reduced_ranks.union(
                            chunk_states[tgt_key].reduced_ranks
                        )
                        chunk_states[tgt_key].reduced_ranks = new_reduced
        
        for rank, expected_chunk in enumerate(ir.postcondition):
            chunk_key = (expected_chunk.cur_device_id, expected_chunk.offset)
            if chunk_key not in chunk_states:
                return False
            
            actual_chunk = chunk_states[chunk_key]
            if (actual_chunk.reduced_ranks != expected_chunk.reduced_ranks or
                actual_chunk.cur_device_id != expected_chunk.cur_device_id or
                actual_chunk.data_size != expected_chunk.data_size or
                actual_chunk.offset != expected_chunk.offset):
                return False
        
        return True
