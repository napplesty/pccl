from .base import Pass
from ..ir.core import CollectiveIR, PrimitiveOpType, Chunk, PrimitiveOp
from typing import List, Dict

class CanonicalizationPass(Pass):
    def __init__(self):
        super().__init__("canonicalization")
    
    def run(self, ir: CollectiveIR) -> bool:
        self._remove_redundant_operations(ir)
        self._simplify_dependencies(ir)
        self._normalize_chunk_offsets(ir)
        return True
    
    def _remove_redundant_operations(self, ir: CollectiveIR):
        operations = ir.get_operation_sequence()
        chunks_tracker = {}
        
        for op in operations:
            if op.op_type in [PrimitiveOpType.COPY, PrimitiveOpType.REDUCE]:
                if op.tgt_chunk:
                    chunk_key = (op.tgt_chunk.cur_device_id, op.tgt_chunk.offset)
                    
                    if chunk_key in chunks_tracker:
                        last_op_id = chunks_tracker[chunk_key]
                        
                        successors = ir.op_dag.get_successors(last_op_id)
                        if len(successors) == 1 and successors[0] == op.op_id:
                            ir.op_dag.remove_edge(last_op_id, op.op_id)
                            op.dependencies = [dep for dep in op.dependencies if dep != last_op_id]
                    
                    chunks_tracker[chunk_key] = op.op_id
    
    def _simplify_dependencies(self, ir: CollectiveIR):
        operations = ir.get_operation_sequence()
        
        for op in operations:
            if len(op.dependencies) > 1:
                transitive_deps = set()
                for dep_id in op.dependencies:
                    transitive_deps.update(self._get_transitive_dependencies(ir, dep_id))
                
                direct_deps = set(op.dependencies)
                redundant_deps = direct_deps.intersection(transitive_deps)
                
                if redundant_deps:
                    op.dependencies = [dep for dep in op.dependencies if dep not in redundant_deps]
                    
                    for redundant_dep in redundant_deps:
                        ir.op_dag.remove_edge(redundant_dep, op.op_id)
    
    def _get_transitive_dependencies(self, ir: CollectiveIR, start_id: int):
        visited = set()
        stack = [start_id]
        
        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            
            predecessors = ir.op_dag.get_predecessors(current_id)
            stack.extend(predecessors)
        
        visited.remove(start_id)
        return visited
    
    def _normalize_chunk_offsets(self, ir: CollectiveIR):
        offset_map = {}
        current_offset = 0
        
        for chunk in ir.precondition:
            if chunk.offset not in offset_map:
                offset_map[chunk.offset] = current_offset
                current_offset += chunk.data_size
        
        def update_chunk_offsets(chunk_list):
            for chunk in chunk_list:
                if chunk.offset in offset_map:
                    chunk.offset = offset_map[chunk.offset]
        
        update_chunk_offsets(ir.precondition)
        update_chunk_offsets(ir.postcondition)
        
        for op in ir.ops.values():
            if op.src_chunk and op.src_chunk.offset in offset_map:
                op.src_chunk.offset = offset_map[op.src_chunk.offset]
            if op.tgt_chunk and op.tgt_chunk.offset in offset_map:
                op.tgt_chunk.offset = offset_map[op.tgt_chunk.offset]
