from .base import Pass
from ..ir.core import CollectiveIR, PrimitiveOpType, Chunk, PrimitiveOp
from typing import List, Dict, Tuple

class ChunkOptimizationPass(Pass):
    def __init__(self, chunk_size: int = 1024):
        super().__init__("chunk_optimization")
        self.chunk_size = chunk_size
    
    def run(self, ir: CollectiveIR) -> bool:
        self._split_large_chunks(ir)
        self._add_temporary_chunks(ir)
        self._optimize_chunk_placement(ir)
        return True
    
    def _split_large_chunks(self, ir: CollectiveIR):
        new_precondition = []
        new_postcondition = []
        
        for chunk in ir.precondition:
            if chunk.data_size > self.chunk_size:
                num_splits = (chunk.data_size + self.chunk_size - 1) // self.chunk_size
                for i in range(num_splits):
                    split_size = min(self.chunk_size, chunk.data_size - i * self.chunk_size)
                    split_chunk = Chunk(
                        reduced_ranks=chunk.reduced_ranks.copy(),
                        cur_device_id=chunk.cur_device_id,
                        data_size=split_size,
                        offset=chunk.offset + i * self.chunk_size
                    )
                    new_precondition.append(split_chunk)
            else:
                new_precondition.append(chunk.copy())
        
        for chunk in ir.postcondition:
            if chunk.data_size > self.chunk_size:
                num_splits = (chunk.data_size + self.chunk_size - 1) // self.chunk_size
                for i in range(num_splits):
                    split_size = min(self.chunk_size, chunk.data_size - i * self.chunk_size)
                    split_chunk = Chunk(
                        reduced_ranks=chunk.reduced_ranks.copy(),
                        cur_device_id=chunk.cur_device_id,
                        data_size=split_size,
                        offset=chunk.offset + i * self.chunk_size
                    )
                    new_postcondition.append(split_chunk)
            else:
                new_postcondition.append(chunk.copy())
        
        ir.set_precondition(new_precondition)
        ir.set_postcondition(new_postcondition)
    
    def _add_temporary_chunks(self, ir: CollectiveIR):
        operations = ir.get_operation_sequence()
        bandwidth_matrix = ir.get_device_bandwidth_matrix()
        
        for i, op in enumerate(operations):
            if (op.op_type == PrimitiveOpType.COPY and op.src_chunk and op.tgt_chunk and
                op.src_chunk.cur_device_id != op.tgt_chunk.cur_device_id):
                
                src_device = op.src_chunk.cur_device_id
                tgt_device = op.tgt_chunk.cur_device_id
                
                if (src_device, tgt_device) not in bandwidth_matrix:
                    continue
                
                direct_bandwidth = bandwidth_matrix[(src_device, tgt_device)]
                
                for intermediate_device in ir.devices:
                    if (intermediate_device != src_device and intermediate_device != tgt_device and
                        (src_device, intermediate_device) in bandwidth_matrix and
                        (intermediate_device, tgt_device) in bandwidth_matrix):
                        
                        indirect_bandwidth = min(
                            bandwidth_matrix[(src_device, intermediate_device)],
                            bandwidth_matrix[(intermediate_device, tgt_device)]
                        )
                        
                        if indirect_bandwidth > direct_bandwidth * 1.2:
                            self._insert_intermediate_copy(ir, op, intermediate_device)
                            break
    
    def _insert_intermediate_copy(self, ir: CollectiveIR, original_op: PrimitiveOp, intermediate_device: int):
        temp_chunk = Chunk(
            reduced_ranks=original_op.src_chunk.reduced_ranks.copy(),
            cur_device_id=intermediate_device,
            data_size=original_op.src_chunk.data_size,
            offset=original_op.src_chunk.offset
        )
        
        first_copy_id = ir.create_operation(
            PrimitiveOpType.COPY,
            src_chunk=original_op.src_chunk,
            tgt_chunk=temp_chunk,
            dependencies=original_op.dependencies
        )
        
        second_copy_id = ir.create_operation(
            PrimitiveOpType.COPY,
            src_chunk=temp_chunk,
            tgt_chunk=original_op.tgt_chunk,
            dependencies=[first_copy_id]
        )
        
        successors = ir.op_dag.get_successors(original_op.op_id)
        for succ_id in successors:
            ir.op_dag.remove_edge(original_op.op_id, succ_id)
            ir.op_dag.add_edge(second_copy_id, succ_id)
            ir.ops[succ_id].dependencies = [
                dep if dep != original_op.op_id else second_copy_id 
                for dep in ir.ops[succ_id].dependencies
            ]
        
        ir.op_dag.nodes.pop(original_op.op_id, None)
        ir.ops.pop(original_op.op_id, None)
    
    def _optimize_chunk_placement(self, ir: CollectiveIR):
        pass
