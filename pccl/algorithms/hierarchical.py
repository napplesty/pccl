from ..ir.core import CollectiveIR, PrimitiveOpType, Chunk
import math

def generate_hierarchical_algorithm(ir: CollectiveIR) -> bool:
    num_ranks = len(ir.precondition)
    if num_ranks < 4:
        return False
    
    chunks_per_rank = len(ir.precondition) // num_ranks
    if chunks_per_rank == 0:
        return False
    
    nodes_per_group = int(math.sqrt(num_ranks))
    num_groups = (num_ranks + nodes_per_group - 1) // nodes_per_group
    
    operations = []
    
    for chunk_idx in range(chunks_per_rank):
        chunk_ops = _generate_hierarchical_for_chunk(ir, chunk_idx, num_ranks, nodes_per_group, num_groups)
        operations.extend(chunk_ops)
    
    _build_operation_dag(ir, operations)
    return True

def _generate_hierarchical_for_chunk(ir: CollectiveIR, chunk_idx: int, num_ranks: int, 
                                   nodes_per_group: int, num_groups: int) -> list:
    operations = []
    
    for group_id in range(num_groups):
        group_start = group_id * nodes_per_group
        group_end = min((group_id + 1) * nodes_per_group, num_ranks)
        group_size = group_end - group_start
        
        if group_size < 2:
            continue
        
        for rank_in_group in range(1, group_size):
            src_rank = group_start
            dst_rank = group_start + rank_in_group
            
            src_chunk_idx = chunk_idx * num_ranks + src_rank
            dst_chunk_idx = chunk_idx * num_ranks + dst_rank
            
            if src_chunk_idx < len(ir.precondition) and dst_chunk_idx < len(ir.precondition):
                src_chunk = ir.precondition[src_chunk_idx]
                dst_chunk = ir.precondition[dst_chunk_idx]
                
                reduce_op = {
                    "type": PrimitiveOpType.REDUCE,
                    "src_chunk": src_chunk,
                    "tgt_chunk": dst_chunk,
                    "phase": "intra_group_reduce",
                    "group": group_id
                }
                operations.append(reduce_op)
    
    for group_id in range(1, num_groups):
        src_group_leader = 0
        dst_group_leader = group_id * nodes_per_group
        
        src_chunk_idx = chunk_idx * num_ranks + src_group_leader
        dst_chunk_idx = chunk_idx * num_ranks + dst_group_leader
        
        if src_chunk_idx < len(ir.precondition) and dst_chunk_idx < len(ir.precondition):
            src_chunk = ir.precondition[src_chunk_idx]
            dst_chunk = ir.precondition[dst_chunk_idx]
            
            reduce_op = {
                "type": PrimitiveOpType.REDUCE,
                "src_chunk": src_chunk,
                "tgt_chunk": dst_chunk,
                "phase": "inter_group_reduce",
                "group": group_id
            }
            operations.append(reduce_op)
    
    for group_id in range(1, num_groups):
        src_group_leader = 0
        dst_group_leader = group_id * nodes_per_group
        
        src_chunk_idx = chunk_idx * num_ranks + src_group_leader
        dst_chunk_idx = chunk_idx * num_ranks + dst_group_leader
        
        if src_chunk_idx < len(ir.precondition) and dst_chunk_idx < len(ir.precondition):
            src_chunk = ir.precondition[src_chunk_idx]
            dst_chunk = ir.precondition[dst_chunk_idx]
            
            copy_op = {
                "type": PrimitiveOpType.COPY,
                "src_chunk": src_chunk,
                "tgt_chunk": dst_chunk,
                "phase": "inter_group_broadcast",
                "group": group_id
            }
            operations.append(copy_op)
    
    for group_id in range(num_groups):
        group_start = group_id * nodes_per_group
        group_end = min((group_id + 1) * nodes_per_group, num_ranks)
        group_size = group_end - group_start
        
        if group_size < 2:
            continue
        
        for rank_in_group in range(1, group_size):
            src_rank = group_start
            dst_rank = group_start + rank_in_group
            
            src_chunk_idx = chunk_idx * num_ranks + src_rank
            dst_chunk_idx = chunk_idx * num_ranks + dst_rank
            
            if src_chunk_idx < len(ir.precondition) and dst_chunk_idx < len(ir.precondition):
                src_chunk = ir.precondition[src_chunk_idx]
                dst_chunk = ir.precondition[dst_chunk_idx]
                
                copy_op = {
                    "type": PrimitiveOpType.COPY,
                    "src_chunk": src_chunk,
                    "tgt_chunk": dst_chunk,
                    "phase": "intra_group_broadcast",
                    "group": group_id
                }
                operations.append(copy_op)
    
    return operations

def _build_operation_dag(ir: CollectiveIR, operations: list):
    phase_order = ["intra_group_reduce", "inter_group_reduce", "inter_group_broadcast", "intra_group_broadcast"]
    operations.sort(key=lambda op: (phase_order.index(op.get("phase", "")), op.get("group", 0)))
    
    prev_ops_by_chunk = {}
    
    for op_dict in operations:
        dependencies = []
        tgt_chunk_key = (op_dict["tgt_chunk"].cur_device_id, op_dict["tgt_chunk"].offset)
        
        if tgt_chunk_key in prev_ops_by_chunk:
            dependencies.append(prev_ops_by_chunk[tgt_chunk_key])
        
        op_id = ir.create_operation(
            op_dict["type"],
            src_chunk=op_dict["src_chunk"],
            tgt_chunk=op_dict["tgt_chunk"],
            dependencies=dependencies
        )
        
        prev_ops_by_chunk[tgt_chunk_key] = op_id
