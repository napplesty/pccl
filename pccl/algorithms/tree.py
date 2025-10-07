from ..ir.core import CollectiveIR, PrimitiveOpType, Chunk
import math

def generate_tree_algorithm(ir: CollectiveIR) -> bool:
    num_ranks = len(ir.precondition)
    if num_ranks < 2:
        return False
    
    chunks_per_rank = len(ir.precondition) // num_ranks
    if chunks_per_rank == 0:
        return False
    
    operations = []
    
    for chunk_idx in range(chunks_per_rank):
        chunk_ops = _generate_tree_for_chunk(ir, chunk_idx, num_ranks)
        operations.extend(chunk_ops)
    
    _build_operation_dag(ir, operations)
    return True

def _generate_tree_for_chunk(ir: CollectiveIR, chunk_idx: int, num_ranks: int) -> list:
    operations = []
    tree_depth = math.ceil(math.log2(num_ranks))
    
    for depth in range(tree_depth):
        stride = 2 ** depth
        for group_start in range(0, num_ranks, 2 * stride):
            for offset in range(stride):
                src_rank = group_start + offset
                dst_rank = group_start + offset + stride
                
                if dst_rank < num_ranks:
                    src_chunk_idx = chunk_idx * num_ranks + src_rank
                    dst_chunk_idx = chunk_idx * num_ranks + dst_rank
                    
                    if src_chunk_idx < len(ir.precondition) and dst_chunk_idx < len(ir.precondition):
                        src_chunk = ir.precondition[src_chunk_idx]
                        dst_chunk = ir.precondition[dst_chunk_idx]
                        
                        reduce_op = {
                            "type": PrimitiveOpType.REDUCE,
                            "src_chunk": src_chunk,
                            "tgt_chunk": dst_chunk,
                            "depth": depth,
                            "group": group_start
                        }
                        operations.append(reduce_op)
    
    root_rank = 0
    for depth in reversed(range(tree_depth)):
        stride = 2 ** depth
        for group_start in range(0, num_ranks, 2 * stride):
            for offset in range(stride):
                src_rank = group_start + offset
                dst_rank = group_start + offset + stride
                
                if dst_rank < num_ranks:
                    src_chunk_idx = chunk_idx * num_ranks + src_rank
                    dst_chunk_idx = chunk_idx * num_ranks + dst_rank
                    
                    if src_chunk_idx < len(ir.precondition) and dst_chunk_idx < len(ir.precondition):
                        src_chunk = ir.precondition[src_chunk_idx]
                        dst_chunk = ir.precondition[dst_chunk_idx]
                        
                        copy_op = {
                            "type": PrimitiveOpType.COPY,
                            "src_chunk": src_chunk,
                            "tgt_chunk": dst_chunk,
                            "depth": depth + tree_depth,
                            "group": group_start
                        }
                        operations.append(copy_op)
    
    return operations

def _build_operation_dag(ir: CollectiveIR, operations: list):
    operations.sort(key=lambda op: (op.get("depth", 0), op.get("group", 0)))
    
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
