from ..ir.core import CollectiveIR, PrimitiveOpType, Chunk

def generate_ring_algorithm(ir: CollectiveIR) -> bool:
    num_ranks = len(ir.precondition)
    if num_ranks < 2:
        return False
    
    # For allreduce, each rank has one chunk in precondition
    data_size = ir.precondition[0].data_size if ir.precondition else 0
    
    operations = []
    
    # Phase 1: Reduce-scatter phase
    for step in range(num_ranks - 1):
        for rank in range(num_ranks):
            src_rank = rank
            dst_rank = (rank + 1) % num_ranks
            
            # Create source and target chunks for reduce operation
            src_chunk = Chunk(
                reduced_ranks={src_rank},
                cur_device_id=src_rank,
                data_size=data_size,
                offset=0
            )
            
            tgt_chunk = Chunk(
                reduced_ranks={src_rank, dst_rank},
                cur_device_id=dst_rank,
                data_size=data_size,
                offset=0
            )
            
            reduce_op = {
                "type": PrimitiveOpType.REDUCE,
                "src_chunk": src_chunk,
                "tgt_chunk": tgt_chunk,
                "src_device": src_rank,
                "tgt_device": dst_rank,
                "step": step,
                "phase": "reduce_scatter"
            }
            operations.append(reduce_op)
    
    # Phase 2: Allgather phase
    for step in range(num_ranks - 1):
        for rank in range(num_ranks):
            src_rank = (rank - 1) % num_ranks
            dst_rank = rank
            
            # Create source and target chunks for copy operation
            src_chunk = Chunk(
                reduced_ranks=set(range(num_ranks)),  # All ranks after reduce phase
                cur_device_id=src_rank,
                data_size=data_size,
                offset=0
            )
            
            tgt_chunk = Chunk(
                reduced_ranks=set(range(num_ranks)),
                cur_device_id=dst_rank,
                data_size=data_size,
                offset=0
            )
            
            copy_op = {
                "type": PrimitiveOpType.COPY,
                "src_chunk": src_chunk,
                "tgt_chunk": tgt_chunk,
                "src_device": src_rank,
                "tgt_device": dst_rank,
                "step": step,
                "phase": "allgather"
            }
            operations.append(copy_op)
    
    _build_operation_dag(ir, operations)
    return True

def _build_operation_dag(ir: CollectiveIR, operations: list):
    # Sort operations by phase and step
    operations.sort(key=lambda op: (
        0 if op.get("phase") == "reduce_scatter" else 1,
        op.get("step", 0),
        op.get("src_device", 0)
    ))
    
    prev_ops_by_device = {}
    
    for op_dict in operations:
        dependencies = []
        src_device = op_dict.get("src_device")
        tgt_device = op_dict.get("tgt_device")
        
        # Add dependency from previous operation on target device
        if tgt_device in prev_ops_by_device:
            dependencies.append(prev_ops_by_device[tgt_device])
        
        # Add dependency from previous operation on source device (for reduce operations)
        if src_device in prev_ops_by_device and op_dict["type"] == PrimitiveOpType.REDUCE:
            dependencies.append(prev_ops_by_device[src_device])
        
        op_id = ir.create_operation(
            op_dict["type"],
            src_chunk=op_dict["src_chunk"],
            tgt_chunk=op_dict["tgt_chunk"],
            src_device=src_device,
            tgt_device=tgt_device,
            dependencies=dependencies
        )
        
        # Update the last operation for target device
        prev_ops_by_device[tgt_device] = op_id
