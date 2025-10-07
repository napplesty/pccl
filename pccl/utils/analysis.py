from ..ir.core import CollectiveIR, PrimitiveOpType
from typing import Dict, List, Tuple

def analyze_communication_pattern(ir: CollectiveIR) -> Dict:
    operations = ir.get_operation_sequence()
    pattern_analysis = {
        "total_operations": len(operations),
        "operation_breakdown": {},
        "communication_volume": 0,
        "critical_path_length": 0,
        "parallelism_opportunities": 0
    }
    
    for op_type in PrimitiveOpType:
        pattern_analysis["operation_breakdown"][op_type.name] = 0
    
    for op in operations:
        pattern_analysis["operation_breakdown"][op.op_type.name] += 1
        
        if op.op_type == PrimitiveOpType.COPY and op.src_chunk:
            pattern_analysis["communication_volume"] += op.src_chunk.data_size
    
    try:
        topological_order = ir.op_dag.topological_sort()
        pattern_analysis["critical_path_length"] = len(topological_order)
    except:
        pattern_analysis["critical_path_length"] = 0
    
    max_concurrent_ops = _calculate_max_concurrent_operations(ir)
    pattern_analysis["parallelism_opportunities"] = max_concurrent_ops
    
    return pattern_analysis

def _calculate_max_concurrent_operations(ir: CollectiveIR) -> int:
    operations = ir.get_operation_sequence()
    if not operations:
        return 0
    
    time_slots = {}
    
    for op in operations:
        start_time = _calculate_operation_start_time(ir, op.op_id)
        end_time = start_time + _estimate_operation_duration(ir, op)
        
        for time in range(start_time, end_time):
            if time not in time_slots:
                time_slots[time] = 0
            time_slots[time] += 1
    
    return max(time_slots.values()) if time_slots else 0

def _calculate_operation_start_time(ir: CollectiveIR, op_id: int) -> int:
    predecessors = ir.op_dag.get_predecessors(op_id)
    if not predecessors:
        return 0
    
    max_predecessor_time = 0
    for pred_id in predecessors:
        pred_time = _calculate_operation_start_time(ir, pred_id) + 1
        if pred_time > max_predecessor_time:
            max_predecessor_time = pred_time
    
    return max_predecessor_time

def _estimate_operation_duration(ir: CollectiveIR, op) -> int:
    if op.op_type == PrimitiveOpType.COPY and op.src_chunk:
        return max(1, op.src_chunk.data_size // 1024)
    elif op.op_type == PrimitiveOpType.REDUCE and op.tgt_chunk:
        return max(1, op.tgt_chunk.data_size // 2048)
    else:
        return 1

def find_communication_bottlenecks(ir: CollectiveIR) -> List[Tuple]:
    operations = ir.get_operation_sequence()
    device_communication = {}
    link_communication = {}
    
    for op in operations:
        if op.op_type == PrimitiveOpType.COPY and op.src_chunk and op.tgt_chunk:
            src_device = op.src_chunk.cur_device_id
            tgt_device = op.tgt_chunk.cur_device_id
            
            if src_device not in device_communication:
                device_communication[src_device] = 0
            device_communication[src_device] += op.src_chunk.data_size
            
            if tgt_device not in device_communication:
                device_communication[tgt_device] = 0
            device_communication[tgt_device] += op.src_chunk.data_size
            
            link_key = (min(src_device, tgt_device), max(src_device, tgt_device))
            if link_key not in link_communication:
                link_communication[link_key] = 0
            link_communication[link_key] += op.src_chunk.data_size
    
    bottlenecks = []
    
    if device_communication:
        max_device = max(device_communication, key=device_communication.get)
        bottlenecks.append(("device", max_device, device_communication[max_device]))
    
    if link_communication:
        max_link = max(link_communication, key=link_communication.get)
        bottlenecks.append(("link", max_link, link_communication[max_link]))
    
    return bottlenecks
