from .base import Pass
from ..ir.core import CollectiveIR, PrimitiveOpType
from typing import Dict, List

class PerformanceModelingPass(Pass):
    def __init__(self):
        super().__init__("performance_modeling")
    
    def run(self, ir: CollectiveIR) -> bool:
        total_cost = self._estimate_total_cost(ir)
        bottleneck_analysis = self._analyze_bottlenecks(ir)
        
        ir.metadata = {
            "estimated_cost": total_cost,
            "bottleneck_analysis": bottleneck_analysis,
            "performance_metrics": self._calculate_performance_metrics(ir)
        }
        
        return True
    
    def _estimate_total_cost(self, ir: CollectiveIR) -> float:
        operations = ir.get_operation_sequence()
        bandwidth_matrix = ir.get_device_bandwidth_matrix()
        total_cost = 0.0
        
        for op in operations:
            if op.op_type == PrimitiveOpType.COPY and op.src_chunk and op.tgt_chunk:
                src_device = op.src_chunk.cur_device_id
                tgt_device = op.tgt_chunk.cur_device_id
                data_size = op.src_chunk.data_size
                
                if (src_device, tgt_device) in bandwidth_matrix:
                    bandwidth = bandwidth_matrix[(src_device, tgt_device)]
                    transfer_time = data_size / bandwidth if bandwidth > 0 else float('inf')
                    total_cost += transfer_time
            
            elif op.op_type == PrimitiveOpType.REDUCE:
                total_cost += op.tgt_chunk.data_size * 0.000001
        
        return total_cost
    
    def _analyze_bottlenecks(self, ir: CollectiveIR) -> Dict:
        operations = ir.get_operation_sequence()
        device_utilization = {device_id: 0.0 for device_id in ir.devices}
        link_utilization = {}
        
        for op in operations:
            if op.op_type == PrimitiveOpType.COPY and op.src_chunk and op.tgt_chunk:
                src_device = op.src_chunk.cur_device_id
                tgt_device = op.tgt_chunk.cur_device_id
                data_size = op.src_chunk.data_size
                
                link_key = (min(src_device, tgt_device), max(src_device, tgt_device))
                if link_key not in link_utilization:
                    link_utilization[link_key] = 0.0
                link_utilization[link_key] += data_size
                
                device_utilization[src_device] += data_size
                device_utilization[tgt_device] += data_size
        
        max_device = max(device_utilization, key=device_utilization.get)
        max_link = max(link_utilization, key=link_utilization.get) if link_utilization else None
        
        return {
            "bottleneck_device": max_device,
            "bottleneck_link": max_link,
            "device_utilization": device_utilization,
            "link_utilization": link_utilization
        }
    
    def _calculate_performance_metrics(self, ir: CollectiveIR) -> Dict:
        operations = ir.get_operation_sequence()
        total_operations = len(operations)
        copy_operations = sum(1 for op in operations if op.op_type == PrimitiveOpType.COPY)
        reduce_operations = sum(1 for op in operations if op.op_type == PrimitiveOpType.REDUCE)
        
        total_data_moved = 0
        for op in operations:
            if op.op_type == PrimitiveOpType.COPY and op.src_chunk:
                total_data_moved += op.src_chunk.data_size
        
        parallelism_factor = self._calculate_parallelism_factor(ir)
        
        return {
            "total_operations": total_operations,
            "copy_operations": copy_operations,
            "reduce_operations": reduce_operations,
            "total_data_moved": total_data_moved,
            "parallelism_factor": parallelism_factor
        }
    
    def _calculate_parallelism_factor(self, ir: CollectiveIR) -> float:
        operations = ir.get_operation_sequence()
        if not operations:
            return 0.0
        
        critical_path_length = len(ir.op_dag.topological_sort())
        total_operations = len(operations)
        
        return total_operations / critical_path_length if critical_path_length > 0 else 0.0
