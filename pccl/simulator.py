import time
from typing import Dict, List, Any
from .ir.core import CollectiveIR, PrimitiveOpType, Chunk
from .utils.analysis import analyze_communication_pattern

class OperationEvent:
    def __init__(self, op_id: int, op_type: PrimitiveOpType, start_time: float, 
                 end_time: float, device_id: int, data_size: int):
        self.op_id = op_id
        self.op_type = op_type
        self.start_time = start_time
        self.end_time = end_time
        self.device_id = device_id
        self.data_size = data_size

class DeviceState:
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.busy_until = 0.0
        self.operation_queue = []

class CollectiveSimulator:
    def __init__(self, ir: CollectiveIR, bandwidth_factor: float = 1.0, 
                 latency_factor: float = 1.0):
        self.ir = ir
        self.bandwidth_factor = bandwidth_factor
        self.latency_factor = latency_factor
        self.device_states = {}
        self.operation_events = []
        self.bandwidth_matrix = ir.get_device_bandwidth_matrix()
        
        for device_id in ir.devices:
            self.device_states[device_id] = DeviceState(device_id)
    
    def simulate(self) -> Dict[str, Any]:
        operations = self.ir.get_operation_sequence()
        current_time = 0.0
        
        for op in operations:
            start_time = self._calculate_operation_start_time(op, current_time)
            duration = self._estimate_operation_duration(op)
            end_time = start_time + duration
            
            event = OperationEvent(
                op_id=op.op_id,
                op_type=op.op_type,
                start_time=start_time,
                end_time=end_time,
                device_id=self._get_operation_device(op),
                data_size=self._get_operation_data_size(op)
            )
            
            self.operation_events.append(event)
            self._update_device_state(op, end_time)
            current_time = max(current_time, end_time)
        
        return self._generate_timeline(current_time)
    
    def _calculate_operation_start_time(self, op, current_time: float) -> float:
        start_time = current_time
        
        for dep_id in op.dependencies:
            dep_event = next((e for e in self.operation_events if e.op_id == dep_id), None)
            if dep_event:
                start_time = max(start_time, dep_event.end_time)
        
        if op.op_type in [PrimitiveOpType.COPY, PrimitiveOpType.REDUCE]:
            if op.tgt_chunk:
                device_id = op.tgt_chunk.cur_device_id
                start_time = max(start_time, self.device_states[device_id].busy_until)
        
        return start_time
    
    def _estimate_operation_duration(self, op) -> float:
        base_latency = 1.0 * self.latency_factor
        
        if op.op_type == PrimitiveOpType.COPY and op.src_chunk and op.tgt_chunk:
            src_device = op.src_chunk.cur_device_id
            tgt_device = op.tgt_chunk.cur_device_id
            data_size = op.src_chunk.data_size
            
            if src_device == tgt_device:
                return base_latency + (data_size / (1024 * 1024)) * 0.001
            
            link_key = (src_device, tgt_device)
            if link_key in self.bandwidth_matrix:
                bandwidth = self.bandwidth_matrix[link_key] * self.bandwidth_factor
                transfer_time = data_size / (bandwidth * 1024 * 1024)
                return base_latency + transfer_time
            else:
                return base_latency + (data_size / (1024 * 1024)) * 0.1
        
        elif op.op_type == PrimitiveOpType.REDUCE and op.tgt_chunk:
            compute_time = op.tgt_chunk.data_size * 0.000001
            return base_latency + compute_time
        
        else:
            return base_latency
    
    def _get_operation_device(self, op) -> int:
        if op.op_type in [PrimitiveOpType.COPY, PrimitiveOpType.REDUCE]:
            if op.tgt_chunk:
                return op.tgt_chunk.cur_device_id
        return -1
    
    def _get_operation_data_size(self, op) -> int:
        if op.op_type in [PrimitiveOpType.COPY, PrimitiveOpType.REDUCE]:
            if op.src_chunk:
                return op.src_chunk.data_size
        return 0
    
    def _update_device_state(self, op, end_time: float):
        device_id = self._get_operation_device(op)
        if device_id != -1:
            self.device_states[device_id].busy_until = end_time
    
    def _generate_timeline(self, total_time: float) -> Dict[str, Any]:
        timeline = {
            "total_time": total_time,
            "operations": [],
            "device_utilization": {},
            "throughput": 0.0
        }
        
        total_data = 0
        for op in self.ir.get_operation_sequence():
            if op.op_type == PrimitiveOpType.COPY and op.src_chunk:
                total_data += op.src_chunk.data_size
        
        if total_time > 0:
            timeline["throughput"] = total_data / total_time
        
        for device_id, state in self.device_states.items():
            utilization = 0.0
            device_events = [e for e in self.operation_events if e.device_id == device_id]
            
            for event in device_events:
                utilization += (event.end_time - event.start_time)
            
            if total_time > 0:
                utilization /= total_time
            
            timeline["device_utilization"][device_id] = min(utilization, 1.0)
        
        for event in self.operation_events:
            timeline["operations"].append({
                "op_id": event.op_id,
                "op_type": event.op_type.name,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "device_id": event.device_id,
                "data_size": event.data_size
            })
        
        return timeline
    
    def print_timeline_report(self):
        timeline = self.simulate()
        
        print(f"=== Collective Communication Simulation Report ===")
        print(f"Total Time: {timeline['total_time']:.6f} seconds")
        print(f"Throughput: {timeline['throughput'] / (1024*1024):.2f} MB/s")
        print(f"Number of Operations: {len(timeline['operations'])}")
        print()
        
        print("Device Utilization:")
        for device_id, utilization in timeline["device_utilization"].items():
            print(f"  Device {device_id}: {utilization*100:.1f}%")
        print()
        
        print("Operation Timeline:")
        for op_info in timeline["operations"]:
            print(f"  Op {op_info['op_id']:3d} ({op_info['op_type']:8s}): "
                  f"Device {op_info['device_id']} | "
                  f"Time [{op_info['start_time']:.6f} - {op_info['end_time']:.6f}] | "
                  f"Size: {op_info['data_size']} bytes")
