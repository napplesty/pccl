from typing import Dict, List, Set, Tuple, Optional
from .primitive_ir import (
    PrimitiveGraph, PrimitiveType, DataType, ComputeType, ExecutorType,
    BufferConfig, ExecutorConfig, PrimitiveConfig
)
from ..collective_ir.core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory, RemoteMemory, Device
from ..collective_ir.core.enums import PrimitiveOpType, CollectiveOpType

class CollectiveToPrimitiveTransformer:
    def __init__(self):
        self.buffer_counter = 0
        self.operator_counter = 0
        self.buffer_map = {}
        self.device_executor_map = {}
    
    def transform(self, collective_ir: CollectiveIR, rank: int = 0) -> PrimitiveGraph:
        graph = PrimitiveGraph(rank=rank)
        
        self._initialize_executors(graph, collective_ir)
        self._create_buffers(graph, collective_ir, rank)
        self._convert_tasks(graph, collective_ir, rank)
        self._add_dependencies(graph)
        
        return graph
    
    def _initialize_executors(self, graph: PrimitiveGraph, collective_ir: CollectiveIR):
        device_types = set()
        for device in collective_ir.cluster.devices_by_id.values():
            device_type = self._map_device_type(device.type)
            device_types.add(device_type)
        
        for executor_type in device_types:
            graph.executors.append(ExecutorConfig(
                executor_type=executor_type,
                num_total_executors=len(collective_ir.cluster.devices_by_id)
            ))
    
    def _map_device_type(self, device_type: str) -> ExecutorType:
        device_type_lower = device_type.lower()
        if 'cuda' in device_type_lower or 'gpu' in device_type_lower:
            return ExecutorType.CUDA
        else:
            return ExecutorType.CPU
    
    def _create_buffers(self, graph: PrimitiveGraph, collective_ir: CollectiveIR, rank: int):
        total_size_bytes = int(collective_ir.data_size_gb * 1024 * 1024 * 1024)
        
        input_buffer = graph.add_buffer(
            idx=0,
            dtype=DataType.F32,
            size=total_size_bytes,
            executor_type=ExecutorType.CUDA
        )
        self.buffer_map[('input', rank)] = 0
        
        output_buffer = graph.add_buffer(
            idx=1,
            dtype=DataType.F32,
            size=total_size_bytes,
            executor_type=ExecutorType.CUDA
        )
        self.buffer_map[('output', rank)] = 1
        
        for device_id, device in collective_ir.cluster.devices_by_id.items():
            if device_id != rank:
                comm_buffer = graph.add_buffer(
                    idx=2 + device_id,
                    dtype=DataType.F32,
                    size=total_size_bytes // 4,
                    executor_type=ExecutorType.CUDA
                )
                self.buffer_map[('comm', device_id)] = 2 + device_id
    
    def _convert_tasks(self, graph: PrimitiveGraph, collective_ir: CollectiveIR, rank: int):
        device_ids = list(collective_ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        for task_id, task in collective_ir.task_map.tasks.items():
            for primitive in task.primitives:
                if primitive.initiator.device_id == rank:
                    self._convert_primitive(graph, primitive, task_id, rank, n_devices)
    
    def _convert_primitive(self, graph: PrimitiveGraph, primitive: CommunicationPrimitive, 
                          task_id: int, rank: int, n_devices: int):
        if primitive.op_type == PrimitiveOpType.COPY:
            self._convert_copy_primitive(graph, primitive, rank)
        elif primitive.op_type == PrimitiveOpType.REDUCE:
            self._convert_reduce_primitive(graph, primitive, rank)
        elif primitive.op_type == PrimitiveOpType.WRITE:
            self._convert_write_primitive(graph, primitive, rank)
    
    def _convert_copy_primitive(self, graph: PrimitiveGraph, primitive: CommunicationPrimitive, rank: int):
        if len(primitive.memory_regions) >= 2:
            src_region = primitive.memory_regions[0]
            dst_region = primitive.memory_regions[1]
            
            target_rank = dst_region.device.device_id
            
            src_buffer_idx = self._get_buffer_index(src_region, rank)
            dst_buffer_idx = self._get_buffer_index(dst_region, target_rank)
            
            if src_buffer_idx is not None and dst_buffer_idx is not None:
                op_config = PrimitiveConfig(
                    type=PrimitiveType.COPY,
                    dtype=DataType.F32,
                    target_rank=target_rank,
                    src_buffer_idx=src_buffer_idx,
                    dst_buffer_idx=dst_buffer_idx,
                    compute_op=ComputeType.SUM,
                    executor_type=ExecutorType.CUDA,
                    num_executors=1,
                    data_size=src_region.size,
                    signal_value=0,
                    num_dependencies=0
                )
                
                graph.add_operator(op_config)
                self.operator_counter += 1
    
    def _convert_reduce_primitive(self, graph: PrimitiveGraph, primitive: CommunicationPrimitive, rank: int):
        if len(primitive.memory_regions) >= 2:
            src_region = primitive.memory_regions[0]
            dst_region = primitive.memory_regions[1]
            
            src_buffer_idx = self._get_buffer_index(src_region, rank)
            dst_buffer_idx = self._get_buffer_index(dst_region, rank)
            
            if src_buffer_idx is not None and dst_buffer_idx is not None:
                op_config = PrimitiveConfig(
                    type=PrimitiveType.COMPUTE,
                    dtype=DataType.F32,
                    target_rank=rank,
                    src_buffer_idx=src_buffer_idx,
                    dst_buffer_idx=dst_buffer_idx,
                    compute_op=ComputeType.SUM,
                    executor_type=ExecutorType.CUDA,
                    num_executors=1,
                    data_size=src_region.size,
                    signal_value=0,
                    num_dependencies=0
                )
                
                graph.add_operator(op_config)
                self.operator_counter += 1
    
    def _convert_write_primitive(self, graph: PrimitiveGraph, primitive: CommunicationPrimitive, rank: int):
        if primitive.memory_regions:
            memory_region = primitive.memory_regions[0]
            buffer_idx = self._get_buffer_index(memory_region, rank)
            
            if buffer_idx is not None:
                op_config = PrimitiveConfig(
                    type=PrimitiveType.WRITE,
                    dtype=DataType.F32,
                    target_rank=rank,
                    src_buffer_idx=buffer_idx,
                    dst_buffer_idx=buffer_idx,
                    compute_op=ComputeType.SUM,
                    executor_type=ExecutorType.CUDA,
                    num_executors=1,
                    data_size=memory_region.size,
                    signal_value=0,
                    num_dependencies=0
                )
                
                graph.add_operator(op_config)
                self.operator_counter += 1
    
    def _get_buffer_index(self, memory_region, rank: int) -> Optional[int]:
        if isinstance(memory_region, LocalMemory):
            if memory_region.address == 0:
                return self.buffer_map.get(('input', rank))
            else:
                return self.buffer_map.get(('comm', memory_region.device.device_id))
        elif isinstance(memory_region, RemoteMemory):
            return self.buffer_map.get(('comm', memory_region.device.device_id))
        return None
    
    def _add_dependencies(self, graph: PrimitiveGraph):
        n_operators = len(graph.operators)
        for i in range(n_operators - 1):
            graph.add_dependency(i, i + 1)

def convert_collective_to_primitive(collective_ir: CollectiveIR, rank: int = 0) -> PrimitiveGraph:
    transformer = CollectiveToPrimitiveTransformer()
    return transformer.transform(collective_ir, rank)
