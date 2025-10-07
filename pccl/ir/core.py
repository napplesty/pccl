from dataclasses import dataclass
from typing import Set, List, Dict, Optional, Any
from .graph import DirectedGraph
from .types import PrimitiveOpType, DeviceType

@dataclass
class Chunk:
    reduced_ranks: Set[int]
    cur_device_id: int
    data_size: int
    offset: int
    
    def copy(self):
        return Chunk(
            reduced_ranks=self.reduced_ranks.copy(),
            cur_device_id=self.cur_device_id,
            data_size=self.data_size,
            offset=self.offset
        )

@dataclass
class Device:
    device_id: int
    device_type: DeviceType
    memory_bandwidth: float
    compute_capability: float

@dataclass
class Link:
    src_device: int
    dst_device: int
    bandwidth: float
    latency: float

@dataclass
class PrimitiveOp:
    op_type: PrimitiveOpType
    op_id: int
    src_chunk: Optional[Chunk]
    tgt_chunk: Optional[Chunk]
    src_device: Optional[int]
    tgt_device: Optional[int]
    dependencies: List[int]
    metadata: Dict[str, Any]

class CollectiveIR:
    def __init__(self, name: str):
        self.name = name
        self.precondition: List[Chunk] = []
        self.postcondition: List[Chunk] = []
        self.devices: Dict[int, Device] = {}
        self.links: List[Link] = []
        self.ops: Dict[int, PrimitiveOp] = {}
        self.op_dag = DirectedGraph()
        self.device_topology = DirectedGraph()
        self._next_op_id = 0
    
    def add_device(self, device: Device):
        self.devices[device.device_id] = device
        self.device_topology.add_node(device.device_id)
    
    def add_link(self, link: Link):
        self.links.append(link)
        self.device_topology.add_edge(link.src_device, link.dst_device)
        self.device_topology.add_edge(link.dst_device, link.src_device)
    
    def create_operation(self, op_type: PrimitiveOpType, src_chunk: Optional[Chunk] = None, 
                        tgt_chunk: Optional[Chunk] = None, src_device: Optional[int] = None,
                        tgt_device: Optional[int] = None, dependencies: List[int] = None,
                        metadata: Dict[str, Any] = None) -> int:
        op_id = self._next_op_id
        self._next_op_id += 1
        
        if dependencies is None:
            dependencies = []
        
        if metadata is None:
            metadata = {}
        
        op = PrimitiveOp(
            op_type=op_type,
            op_id=op_id,
            src_chunk=src_chunk,
            tgt_chunk=tgt_chunk,
            src_device=src_device,
            tgt_device=tgt_device,
            dependencies=dependencies,
            metadata=metadata
        )
        
        self.ops[op_id] = op
        self.op_dag.add_node(op_id)
        
        for dep_id in dependencies:
            self.op_dag.add_edge(dep_id, op_id)
        
        return op_id
    
    def set_precondition(self, chunks: List[Chunk]):
        self.precondition = chunks
    
    def set_postcondition(self, chunks: List[Chunk]):
        self.postcondition = chunks
    
    def validate_topology(self) -> bool:
        components = self.device_topology.get_connected_components()
        return len(components) == 1
    
    def validate_dag(self) -> bool:
        return self.op_dag.is_acyclic()
    
    def get_device_bandwidth_matrix(self) -> Dict[tuple, float]:
        bandwidth_matrix = {}
        for link in self.links:
            bandwidth_matrix[(link.src_device, link.dst_device)] = link.bandwidth
            bandwidth_matrix[(link.dst_device, link.src_device)] = link.bandwidth
        return bandwidth_matrix
    
    def get_operation_sequence(self) -> List[PrimitiveOp]:
        try:
            order = self.op_dag.topological_sort()
            return [self.ops[node_id] for node_id in order]
        except ValueError:
            return []
