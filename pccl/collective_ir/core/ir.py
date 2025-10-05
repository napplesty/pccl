from dataclasses import dataclass, field
from typing import List, Dict, Set, Union, Optional, Tuple, Any
from .enums import *
from .exceptions import *

@dataclass
class Chunk:
    chunk_id: int
    device_id: int
    state: ChunkState
    data_size: int
    offset: int
    reduced_ranks: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_fully_reduced(self, all_ranks: Set[int]) -> bool:
        return self.state == ChunkState.REDUCED and self.reduced_ranks == all_ranks

    def add_reduced_rank(self, rank: int):
        self.reduced_ranks.add(rank)
        if len(self.reduced_ranks) > 1:
            self.state = ChunkState.PARTIAL

    def mark_fully_reduced(self, all_ranks: Set[int]):
        self.reduced_ranks = all_ranks.copy()
        self.state = ChunkState.REDUCED

@dataclass
class ReducedChunk:
    base_chunk: Chunk
    reduced_src_ranks: Set[int]
    chunk_offset_index: int
    size: int
    
    @property
    def device_id(self) -> int:
        return self.base_chunk.device_id
    
    @property
    def state(self) -> ChunkState:
        return self.base_chunk.state

@dataclass
class Precondition:
    chunk_states: Dict[Tuple[int, int], ChunkState]
    required_devices: Set[int]
    required_chunks: Set[int]
    reduced_rank_requirements: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)
    
    def is_satisfied(self, current_chunks: Dict[int, Chunk]) -> bool:
        for (device_id, chunk_id), required_state in self.chunk_states.items():
            chunk_key = (device_id, chunk_id)
            if chunk_key not in current_chunks:
                return False
            
            chunk = current_chunks[chunk_key]
            if chunk.state != required_state:
                return False
            
            if chunk_key in self.reduced_rank_requirements:
                required_ranks = self.reduced_rank_requirements[chunk_key]
                if not required_ranks.issubset(chunk.reduced_ranks):
                    return False
        
        return True

@dataclass
class Postcondition:
    chunk_states: Dict[Tuple[int, int], ChunkState]
    produced_devices: Set[int]
    produced_chunks: Set[int]
    reduced_rank_updates: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)

@dataclass
class CollectiveSpec:
    op_type: CollectiveOpType
    preconditions: List[Precondition]
    postconditions: List[Postcondition]
    data_size_gb: float
    involved_devices: Set[int]
    
    def validate_preconditions(self, current_chunks: Dict[int, Chunk]) -> bool:
        return any(precond.is_satisfied(current_chunks) for precond in self.preconditions)

@dataclass
class NetworkTopology:
    bandwidth_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    latency_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    def get_bandwidth(self, device1_id: int, device2_id: int) -> float:
        return self.bandwidth_matrix.get((device1_id, device2_id), 1.0)
    
    def get_latency(self, device1_id: int, device2_id: int) -> float:
        return self.latency_matrix.get((device1_id, device2_id), 1.0)

@dataclass
class Device:
    device_id: int
    type: str
    properties: Dict = field(default_factory=dict)
    memory_capacity_gb: float = 0.0
    bandwidth_gbs: float = 0.0

    def __hash__(self):
        return hash(self.device_id)
    
    def __eq__(self, other):
        if not isinstance(other, Device):
            return False
        return self.device_id == other.device_id

@dataclass
class Switch:
    switch_id: int
    bandwidth_gbs: float = 0.0
    connected_hosts: List["Host"] = field(default_factory=list)
    connected_switches: List["Switch"] = field(default_factory=list)

@dataclass
class Host:
    host_id: int
    devices: List[Device] = field(default_factory=list)
    connected_switches: List[Switch] = field(default_factory=list)

    def get_device(self, device_id: int) -> Optional[Device]:
        return next((d for d in self.devices if d.device_id == device_id), None)

@dataclass
class ClusterMesh:
    hosts: List[Host] = field(default_factory=list)
    switches: List[Switch] = field(default_factory=list)
    network_topology: NetworkTopology = field(default_factory=NetworkTopology)
    devices_by_id: Dict[int, Device] = field(init=False, default_factory=dict)

    def __post_init__(self):
        for host in self.hosts:
            for dev in host.devices:
                if dev.device_id in self.devices_by_id:
                    raise InvalidTopologyError(f"Device {dev.device_id} already exists")
                self.devices_by_id[dev.device_id] = dev

    def get_device(self, device_id: int) -> Device:
        if device_id not in self.devices_by_id:
            raise DeviceNotFoundError(f"Device {device_id} not found in cluster")
        return self.devices_by_id[device_id]

    def validate_connectivity(self) -> bool:
        for switch in self.switches:
            for host in switch.connected_hosts:
                if host not in self.hosts:
                    return False
            for other_switch in switch.connected_switches:
                if other_switch not in self.switches:
                    return False
        return True

@dataclass
class MemoryRegion:
    device: Device
    address: int
    size: int
    memory_type: MemoryType = MemoryType.DRAM
    access: AccessPermission = AccessPermission.READ_WRITE

@dataclass
class LocalMemory(MemoryRegion):
    pass

@dataclass
class RemoteMemory(MemoryRegion):
    pass

@dataclass
class CommunicationPrimitive:
    initiator: Device
    op_type: PrimitiveOpType
    memory_regions: List[Union[LocalMemory, RemoteMemory]]
    estimated_duration_ms: float = 0.0
    chunk_updates: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)

    def __post_init__(self):
        if not any(isinstance(reg, LocalMemory) for reg in self.memory_regions):
            raise ValueError("Primitive must involve at least one local memory region")
        
        if self.memory_regions:
            total_size = sum(region.size for region in self.memory_regions)
            self.estimated_duration_ms = self._estimate_duration(total_size)

    def _estimate_duration(self, total_size: int) -> float:
        base_latency = {
            PrimitiveOpType.WRITE: 0.1,
            PrimitiveOpType.COPY: 0.2,
            PrimitiveOpType.REDUCE: 0.5,
            PrimitiveOpType.NOTIFY: 0.01,
            PrimitiveOpType.GET_NOTIFIED: 0.01,
        }
        return base_latency.get(self.op_type, 1.0)

@dataclass
class Task:
    task_id: int
    primitives: List[CommunicationPrimitive]
    dependencies: List["Task"] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    estimated_start_time: float = 0.0
    estimated_end_time: float = 0.0
    chunk_updates: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)

    def add_dependency(self, task: "Task") -> None:
        if self._creates_cycle(task):
            raise CircularDependencyError(
                f"Adding dependency from task {self.task_id} to {task.task_id} creates a cycle"
            )
        self.dependencies.append(task)

    def _creates_cycle(self, new_dependency: "Task") -> bool:
        visited = set()
        return self._has_path_to(new_dependency, visited)

    def _has_path_to(self, target: "Task", visited: Set[int]) -> bool:
        if self.task_id in visited:
            return False
        visited.add(self.task_id)
        
        if self == target:
            return True
            
        for dep in self.dependencies:
            if dep._has_path_to(target, visited):
                return True
        return False

    @property
    def total_estimated_duration(self) -> float:
        return sum(prim.estimated_duration_ms for prim in self.primitives)

@dataclass
class TaskMap:
    op_type: CollectiveOpType
    data_size_gb: float
    tasks: Dict[int, Task] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists")
        self.tasks[task.task_id] = task

    def get_ready_tasks(self) -> List[Task]:
        ready_tasks = []
        for task in self.tasks.values():
            if all(dep.status == TaskStatus.COMPLETED for dep in task.dependencies):
                ready_tasks.append(task)
        return ready_tasks

    def validate_dependencies(self) -> bool:
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep.task_id not in self.tasks:
                    return False
        return True

@dataclass(frozen=True)
class CollectiveIR:
    cluster: ClusterMesh
    collective_op: CollectiveOpType
    data_size_gb: float
    task_map: TaskMap
    collective_spec: CollectiveSpec
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate_ir()

    def _validate_ir(self) -> None:
        if self.task_map.op_type != self.collective_op:
            raise ValueError(
                f"TaskMap op_type [{self.task_map.op_type.name}] "
                f"mismatches CollectiveIR op [{self.collective_op.name}]"
            )
        
        self._validate_device_references()
        
        if not self.task_map.validate_dependencies():
            raise ValueError("Invalid task dependencies")
        
        if not self.cluster.validate_connectivity():
            raise InvalidTopologyError("Cluster has invalid connectivity")
        
        if self.collective_spec.op_type != self.collective_op:
            raise ValueError("CollectiveSpec op_type mismatches CollectiveIR op")

    def _validate_device_references(self) -> None:
        task_devices: Set[int] = set()
        for task in self.task_map.tasks.values():
            for prim in task.primitives:
                task_devices.add(prim.initiator.device_id)
                for mem in prim.memory_regions:
                    task_devices.add(mem.device.device_id)
        
        cluster_devices = set(self.cluster.devices_by_id.keys())
        missing_devices = task_devices - cluster_devices
        if missing_devices:
            raise DeviceNotFoundError(f"Task references devices not in cluster: {missing_devices}")

    def estimate_completion_time(self) -> float:
        total_time = 0.0
        for task in self.task_map.tasks.values():
            total_time += task.total_estimated_duration
        return total_time

    def get_current_chunk_states(self) -> Dict[Tuple[int, int], Chunk]:
        current_chunks = {}
        for task in self.task_map.tasks.values():
            if task.status == TaskStatus.COMPLETED:
                for prim in task.primitives:
                    for mem in prim.memory_regions:
                        chunk_key = (mem.device.device_id, id(mem))
                        if chunk_key not in current_chunks:
                            current_chunks[chunk_key] = Chunk(
                                chunk_id=id(mem),
                                device_id=mem.device.device_id,
                                state=ChunkState.VALID,
                                data_size=mem.size,
                                offset=mem.address
                            )
                        
                        chunk = current_chunks[chunk_key]
                        if prim.op_type == PrimitiveOpType.REDUCE:
                            for update_key, ranks in prim.chunk_updates.items():
                                if update_key == chunk_key:
                                    for rank in ranks:
                                        chunk.add_reduced_rank(rank)
                        
                        elif prim.op_type == PrimitiveOpType.COPY:
                            chunk.state = ChunkState.VALID
        
        return current_chunks
