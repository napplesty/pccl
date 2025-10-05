import json
from ..core.ir import *
from ..core.enums import *

class IRJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (CollectiveOpType, PrimitiveOpType, TaskStatus, MemoryType, AccessPermission, ChunkState)):
            return obj.name
        elif isinstance(obj, Device):
            return {
                "type": "Device",
                "id": obj.device_id,
                "device_type": obj.type,
                "props": obj.properties,
                "memory_capacity_gb": obj.memory_capacity_gb,
                "bandwidth_gbs": obj.bandwidth_gbs
            }
        elif isinstance(obj, Host):
            return {
                "type": "Host",
                "id": obj.host_id,
                "devices": [d.device_id for d in obj.devices],
                "switches": [s.switch_id for s in obj.connected_switches]
            }
        elif isinstance(obj, Switch):
            return {
                "type": "Switch",
                "id": obj.switch_id,
                "bandwidth_gbs": obj.bandwidth_gbs,
                "hosts": [h.host_id for h in obj.connected_hosts],
                "switches": [s.switch_id for s in obj.connected_switches]
            }
        elif isinstance(obj, ClusterMesh):
            return {
                "type": "ClusterMesh",
                "devices": list(obj.devices_by_id.values()),
                "hosts": obj.hosts,
                "switches": obj.switches,
                "network_topology": obj.network_topology
            }
        elif isinstance(obj, NetworkTopology):
            return {
                "type": "NetworkTopology",
                "bandwidth_matrix": {f"{k[0]}-{k[1]}": v for k, v in obj.bandwidth_matrix.items()},
                "latency_matrix": {f"{k[0]}-{k[1]}": v for k, v in obj.latency_matrix.items()}
            }
        elif isinstance(obj, LocalMemory):
            return {
                "type": "LocalMemory",
                "device_id": obj.device.device_id,
                "address": obj.address,
                "size": obj.size,
                "memory_type": obj.memory_type,
                "access": obj.access
            }
        elif isinstance(obj, RemoteMemory):
            return {
                "type": "RemoteMemory",
                "device_id": obj.device.device_id,
                "address": obj.address,
                "size": obj.size,
                "memory_type": obj.memory_type,
                "access": obj.access
            }
        elif isinstance(obj, CommunicationPrimitive):
            return {
                "type": "CommunicationPrimitive",
                "initiator_id": obj.initiator.device_id,
                "op_type": obj.op_type,
                "memory_regions": obj.memory_regions,
                "estimated_duration_ms": obj.estimated_duration_ms,
                "chunk_updates": {f"{k[0]}-{k[1]}": list(v) for k, v in obj.chunk_updates.items()}
            }
        elif isinstance(obj, Task):
            return {
                "type": "Task",
                "id": obj.task_id,
                "primitives": obj.primitives,
                "deps": [t.task_id for t in obj.dependencies],
                "status": obj.status,
                "estimated_start_time": obj.estimated_start_time,
                "estimated_end_time": obj.estimated_end_time,
                "chunk_updates": {f"{k[0]}-{k[1]}": list(v) for k, v in obj.chunk_updates.items()}
            }
        elif isinstance(obj, TaskMap):
            return {
                "type": "TaskMap",
                "op_type": obj.op_type,
                "data_size": obj.data_size_gb,
                "tasks": list(obj.tasks.values())
            }
        elif isinstance(obj, Chunk):
            return {
                "type": "Chunk",
                "chunk_id": obj.chunk_id,
                "device_id": obj.device_id,
                "state": obj.state,
                "data_size": obj.data_size,
                "offset": obj.offset,
                "reduced_ranks": list(obj.reduced_ranks),
                "metadata": obj.metadata
            }
        elif isinstance(obj, ReducedChunk):
            return {
                "type": "ReducedChunk",
                "base_chunk": obj.base_chunk,
                "reduced_src_ranks": list(obj.reduced_src_ranks),
                "chunk_offset_index": obj.chunk_offset_index,
                "size": obj.size
            }
        elif isinstance(obj, Precondition):
            return {
                "type": "Precondition",
                "chunk_states": {f"{k[0]}-{k[1]}": v.name for k, v in obj.chunk_states.items()},
                "required_devices": list(obj.required_devices),
                "required_chunks": list(obj.required_chunks),
                "reduced_rank_requirements": {f"{k[0]}-{k[1]}": list(v) for k, v in obj.reduced_rank_requirements.items()}
            }
        elif isinstance(obj, Postcondition):
            return {
                "type": "Postcondition",
                "chunk_states": {f"{k[0]}-{k[1]}": v.name for k, v in obj.chunk_states.items()},
                "produced_devices": list(obj.produced_devices),
                "produced_chunks": list(obj.produced_chunks),
                "reduced_rank_updates": {f"{k[0]}-{k[1]}": list(v) for k, v in obj.reduced_rank_updates.items()}
            }
        elif isinstance(obj, CollectiveSpec):
            return {
                "type": "CollectiveSpec",
                "op_type": obj.op_type.name,
                "preconditions": obj.preconditions,
                "postconditions": obj.postconditions,
                "data_size_gb": obj.data_size_gb,
                "involved_devices": list(obj.involved_devices)
            }
        elif isinstance(obj, CollectiveIR):
            return {
                "type": "CollectiveIR",
                "cluster": obj.cluster,
                "collective_op": obj.collective_op,
                "data_size": obj.data_size_gb,
                "task_map": obj.task_map,
                "collective_spec": obj.collective_spec,
                "metadata": obj.metadata
            }
        return super().default(obj)

def serialize(ir: CollectiveIR) -> str:
    return json.dumps(ir, cls=IRJSONEncoder, indent=2)
