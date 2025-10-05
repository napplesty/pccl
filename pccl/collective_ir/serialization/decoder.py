import json
from ..core.ir import *
from ..core.enums import *

def deserialize(json_str: str) -> CollectiveIR:
    def _reconstruct_cluster(cluster_data: dict) -> ClusterMesh:
        devices = {}
        hosts = {}
        switches = {}
        
        for d_data in cluster_data.get("devices", []):
            if d_data["type"] == "Device":
                devices[d_data["id"]] = Device(
                    d_data["id"],
                    d_data["device_type"],
                    d_data.get("props", {}),
                    d_data.get("memory_capacity_gb", 0.0),
                    d_data.get("bandwidth_gbs", 0.0)
                )
        
        for h_data in cluster_data.get("hosts", []):
            if h_data["type"] == "Host":
                host_devices = [devices[did] for did in h_data["devices"] if did in devices]
                hosts[h_data["id"]] = Host(h_data["id"], host_devices, [])
        
        for s_data in cluster_data.get("switches", []):
            if s_data["type"] == "Switch":
                switches[s_data["id"]] = Switch(
                    s_data["id"],
                    s_data.get("bandwidth_gbs", 0.0),
                    [],
                    []
                )
        
        for h_data in cluster_data.get("hosts", []):
            if h_data["type"] == "Host":
                host = hosts[h_data["id"]]
                host.connected_switches = [switches[sid] for sid in h_data["switches"] if sid in switches]
        
        for s_data in cluster_data.get("switches", []):
            if s_data["type"] == "Switch":
                switch = switches[s_data["id"]]
                switch.connected_hosts = [hosts[hid] for hid in s_data["hosts"] if hid in hosts]
                switch.connected_switches = [switches[sid] for sid in s_data["switches"] if sid in switches]
        
        topology_data = cluster_data.get("network_topology", {})
        bandwidth_matrix = {}
        latency_matrix = {}
        
        if topology_data.get("type") == "NetworkTopology":
            for key, value in topology_data.get("bandwidth_matrix", {}).items():
                dev1, dev2 = map(int, key.split("-"))
                bandwidth_matrix[(dev1, dev2)] = value
            for key, value in topology_data.get("latency_matrix", {}).items():
                dev1, dev2 = map(int, key.split("-"))
                latency_matrix[(dev1, dev2)] = value
        
        network_topology = NetworkTopology(bandwidth_matrix, latency_matrix)
        
        return ClusterMesh(
            list(hosts.values()),
            list(switches.values()),
            network_topology
        )
    def _reconstruct_collective_spec(spec_data: dict) -> CollectiveSpec:
        if spec_data["type"] != "CollectiveSpec":
            raise ValueError("Invalid CollectiveSpec entry")
        
        preconditions = []
        for pre_data in spec_data.get("preconditions", []):
            if pre_data["type"] != "Precondition":
                raise ValueError("Invalid Precondition entry")
            
            chunk_states = {}
            for key_str, state_name in pre_data.get("chunk_states", {}).items():
                dev_id, chunk_id = map(int, key_str.split("-"))
                chunk_states[(dev_id, chunk_id)] = ChunkState[state_name]
            
            reduced_rank_requirements = {}
            for key_str, ranks in pre_data.get("reduced_rank_requirements", {}).items():
                dev_id, chunk_id = map(int, key_str.split("-"))
                reduced_rank_requirements[(dev_id, chunk_id)] = set(ranks)
            
            preconditions.append(Precondition(
                chunk_states=chunk_states,
                required_devices=set(pre_data.get("required_devices", [])),
                required_chunks=set(pre_data.get("required_chunks", [])),
                reduced_rank_requirements=reduced_rank_requirements
            ))
        
        postconditions = []
        for post_data in spec_data.get("postconditions", []):
            if post_data["type"] != "Postcondition":
                raise ValueError("Invalid Postcondition entry")
            
            chunk_states = {}
            for key_str, state_name in post_data.get("chunk_states", {}).items():
                dev_id, chunk_id = map(int, key_str.split("-"))
                chunk_states[(dev_id, chunk_id)] = ChunkState[state_name]
            
            reduced_rank_updates = {}
            for key_str, ranks in post_data.get("reduced_rank_updates", {}).items():
                dev_id, chunk_id = map(int, key_str.split("-"))
                reduced_rank_updates[(dev_id, chunk_id)] = set(ranks)
            
            postconditions.append(Postcondition(
                chunk_states=chunk_states,
                produced_devices=set(post_data.get("produced_devices", [])),
                produced_chunks=set(post_data.get("produced_chunks", [])),
                reduced_rank_updates=reduced_rank_updates
            ))
        
        return CollectiveSpec(
            op_type=CollectiveOpType[spec_data["op_type"]],
            preconditions=preconditions,
            postconditions=postconditions,
            data_size_gb=spec_data["data_size_gb"],
            involved_devices=set(spec_data.get("involved_devices", []))
        )
    def _reconstruct_task_map(task_map_data: dict, cluster: ClusterMesh) -> TaskMap:
        if task_map_data["type"] != "TaskMap":
            raise ValueError("Invalid TaskMap entry")
        tasks = {}
        
        for t_data in task_map_data["tasks"]:
            if t_data["type"] != "Task":
                raise ValueError("Invalid Task entry")
            
            primitives = []
            for p_data in t_data["primitives"]:
                if p_data["type"] != "CommunicationPrimitive":
                    raise ValueError("Invalid Primitive entry")
                
                initiator = cluster.get_device(p_data["initiator_id"])
                op_type = PrimitiveOpType[p_data["op_type"]]
                mem_regions = []
                
                for mem_data in p_data["memory_regions"]:
                    dev = cluster.get_device(mem_data["device_id"])
                    memory_type = MemoryType[mem_data.get("memory_type", "DRAM")]
                    access = AccessPermission[mem_data.get("access", "READ_WRITE")]
                    
                    if mem_data["type"] == "LocalMemory":
                        mem_regions.append(LocalMemory(
                            dev, mem_data["address"], mem_data["size"], memory_type, access
                        ))
                    elif mem_data["type"] == "RemoteMemory":
                        mem_regions.append(RemoteMemory(
                            dev, mem_data["address"], mem_data["size"], memory_type, access
                        ))
                
                chunk_updates = {}
                for key_str, ranks in p_data.get("chunk_updates", {}).items():
                    dev_id, chunk_id = map(int, key_str.split("-"))
                    chunk_updates[(dev_id, chunk_id)] = set(ranks)
                
                primitive = CommunicationPrimitive(initiator, op_type, mem_regions)
                primitive.chunk_updates = chunk_updates
                
                if "estimated_duration_ms" in p_data:
                    primitive.estimated_duration_ms = p_data["estimated_duration_ms"]
                
                primitives.append(primitive)
            
            chunk_updates = {}
            for key_str, ranks in t_data.get("chunk_updates", {}).items():
                dev_id, chunk_id = map(int, key_str.split("-"))
                chunk_updates[(dev_id, chunk_id)] = set(ranks)
            
            task = Task(
                t_data["id"],
                primitives,
                [],
                TaskStatus[t_data["status"]],
                t_data.get("estimated_start_time", 0.0),
                t_data.get("estimated_end_time", 0.0),
                chunk_updates
            )
            tasks[t_data["id"]] = task
        
        for t_data in task_map_data["tasks"]:
            if t_data["type"] == "Task":
                task = tasks[t_data["id"]]
                for dep_id in t_data.get("deps", []):
                    if dep_id in tasks:
                        task.dependencies.append(tasks[dep_id])
        
        return TaskMap(
            CollectiveOpType[task_map_data["op_type"]],
            task_map_data["data_size"],
            tasks
        )
    data = json.loads(json_str)
    if data.get("type") != "CollectiveIR":
        raise ValueError("Invalid CollectiveIR format")
    cluster = _reconstruct_cluster(data["cluster"])
    collective_spec = _reconstruct_collective_spec(data["collective_spec"])
    task_map = _reconstruct_task_map(data["task_map"], cluster)
    return CollectiveIR(
        cluster=cluster,
        collective_op=CollectiveOpType[data["collective_op"]],
        data_size_gb=data["data_size"],
        task_map=task_map,
        collective_spec=collective_spec,
        metadata=data.get("metadata", {})
    )
