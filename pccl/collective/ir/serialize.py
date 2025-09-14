import json
from .mesh import ClusterMesh, Device, Host, Switch
from .memory import LocalMemory, RemoteMemory
from .collective import CollectiveOpType
from .primitive import CommunicationPrimitive, PrimitiveOpType
from .task import Task, TaskMap, TaskStatus
from .core import CollectiveIR


class IRJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Device):
            return {"type": "Device", "id": obj.device_id, "device_type": obj.type, "props": obj.properties}
        elif isinstance(obj, Host):
            return {"type": "Host", "id": obj.host_id, "devices": [d.device_id for d in obj.devices], "switches": [s.switch_id for s in obj.connected_switches]}
        elif isinstance(obj, Switch):
            return {"type": "Switch", "id": obj.switch_id, "hosts": [h.host_id for h in obj.connected_hosts], "switches": [s.switch_id for s in obj.connected_switches]}
        elif isinstance(obj, ClusterMesh):
            return {"type": "ClusterMesh", "devices": list(obj.devices_by_id.values()), "hosts": obj.hosts, "switches": obj.switches}
        elif isinstance(obj, LocalMemory):
            return {"type": "LocalMemory", "device_id": obj.device.device_id, "address": obj.address, "size": obj.size}
        elif isinstance(obj, RemoteMemory):
            return {"type": "RemoteMemory", "device_id": obj.device.device_id, "address": obj.address, "size": obj.size}
        elif isinstance(obj, CommunicationPrimitive):
            return {"type": "CommunicationPrimitive", "initiator_id": obj.initiator.device_id, "op_type": obj.op_type.name, "memory_regions": obj.memory_regions}
        elif isinstance(obj, Task):
            return {"type": "Task", "id": obj.task_id, "primitives": obj.primitives, "deps": [t.task_id for t in obj.dependencies], "status": obj.status.name}
        elif isinstance(obj, TaskMap):
            return {"type": "TaskMap", "op_type": obj.op_type.name, "data_size": obj.data_size_gb, "tasks": list(obj.tasks.values())}
        elif isinstance(obj, CollectiveIR):
            return {"type": "CollectiveIR", "cluster": obj.cluster, "collective_op": obj.collective_op.name, "data_size": obj.data_size_gb, "task_map": obj.task_map}
        return super().default(obj)


def serialize(ir: CollectiveIR) -> str:
    return json.dumps(ir, cls=IRJSONEncoder, indent=2)


def deserialize(json_str: str) -> CollectiveIR:
    data = json.loads(json_str)
    if data.get("type") != "CollectiveIR":
        raise ValueError("Invalid CollectiveIR format")

    cluster_data = data["cluster"]
    devices = {d["id"]: Device(d["id"], d["device_type"], d["props"]) for d in cluster_data["devices"] if d["type"] == "Device"}
    hosts = {h["id"]: Host(h["id"], [devices[did] for did in h["devices"]], []) for h in cluster_data["hosts"] if h["type"] == "Host"}
    switches = {s["id"]: Switch(s["id"], [hosts[hid] for hid in s["hosts"]], []) for s in cluster_data["switches"] if s["type"] == "Switch"}

    for h_data in cluster_data["hosts"]:
        if h_data["type"] == "Host":
            hosts[h_data["id"]].connected_switches = [switches[sid] for sid in h_data["switches"]]
    for s_data in cluster_data["switches"]:
        if s_data["type"] == "Switch":
            switches[s_data["id"]].connected_switches = [switches[sid] for sid in s_data["switches"]]

    cluster = ClusterMesh(list(hosts.values()), list(switches.values()))

    task_map_data = data["task_map"]
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
                if mem_data["type"] == "LocalMemory":
                    mem_regions.append(LocalMemory(dev, mem_data["address"], mem_data["size"]))
                elif mem_data["type"] == "RemoteMemory":
                    mem_regions.append(RemoteMemory(dev, mem_data["address"], mem_data["size"]))
            
            primitives.append(CommunicationPrimitive(initiator, op_type, mem_regions))
        
        tasks[t_data["id"]] = Task(t_data["id"], primitives, [], TaskStatus[t_data["status"]])

    for t_data in task_map_data["tasks"]:
        if t_data["type"] == "Task":
            tasks[t_data["id"]].dependencies = [tasks[did] for did in t_data.get("deps", [])]

    task_map = TaskMap(CollectiveOpType[task_map_data["op_type"]], task_map_data["data_size"], tasks)

    return CollectiveIR(
        cluster=cluster,
        collective_op=CollectiveOpType[data["collective_op"]],
        data_size_gb=data["data_size"],
        task_map=task_map
    )
