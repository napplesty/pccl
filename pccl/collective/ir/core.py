from dataclasses import dataclass, field
from typing import Set
from .mesh import ClusterMesh
from .collective import CollectiveOpType
from .task import TaskMap


@dataclass(frozen=True)
class CollectiveIR:
    cluster: ClusterMesh
    collective_op: CollectiveOpType
    data_size_gb: float
    task_map: TaskMap

    def __post_init__(self):
        if self.task_map.op_type != self.collective_op:
            raise ValueError(
                f"TaskMap op_type [{self.task_map.op_type.name}] "
                f"mismatches CollectiveIR op [{self.collective_op.name}]"
            )
        
        task_devices: Set[int] = set()
        for task in self.task_map.tasks.values():
            for prim in task.primitives:
                task_devices.add(prim.initiator.device_id)
                for mem in prim.memory_regions:
                    task_devices.add(mem.device.device_id)
        
        cluster_devices = set(self.cluster.devices_by_id.keys())
        missing_devices = task_devices - cluster_devices
        if missing_devices:
            raise ValueError(f"Task references devices not in cluster: {missing_devices}")
