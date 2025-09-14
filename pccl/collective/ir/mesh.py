from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Device:
    device_id: int
    type: str
    properties: Dict = field(default_factory=dict)


@dataclass
class Switch:
    switch_id: int
    connected_hosts: List["Host"] = field(default_factory=list)
    connected_switches: List["Switch"] = field(default_factory=list)


@dataclass
class Host:
    host_id: int
    devices: List[Device] = field(default_factory=list)
    connected_switches: List[Switch] = field(default_factory=list)

    def get_device(self, device_id: int):
        return next((d for d in self.devices if d.device_id == device_id), None)


@dataclass
class ClusterMesh:
    hosts: List[Host] = field(default_factory=list)
    switches: List[Switch] = field(default_factory=list)
    devices_by_id: Dict[int, Device] = field(init=False, default_factory=dict)

    def __post_init__(self):
        for host in self.hosts:
            for dev in host.devices:
                if dev.device_id in self.devices_by_id:
                    raise ValueError(f"Device {dev.device_id} already exists")
                self.devices_by_id[dev.device_id] = dev

    def get_device(self, device_id: int):
        return self.devices_by_id.get(device_id)
    