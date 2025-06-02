from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from pccl.logging import get_logger
from pccl.llang.channels import ChannelType

class NetworkRole(Enum):
    Device = 0
    Switch = 1
    OpticalSwitch = 2

@dataclass
class Device:
    role: NetworkRole = NetworkRole.Device
    rank: int
    nid: int
    available_channels: List[ChannelType]
    network_infos: List[str]
    connected_to: List[int]

@dataclass
class Switch:
    role: NetworkRole = NetworkRole.Switch
    n_id: int
    control_plane: str
    connected_to: List[int]

@dataclass
class OpticalSwitch:
    role: NetworkRole = NetworkRole.OpticalSwitch
    n_id: int
    control_plane: str
    connected_to: List[int]

class NetworkContext:
    def __init__(self):
        self.devices: List[Device] = []
        self.switches: List[Switch] = []
        self.optical_switches: List[OpticalSwitch] = []

    def add_device(self, device: Device):
        self.devices.append(device)

    def add_switch(self, switch: Switch):
        self.switches.append(switch)

    def add_optical_switch(self, optical_switch: OpticalSwitch):
        self.optical_switches.append(optical_switch)

    def get_device(self, rank: int) -> Device:
        for device in self.devices:
            if device.rank == rank:
                return device