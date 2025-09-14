from dataclasses import dataclass
from .mesh import Device


@dataclass
class LocalMemory:
    device: Device
    address: int
    size: int


@dataclass
class RemoteMemory:
    device: Device
    address: int
    size: int
