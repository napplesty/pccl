from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List
from .mesh import Device
from .memory import LocalMemory, RemoteMemory


class PrimitiveOpType(Enum):
    READ = auto()
    WRITE = auto()
    REDUCE = auto()
    NOTIFY = auto()


@dataclass
class CommunicationPrimitive:
    initiator: Device
    op_type: PrimitiveOpType
    memory_regions: List[LocalMemory | RemoteMemory]

    def __post_init__(self):
        if not any(isinstance(reg, LocalMemory) for reg in self.memory_regions):
            raise ValueError("Primitive must involve at least one local memory region")
        
