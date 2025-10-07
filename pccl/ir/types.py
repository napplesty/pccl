from enum import Enum, auto
from dataclasses import dataclass
from typing import Set, List, Dict, Optional, Any

class PrimitiveOpType(Enum):
    COPY = auto()
    REDUCE = auto()
    NOTIFY = auto()
    GET_NOTIFIED = auto()

class DeviceType(Enum):
    CUDA = auto()
    CPU = auto()
    ROCM = auto()
