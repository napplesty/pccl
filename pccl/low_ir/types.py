from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple, Optional

class PrimitiveType(Enum):
    WRITE = auto()
    COMPUTE = auto()
    COPY = auto()
    SIGNAL = auto()
    WAITSIGNAL = auto()

class DataType(Enum):
    F32 = auto()
    F16 = auto()
    BF16 = auto()

class ComputeType(Enum):
    SUM = auto()
    MAX = auto()
    MIN = auto()
    PROD = auto()

class ExecutorType(Enum):
    CPU = auto()
    CUDA = auto()
    LAST = auto()

@dataclass
class TensorInfo:
    shape: Tuple[int, ...]
    dtype: DataType
    device_type: ExecutorType
    device_id: int = 0
