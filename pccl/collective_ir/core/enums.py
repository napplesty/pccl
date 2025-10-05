from enum import Enum, auto

class CollectiveOpType(Enum):
    ALLREDUCE = auto()
    REDUCE = auto()
    BROADCAST = auto()
    GATHER = auto()
    SCATTER = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()
    REDUCESCATTER = auto()

class PrimitiveOpType(Enum):
    WRITE = auto()
    COPY = auto()
    REDUCE = auto()
    NOTIFY = auto()
    GET_NOTIFIED = auto()

class MemoryType(Enum):
    DRAM = auto()
    HBM = auto()
    SHARED = auto()

class AccessPermission(Enum):
    READ_ONLY = auto()
    WRITE_ONLY = auto()
    READ_WRITE = auto()

class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

class ChunkState(Enum):
    INVALID = auto()
    VALID = auto()
    PARTIAL = auto()
    REDUCED = auto()
    SCATTERED = auto()
    GATHERED = auto()
