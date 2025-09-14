from enum import Enum, auto


class CollectiveOpType(Enum):
    ALLREDUCE = auto()
    BROADCAST = auto()
    GATHER = auto()
    SCATTER = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()
    