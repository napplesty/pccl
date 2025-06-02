from dataclasses import dataclass
from enum import Enum
from pccl.logging import get_logger
from pccl.llang.collectives import Collective
from pccl.llang.instruction import ChunkRef

class ReplicationPolicy(Enum):
    duplicated = "duplicated"
    distributed = "distributed"

_current_program = None

def _curr():
    global _current_program
    if _current_program is None:
        raise RuntimeError("No Program in context")
    return _current_program

class MSCCLPPProgram:
    def __init__(
        self,
        name: str,
        collective: Collective,
        num_ranks: int,
        instances: int,
        protocol: str = "Simple",
        instr_fusion: bool = True,
        replication_policy: ReplicationPolicy = ReplicationPolicy.duplicated,
        num_threads_per_block: int = 1024,
        use_double_scratch_buffer: bool = False,
        min_message_size: int = 0,
        max_message_size: int = 2**64 - 1,
    ):
        self.name = name
        self.collective = collective
        self.num_ranks = num_ranks
        self.instances = instances
        self.protocol = protocol
        self.ranks = []
        self.instr_dag = None
        self.buffers = []

    def get_ref(self, rank, buffer, index, size):
        return ChunkRef(rank, buffer, index, size)

    def get_rank_ref(self, rank):
        return RankRef(rank, self)

    def check(self):
        return self.collective.check(self)

    def generate_json(self):
        return {}

@dataclass
class RankRef:
    rank: int
    prog: MSCCLPPProgram

    def _get_barrier_id(self, tb_list) -> int:
        return self.prog.ranks[self.rank].get_barrier_id(tb_list)

    def barrier(self, tb_list):
        barrier_id = self._get_barrier_id(tb_list)
        return self.prog.instr_dag.add_barrier(self.rank, tb_list, barrier_id)

def Json():
    return _curr().generate_json()

def chunk(rank, buffer, index, size=1) -> ChunkRef:
    if _curr().buffers[rank][buffer][index] is None:
        return None
    return _curr().get_ref(rank, buffer, index, size)

def rank(rank) -> RankRef:
    return _curr().get_rank_ref(rank)

def Check():
    return _curr().check()
