# from dataclasses import dataclass, field
# from enum import Enum
# from typing import Union, List, Dict, Any
# from pccl.llang.buffer import Buffer

# @dataclass
# class GpuConfig:
#     """Configuration for GPU operations"""
#     precopies: list = field(default_factory=list)
#     postcopies: list = field(default_factory=list)
#     inputs: dict = field(default_factory=dict)
#     outputs: dict = field(default_factory=dict)
#     input_chunks: int = 0
#     output_chunks: int = 0
#     scratch_chunks: int = 0
#     scratch: dict = field(default_factory=dict)
#     channels: dict = field(default_factory=dict)

# @dataclass
# class Gpu:
#     rank: int
#     threadblocks: list = field(default_factory=list)
#     config: GpuConfig = field(default_factory=GpuConfig)

#     def scratch_size(self):
#         return max((idx for addr, idx in self.config.scratch.items()), default=-1) + 1

# @dataclass
# class ProgramConfig:
#     """Configuration for program execution"""
#     num_chunk_groups: int = 1
#     num_threads_per_block: int = 1024
#     use_double_scratch_buffer: bool = False
#     min_message_size: int = 0
#     max_message_size: int = 2**64 - 1

# @dataclass
# class Program:
#     name: str
#     collective: str
#     inplace: bool
#     protocol: str
#     gpus: List[Gpu] = field(default_factory=list)
#     config: ProgramConfig = field(default_factory=ProgramConfig)

# @dataclass
# class Threadblock:
#     channel: int = -1
#     send: int = -1
#     recv: int = -1
#     ops: list = field(default_factory=list)
#     rbid: int = -1
#     id: int = -1
#     channels: list = field(default_factory=list)

#     def __eq__(self, other):
#         return self is other

#     def __hash__(self):
#         return id(self)

# class Instruction(Enum):
#     start = "start"
#     nop = "nop"
#     put = "put"
#     signal = "signal"
#     flush = "flush"
#     write = "write"
#     reduce_write = "rw"
#     group_write = "gw"
#     reduce = "reduce"
#     get = "get"
#     wait = "wait"
#     read = "read"
#     read_reduce = "rr"
#     group_read_reduce = "grr"
#     barrier = "barrier"
#     net_configure = "nconf"

#     def __str__(self):
#         return self.value

# @dataclass
# class ChunkRef:
#     rank: int
#     buffer: Buffer
#     index: int
#     size: int

#     def __hash__(self):
#         return hash((self.rank, self.buffer, self.index, self.size))

# class ChannelType(Enum):
#     port = "port"
#     memory = "memory"
#     nvls = "nvls"
#     none = "none"

#     def __str__(self):
#         return self.value

# @dataclass(frozen=True)
# class Channel:
#     srcBuffer: Buffer
#     dstBuffer: Buffer
#     type: ChannelType
#     connected_to: Union[int, List[int]]

#     def __hash__(self):
#         # Ensure connected_to is converted to a tuple if it's a list
#         connected_to_hashable = (
#             tuple(self.connected_to) if isinstance(self.connected_to, list)
#             else self.connected_to
#         )
#         return hash((self.srcBuffer, self.dstBuffer, self.type, connected_to_hashable))

# @dataclass
# class OpConfig:
#     """Configuration for operation execution"""
#     depends: list = field(default_factory=list)
#     step: int = -1
#     tb: int = -1
#     prev: list = field(default_factory=list)
#     next: list = field(default_factory=list)
#     channel: int = -1
#     channel_type: ChannelType = ChannelType.none
#     nconf: str = ""
#     srcs: list = field(default_factory=list)
#     dsts: list = field(default_factory=list)
#     extra: dict = field(default_factory=dict)

# @dataclass
# class Op:
#     inst: Instruction
#     rank: int
#     src: ChunkRef
#     dst: ChunkRef
#     config: OpConfig = field(default_factory=OpConfig)

#     def cnt(self):
#         if self.src:
#             if self.dst:
#                 assert self.src.size == self.dst.size
#             return self.src.size
#         if self.dst:
#             return self.dst.size
#         return 0

#     def __eq__(self, other):
#         return self is other

#     def __hash__(self):
#         return id(self)

#     def __repr__(self):
#         return "Op(%s, %d, %s, %s, step:%d, tb:%d)" % (
#             self.inst, self.rank, self.src, self.dst,
#             self.config.step, self.config.tb
#         )
