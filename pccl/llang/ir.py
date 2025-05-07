# from abc import ABC, abstractmethod
# from collections import defaultdict
# from dataclasses import asdict, dataclass
# import json
# from typing import Dict, List, Optional, Union

# from pccl.llang.buffer import Buffer
# from pccl.llang.collectives import Collective
# from pccl.llang.instruction import Instruction
# from pccl.llang.program import MSCCLPPProgram

# _local_src_insts_mscclpp: set = {
#     Instruction.put,
#     Instruction.signal,
#     Instruction.flush,
#     Instruction.write,
#     Instruction.reduce_write,
#     Instruction.group_write,
#     Instruction.net_configure
# }
# _local_dst_insts_mscclpp: set = {
#     Instruction.get,
#     Instruction.wait,
#     Instruction.read,
#     Instruction.read_reduce,
#     Instruction.group_read_reduce,
#     Instruction.reduce,
# }
