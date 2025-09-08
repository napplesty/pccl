from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TaskNode:
    id: int
    cost: float
    metadata: Dict[str, Any]

@dataclass
class TaskDependency:
    src_id: int
    tgt_id: int

@dataclass
class 