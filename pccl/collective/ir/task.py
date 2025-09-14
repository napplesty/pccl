from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict
from .primitive import CommunicationPrimitive
from .collective import CollectiveOpType


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()


@dataclass
class Task:
    task_id: int
    primitives: List[CommunicationPrimitive]
    dependencies: List["Task"] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class TaskMap:
    op_type: CollectiveOpType
    data_size_gb: float
    tasks: Dict[int, Task] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists")
        self.tasks[task.task_id] = task
