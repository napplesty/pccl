from enum import Enum, auto
from typing import List, Optional
from ir import CollectiveIR

class DiagnosticLevel(Enum):
    NOTE = auto()
    WARNING = auto()
    ERROR = auto()

class Diagnostic:
    def __init__(self, level: DiagnosticLevel, message: str, location: Optional[str] = None):
        self.level = level
        self.message = message
        self.location = location

    def __str__(self):
        loc = f" ({self.location})" if self.location else ""
        return f"[{self.level.name}]{loc}: {self.message}"

class Pass:
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, ir: CollectiveIR, diagnostics: List[Diagnostic]) -> CollectiveIR:
        raise NotImplementedError
