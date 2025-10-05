from abc import ABC, abstractmethod
from ..core.ir import CollectiveIR

class IRPass(ABC):
    @abstractmethod
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    def description(self) -> str:
        return f"IR Pass: {self.name}"
