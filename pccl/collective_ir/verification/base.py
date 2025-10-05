from abc import ABC, abstractmethod
from typing import List
from ..core.ir import CollectiveIR
from .results import VerificationResult

class IRVerifier(ABC):
    @abstractmethod
    def verify(self, ir: CollectiveIR) -> VerificationResult:
        pass
