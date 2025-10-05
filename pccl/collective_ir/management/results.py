from dataclasses import dataclass, field
from typing import List, Dict, Any
from ..core.ir import CollectiveIR

@dataclass
class PassExecutionResult:
    success: bool
    execution_time: float
    input_ir: CollectiveIR
    output_ir: CollectiveIR
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
