from .base import PerformanceModel
from ..core.ir import CommunicationPrimitive, Task
from ..core.enums import PrimitiveOpType

class ComputationCostModel:
    def __init__(self):
        self.computation_rates = {
            PrimitiveOpType.REDUCE: {
                "float32": 1000.0,
                "float16": 1000.0,
                "int32": 1000.0,  
                "int8": 1000.0,   
            },
            PrimitiveOpType.COPY: {
                "default": 1000.0
            }
        }
    
    def estimate_computation_time(self, op_type: PrimitiveOpType, data_size_bytes: int, 
                                data_type: str = "float32") -> float:
        if op_type not in self.computation_rates:
            return 0.0
        
        rates = self.computation_rates[op_type]
        rate_gbs = rates.get(data_type, rates.get("default", 50.0))
        
        if rate_gbs <= 0:
            return float('inf')
        
        computation_time_ms = (data_size_bytes / (rate_gbs * 1024 * 1024 * 1024)) * 1000
        return computation_time_ms
