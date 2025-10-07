from abc import ABC, abstractmethod
from typing import Dict, Any
from ..ir.core import CollectiveIR

class Pass(ABC):
    def __init__(self, name: str):
        self.name = name
        self.requires = []
        self.preserves = []
    
    @abstractmethod
    def run(self, ir: CollectiveIR) -> bool:
        pass

class PassManager:
    def __init__(self):
        self.passes = []
        self.pass_results = {}
    
    def add_pass(self, pass_instance: Pass):
        self.passes.append(pass_instance)
    
    def run(self, ir: CollectiveIR) -> bool:
        self.pass_results.clear()
        
        for pass_instance in self.passes:
            success = pass_instance.run(ir)
            self.pass_results[pass_instance.name] = success
            if not success:
                return False
        
        return True
    
    def get_pass_result(self, pass_name: str) -> bool:
        return self.pass_results.get(pass_name, False)
