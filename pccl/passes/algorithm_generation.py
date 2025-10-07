from .base import Pass
from ..ir.core import CollectiveIR, PrimitiveOpType, Chunk, DeviceType
from ..algorithms import ring, tree, hierarchical
from typing import List, Dict

class AlgorithmGenerationPass(Pass):
    def __init__(self, algorithm_type: str = "auto"):
        super().__init__("algorithm_generation")
        self.algorithm_type = algorithm_type
    
    def run(self, ir: CollectiveIR) -> bool:
        if self.algorithm_type == "auto":
            return self._generate_auto_algorithm(ir)
        elif self.algorithm_type == "ring":
            return ring.generate_ring_algorithm(ir)
        elif self.algorithm_type == "tree":
            return tree.generate_tree_algorithm(ir)
        elif self.algorithm_type == "hierarchical":
            return hierarchical.generate_hierarchical_algorithm(ir)
        else:
            return False
    
    def _generate_auto_algorithm(self, ir: CollectiveIR) -> bool:
        num_devices = len(ir.devices)
        total_data_size = sum(chunk.data_size for chunk in ir.precondition)
        
        if num_devices <= 4:
            return tree.generate_tree_algorithm(ir)
        elif total_data_size < 1024 * 1024:
            return ring.generate_ring_algorithm(ir)
        else:
            return hierarchical.generate_hierarchical_algorithm(ir)
