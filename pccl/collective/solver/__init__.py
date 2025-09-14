from .ilp_pass import ILPOptimizationPass
from .infomap_pass import InfomapOptimizationPass
from ir.pipeline import PassRegistry

registry = PassRegistry()
registry.register("ILPOptimizationPass", ILPOptimizationPass)
registry.register("InfomapOptimizationPass", InfomapOptimizationPass)
