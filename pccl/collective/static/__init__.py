from .ring_pass import RingAllReducePass
from .halving_pass import HalvingAndDoublingPass
from .binary_tree_pass import DoubleBinaryTreeBroadcastPass
from ir.pipeline import PassRegistry
from ir.pipeline.canonic_passes import CanonicPass
from ir.pipeline.validation_passes import IRValidationPass

registry = PassRegistry()
registry.register("RingAllReducePass", RingAllReducePass)
registry.register("HalvingAndDoublingPass", HalvingAndDoublingPass)
registry.register("DoubleBinaryTreeBroadcastPass", DoubleBinaryTreeBroadcastPass)
registry.register("CanonicPass", CanonicPass)
registry.register("IRValidationPass", IRValidationPass)