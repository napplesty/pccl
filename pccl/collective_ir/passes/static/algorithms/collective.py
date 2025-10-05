from ....core.ir import CollectiveIR
from ....core.enums import CollectiveOpType
from ...base import IRPass
from .allreduce import RingAllReducePass, DoubleBinaryTreePass, HalvingDoublingPass, AllReduceAlgorithmSelectionPass
from .broadcast import TreeBroadcastPass
from .alltoall import AllToAllOptimizationPass, PairwiseExchangeAllToAllPass
from .allgather import AllGatherOptimizationPass, RingAllGatherPass, RecursiveDoublingAllGatherPass, BruckAllGatherPass
from .reducescatter import ReduceScatterOptimizationPass, RingReduceScatterPass, PairwiseReduceScatterPass

class CollectiveOptimizationPass(IRPass):
    def __init__(self, algorithm_selection_strategy: str = "auto"):
        self.algorithm_selection_strategy = algorithm_selection_strategy
    
    @property
    def name(self) -> str:
        return f"CollectiveOptimizationPass({self.algorithm_selection_strategy})"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        device_ids = list(ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        if ir.collective_op == CollectiveOpType.ALLREDUCE:
            return self._optimize_allreduce(ir, n_devices)
        elif ir.collective_op == CollectiveOpType.BROADCAST:
            return self._optimize_broadcast(ir, n_devices)
        elif ir.collective_op == CollectiveOpType.ALLTOALL:
            return self._optimize_alltoall(ir, n_devices)
        elif ir.collective_op == CollectiveOpType.ALLGATHER:
            return self._optimize_allgather(ir, n_devices)
        elif ir.collective_op == CollectiveOpType.REDUCESCATTER:
            return self._optimize_reducescatter(ir, n_devices)
        elif ir.collective_op == CollectiveOpType.SCATTER:
            return self._optimize_scatter(ir, n_devices)
        elif ir.collective_op == CollectiveOpType.GATHER:
            return self._optimize_gather(ir, n_devices)
        
        return ir
    
    def _optimize_allreduce(self, ir: CollectiveIR, n_devices: int) -> CollectiveIR:
        if self.algorithm_selection_strategy == "auto":
            if n_devices <= 4:
                return DoubleBinaryTreePass().run(ir)
            elif n_devices <= 8:
                return HalvingDoublingPass().run(ir)
            else:
                return RingAllReducePass().run(ir)
        elif self.algorithm_selection_strategy == "ring":
            return RingAllReducePass().run(ir)
        elif self.algorithm_selection_strategy == "tree":
            return DoubleBinaryTreePass().run(ir)
        elif self.algorithm_selection_strategy == "butterfly":
            return HalvingDoublingPass().run(ir)
        else:
            return AllReduceAlgorithmSelectionPass().run(ir)
    
    def _optimize_broadcast(self, ir: CollectiveIR, n_devices: int) -> CollectiveIR:
        return TreeBroadcastPass().run(ir)
    
    def _optimize_alltoall(self, ir: CollectiveIR, n_devices: int) -> CollectiveIR:
        if n_devices <= 8:
            return AllToAllOptimizationPass().run(ir)
        else:
            return PairwiseExchangeAllToAllPass().run(ir)
    
    def _optimize_allgather(self, ir: CollectiveIR, n_devices: int) -> CollectiveIR:
        if self.algorithm_selection_strategy == "auto":
            # 检查设备数量是否为2的幂次
            if (n_devices & (n_devices - 1)) == 0:  # 2的幂次
                if n_devices <= 16:
                    return RingAllGatherPass().run(ir)
                else:
                    return RecursiveDoublingAllGatherPass().run(ir)
            else:
                return BruckAllGatherPass().run(ir)
        elif self.algorithm_selection_strategy == "ring":
            return RingAllGatherPass().run(ir)
        elif self.algorithm_selection_strategy == "recursive_doubling":
            return RecursiveDoublingAllGatherPass().run(ir)
        elif self.algorithm_selection_strategy == "bruck":
            return BruckAllGatherPass().run(ir)
        else:
            return AllGatherOptimizationPass().run(ir)
    
    def _optimize_reducescatter(self, ir: CollectiveIR, n_devices: int) -> CollectiveIR:
        if self.algorithm_selection_strategy == "auto":
            if n_devices <= 8:
                return RingReduceScatterPass().run(ir)
            else:
                return PairwiseReduceScatterPass().run(ir)
        elif self.algorithm_selection_strategy == "ring":
            return RingReduceScatterPass().run(ir)
        elif self.algorithm_selection_strategy == "pairwise":
            return PairwiseReduceScatterPass().run(ir)
        else:
            return ReduceScatterOptimizationPass().run(ir)
    
    def _optimize_scatter(self, ir: CollectiveIR, n_devices: int) -> CollectiveIR:
        return TreeBroadcastPass().run(ir)
    
    def _optimize_gather(self, ir: CollectiveIR, n_devices: int) -> CollectiveIR:
        return TreeBroadcastPass().run(ir)
