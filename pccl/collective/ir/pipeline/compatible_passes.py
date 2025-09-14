from .passes import Pass, Diagnostic, DiagnosticLevel
from ir import CollectiveIR, CollectiveOpType
from typing import List


class AlgorithmCompatibilityPass(Pass):
    def __init__(self, algorithm: str):
        self.algorithm = algorithm

    def run(self, ir: CollectiveIR, diags: List[Diagnostic]) -> CollectiveIR:
        check_fn = getattr(self, f"_check_{self.algorithm}", None)
        if not check_fn:
            diags.append(Diagnostic(
                DiagnosticLevel.ERROR,
                f"Unknown algorithm: {self.algorithm}",
                loc="AlgorithmCompatibility"
            ))
            return ir
        check_fn(ir, diags)
        return ir

    def _check_ring_allreduce(self, ir: CollectiveIR, diags: List[Diagnostic]):
        if ir.collective_op != CollectiveOpType.ALLREDUCE:
            diags.append(Diagnostic(
                DiagnosticLevel.ERROR,
                f"Ring AllReduce requires ALLREDUCE, got {ir.collective_op.name}",
                loc="CollectiveIR.collective_op"
            ))
        n_dev = len(ir.cluster.devices_by_id)
        if n_dev < 2:
            diags.append(Diagnostic(
                DiagnosticLevel.ERROR,
                f"Ring AllReduce requires ≥2 devices, got {n_dev}",
                loc="ClusterMesh.devices"
            ))

    def _check_halving_doubling_allreduce(self, ir: CollectiveIR, diags: List[Diagnostic]):
        if ir.collective_op != CollectiveOpType.ALLREDUCE:
            diags.append(Diagnostic(
                DiagnosticLevel.ERROR,
                f"Halving-Doubling requires ALLREDUCE, got {ir.collective_op.name}",
                loc="CollectiveIR.collective_op"
            ))
        n_dev = len(ir.cluster.devices_by_id)
        if (n_dev & (n_dev - 1)) != 0:
            diags.append(Diagnostic(
                DiagnosticLevel.ERROR,
                f"Halving-Doubling requires power-of-2 devices, got {n_dev}",
                loc="ClusterMesh.devices"
            ))

    def _check_double_binary_tree_broadcast(self, ir: CollectiveIR, diags: List[Diagnostic]):
        if ir.collective_op != CollectiveOpType.BROADCAST:
            diags.append(Diagnostic(
                DiagnosticLevel.ERROR,
                f"Double Binary Tree requires BROADCAST, got {ir.collective_op.name}",
                loc="CollectiveIR.collective_op"
            ))
        n_dev = len(ir.cluster.devices_by_id)
        if n_dev < 1:
            diags.append(Diagnostic(
                DiagnosticLevel.ERROR,
                f"Broadcast requires ≥1 device, got {n_dev}",
                loc="ClusterMesh.devices"
            ))