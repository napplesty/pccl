from ir.pipeline import Pass, Diagnostic, DiagnosticLevel
from ir import *
from typing import List


class HalvingAndDoublingPass(Pass):
    def run(self, ir: CollectiveIR, diags: List[Diagnostic]) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLREDUCE:
            raise ValueError("HalvingAndDoublingPass仅支持ALLREDUCE")
        
        devices = sorted(ir.cluster.devices_by_id.values(), key=lambda d: d.device_id)
        n = len(devices)
        if (n & (n-1)) != 0:
            raise ValueError("Halving-and-Doubling需要2的幂次设备")
        
        task_map = TaskMap(ir.collective_op, ir.data_size_gb)
        task_id = 0

        for k in range(n.bit_length()-1):
            primitives = []
            mask = 1 << k

            for i in range(n):
                if (i & mask) == 0:
                    dev_i = devices[i]
                    dev_j = devices[i ^ mask]

                    # Device i: Reduce from j
                    mem_i = LocalMemory(dev_i, 0, ir.data_size_gb)
                    mem_j = RemoteMemory(dev_j, 0, ir.data_size_gb)
                    primitives.append(CommunicationPrimitive(dev_i, PrimitiveOpType.REDUCE, [mem_i, mem_j]))

                    # Device j: Reduce from i
                    mem_j_local = LocalMemory(dev_j, 0, ir.data_size_gb)
                    mem_i_remote = RemoteMemory(dev_i, 0, ir.data_size_gb)
                    primitives.append(CommunicationPrimitive(dev_j, PrimitiveOpType.REDUCE, [mem_j_local, mem_i_remote]))

            dependencies = [task_map.tasks[task_id-1]] if task_id > 0 else []
            task_map.add_task(Task(task_id, primitives, dependencies))
            task_id += 1

        return CollectiveIR(ir.cluster, ir.collective_op, ir.data_size_gb, task_map)
    