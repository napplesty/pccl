from ir.pipeline import Pass, Diagnostic, DiagnosticLevel
from ir import *
from typing import List


class DoubleBinaryTreeBroadcastPass(Pass):
    def run(self, ir: CollectiveIR, diags: List[Diagnostic]) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.BROADCAST:
            raise ValueError("DoubleBinaryTreeBroadcastPass仅支持BROADCAST")
        
        devices = sorted(ir.cluster.devices_by_id.values(), key=lambda d: d.device_id)
        n = len(devices)
        if n < 1:
            raise ValueError("Broadcast需要至少1个设备")
        
        task_map = TaskMap(ir.collective_op, ir.data_size_gb)
        task_id = 0
        root = devices[0]

        for level in range((n-1).bit_length()):
            primitives = []
            start = (1 << level) - 1
            end = (1 << (level+1)) - 2

            for i in range(start, min(end+1, n)):
                parent = devices[i]
                left = 2*i + 1
                right = 2*i + 2

                if left < n:
                    left_dev = devices[left]
                    mem_local = LocalMemory(parent, 0, ir.data_size_gb)
                    mem_remote = RemoteMemory(left_dev, 0, ir.data_size_gb)
                    primitives.append(CommunicationPrimitive(parent, PrimitiveOpType.WRITE, [mem_local, mem_remote]))

                if right < n:
                    right_dev = devices[right]
                    mem_local = LocalMemory(parent, 0, ir.data_size_gb)
                    mem_remote = RemoteMemory(right_dev, 0, ir.data_size_gb)
                    primitives.append(CommunicationPrimitive(parent, PrimitiveOpType.WRITE, [mem_local, mem_remote]))

            dependencies = [task_map.tasks[task_id-1]] if task_id > 0 else []
            task_map.add_task(Task(task_id, primitives, dependencies))
            task_id += 1

        return CollectiveIR(ir.cluster, ir.collective_op, ir.data_size_gb, task_map)