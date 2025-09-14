from ir.pipeline import Pass, Diagnostic, DiagnosticLevel
from ir import *
from typing import List


class RingAllReducePass(Pass):
    def run(self, ir: CollectiveIR, diags: List[Diagnostic]) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLREDUCE:
            raise ValueError("RingAllReducePass仅支持ALLREDUCE")
        
        devices = sorted(ir.cluster.devices_by_id.values(), key=lambda d: d.device_id)
        n = len(devices)
        if n < 2:
            raise ValueError("Ring AllReduce需要至少2个设备")
        
        chunk_size = ir.data_size_gb / n
        task_map = TaskMap(ir.collective_op, ir.data_size_gb)
        task_id = 0

        # 阶段1: Reduce Scatter
        for r in range(n-1):
            primitives = []
            for i in range(n):
                dev_i = devices[i]
                next_dev = devices[(i+1)%n]
                prev_dev = devices[(i-1)%n]

                # Write to next device
                send_local = LocalMemory(dev_i, r*chunk_size, chunk_size)
                send_remote = RemoteMemory(next_dev, r*chunk_size, chunk_size)
                primitives.append(CommunicationPrimitive(dev_i, PrimitiveOpType.WRITE, [send_local, send_remote]))

                # Reduce from previous device
                recv_local = LocalMemory(dev_i, r*chunk_size, chunk_size)
                recv_remote = RemoteMemory(prev_dev, r*chunk_size, chunk_size)
                primitives.append(CommunicationPrimitive(dev_i, PrimitiveOpType.REDUCE, [recv_local, recv_remote]))

            dependencies = [task_map.tasks[task_id-1]] if task_id > 0 else []
            task_map.add_task(Task(task_id, primitives, dependencies))
            task_id += 1

        # 阶段2: All Gather
        for r in range(n-1):
            primitives = []
            for i in range(n):
                dev_i = devices[i]
                next_dev = devices[(i+1)%n]
                prev_dev = devices[(i-1)%n]

                # Write to next device
                send_local = LocalMemory(dev_i, (r+1)*chunk_size, chunk_size)
                send_remote = RemoteMemory(next_dev, (r+1)*chunk_size, chunk_size)
                primitives.append(CommunicationPrimitive(dev_i, PrimitiveOpType.WRITE, [send_local, send_remote]))

                # Read from previous device
                recv_local = LocalMemory(dev_i, (r+1)*chunk_size, chunk_size)
                recv_remote = RemoteMemory(prev_dev, (r+1)*chunk_size, chunk_size)
                primitives.append(CommunicationPrimitive(dev_i, PrimitiveOpType.READ, [recv_local, recv_remote]))

            dependencies = [task_map.tasks[task_id-1]]
            task_map.add_task(Task(task_id, primitives, dependencies))
            task_id += 1

        return CollectiveIR(ir.cluster, ir.collective_op, ir.data_size_gb, task_map)