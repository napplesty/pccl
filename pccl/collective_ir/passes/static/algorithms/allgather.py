from typing import List, Dict
from ....core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory, RemoteMemory
from ....core.enums import CollectiveOpType, PrimitiveOpType
from ...base import IRPass

class AllGatherOptimizationPass(IRPass):
    @property
    def name(self) -> str:
        return "AllGatherOptimizationPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLGATHER:
            return ir
        
        device_ids = list(ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        if n_devices < 2:
            return ir
        
        ir.task_map.tasks.clear()
        
        new_tasks = {}
        task_id = 0
        
        chunk_size = int(ir.data_size_gb * 1024 * 1024 * 1024 / n_devices)

        for step in range(n_devices - 1):
            for rank in range(n_devices):
                src_rank = (rank - step - 1) % n_devices
                dst_rank = rank
                
                src_device = ir.cluster.get_device(device_ids[src_rank])
                dst_device = ir.cluster.get_device(device_ids[dst_rank])
                chunk_idx = (rank - step - 1) % n_devices
                src_offset = chunk_idx * chunk_size
                dst_offset = chunk_idx * chunk_size
                
                src_memory = LocalMemory(src_device, src_offset, chunk_size)
                dst_memory = RemoteMemory(dst_device, dst_offset, chunk_size)
                
                primitive = CommunicationPrimitive(
                    initiator=src_device,
                    op_type=PrimitiveOpType.COPY,
                    memory_regions=[src_memory, dst_memory]
                )
                
                task = Task(task_id, [primitive])
                new_tasks[task_id] = task
                task_id += 1
        
        ir.task_map.tasks = new_tasks
        return ir

class RingAllGatherPass(IRPass):
    @property
    def name(self) -> str:
        return "RingAllGatherPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        return AllGatherOptimizationPass().run(ir)

class RecursiveDoublingAllGatherPass(IRPass):
    @property
    def name(self) -> str:
        return "RecursiveDoublingAllGatherPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLGATHER:
            return ir
        
        device_ids = list(ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        if n_devices < 2:
            return ir
        
        ir.task_map.tasks.clear()
        
        new_tasks = {}
        task_id = 0
        
        chunk_size = int(ir.data_size_gb * 1024 * 1024 * 1024 / n_devices)
        log_n = (n_devices - 1).bit_length()

        for step in range(log_n):
            stride = 1 << step
            for rank in range(n_devices):
                partner = rank ^ stride
                if partner < n_devices and rank < partner:
                    rank_device = ir.cluster.get_device(device_ids[rank])
                    partner_device = ir.cluster.get_device(device_ids[partner])

                    if rank & stride:
                        for chunk_idx in range(stride):
                            src_offset = chunk_idx * chunk_size
                            dst_offset = (chunk_idx + stride) * chunk_size
                            
                            src_memory = LocalMemory(rank_device, src_offset, chunk_size)
                            dst_memory = RemoteMemory(partner_device, dst_offset, chunk_size)
                            
                            primitive = CommunicationPrimitive(
                                initiator=rank_device,
                                op_type=PrimitiveOpType.COPY,
                                memory_regions=[src_memory, dst_memory]
                            )
                            
                            task = Task(task_id, [primitive])
                            new_tasks[task_id] = task
                            task_id += 1
                    else:
                        for chunk_idx in range(stride):
                            src_offset = (chunk_idx + stride) * chunk_size
                            dst_offset = chunk_idx * chunk_size
                            
                            src_memory = LocalMemory(rank_device, src_offset, chunk_size)
                            dst_memory = RemoteMemory(partner_device, dst_offset, chunk_size)
                            
                            primitive = CommunicationPrimitive(
                                initiator=rank_device,
                                op_type=PrimitiveOpType.COPY,
                                memory_regions=[src_memory, dst_memory]
                            )
                            
                            task = Task(task_id, [primitive])
                            new_tasks[task_id] = task
                            task_id += 1
        
        ir.task_map.tasks = new_tasks
        return ir

class BruckAllGatherPass(IRPass):
    @property
    def name(self) -> str:
        return "BruckAllGatherPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLGATHER:
            return ir
        
        device_ids = list(ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        if n_devices < 2:
            return ir
        
        ir.task_map.tasks.clear()
        
        new_tasks = {}
        task_id = 0
        
        chunk_size = int(ir.data_size_gb * 1024 * 1024 * 1024 / n_devices)

        steps = (n_devices - 1).bit_length()
        
        for step in range(steps):
            stride = 1 << step
            for rank in range(n_devices):
                partner = (rank + stride) % n_devices
                
                if partner != rank:
                    rank_device = ir.cluster.get_device(device_ids[rank])
                    partner_device = ir.cluster.get_device(device_ids[partner])

                    for chunk_idx in range(stride):
                        if (rank // stride) % 2 == 0:
                            src_offset = ((rank + chunk_idx) % n_devices) * chunk_size
                            dst_offset = src_offset
                            
                            src_memory = LocalMemory(rank_device, src_offset, chunk_size)
                            dst_memory = RemoteMemory(partner_device, dst_offset, chunk_size)
                            
                            primitive = CommunicationPrimitive(
                                initiator=rank_device,
                                op_type=PrimitiveOpType.COPY,
                                memory_regions=[src_memory, dst_memory]
                            )
                            
                            task = Task(task_id, [primitive])
                            new_tasks[task_id] = task
                            task_id += 1
        
        ir.task_map.tasks = new_tasks
        return ir
