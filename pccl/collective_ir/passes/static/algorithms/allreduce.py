from typing import List, Dict
from ....core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory, RemoteMemory, ClusterMesh
from ....core.enums import CollectiveOpType, PrimitiveOpType
from ...base import IRPass

class RingAllReducePass(IRPass):
    
    @property
    def name(self) -> str:
        return "RingAllReducePass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLREDUCE:
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
                src_rank = (rank + step) % n_devices
                dst_rank = (rank + step + 1) % n_devices
                
                src_device = ir.cluster.get_device(device_ids[src_rank])
                dst_device = ir.cluster.get_device(device_ids[dst_rank])
                
                chunk_idx = rank
                src_offset = chunk_idx * chunk_size
                dst_offset = chunk_idx * chunk_size
                
                src_memory = LocalMemory(src_device, src_offset, chunk_size)
                dst_memory = RemoteMemory(dst_device, dst_offset, chunk_size)
                
                primitive = CommunicationPrimitive(
                    initiator=src_device,
                    op_type=PrimitiveOpType.REDUCE,
                    memory_regions=[src_memory, dst_memory],
                    chunk_updates={(dst_device.device_id, chunk_idx): {src_device.device_id, dst_device.device_id}}
                )
                
                task = Task(task_id, [primitive])
                new_tasks[task_id] = task
                task_id += 1
        
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
    
    def _create_reduce_scatter_phase(self, device_ids: List[int], start_id: int, 
                                   data_size: float, cluster: ClusterMesh) -> Dict[int, Task]:
        tasks = {}
        task_id = start_id
        n_devices = len(device_ids)
        chunk_size = int(data_size * 1024 * 1024 * 1024 / n_devices)
        
        for step in range(n_devices - 1):
            for rank in range(n_devices):
                src_rank = (rank + step) % n_devices
                dst_rank = (rank + step + 1) % n_devices
                
                src_device = cluster.get_device(device_ids[src_rank])
                dst_device = cluster.get_device(device_ids[dst_rank])
                
                chunk_idx = rank
                src_offset = chunk_idx * chunk_size
                dst_offset = chunk_idx * chunk_size
                
                src_memory = LocalMemory(src_device, src_offset, chunk_size)
                dst_memory = RemoteMemory(dst_device, dst_offset, chunk_size)
                
                primitive = CommunicationPrimitive(
                    initiator=src_device,
                    op_type=PrimitiveOpType.REDUCE,
                    memory_regions=[src_memory, dst_memory]
                )
                
                task = Task(task_id, [primitive])
                tasks[task_id] = task
                task_id += 1
        
        return tasks
    
    def _create_allgather_phase(self, device_ids: List[int], start_id: int, 
                              data_size: float, cluster: ClusterMesh) -> Dict[int, Task]:
        tasks = {}
        task_id = start_id
        n_devices = len(device_ids)
        chunk_size = int(data_size * 1024 * 1024 * 1024 / n_devices)
        
        for step in range(n_devices - 1):
            for rank in range(n_devices):
                src_rank = (rank - step - 1) % n_devices
                dst_rank = rank
                
                src_device = cluster.get_device(device_ids[src_rank])
                dst_device = cluster.get_device(device_ids[dst_rank])
                
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
                tasks[task_id] = task
                task_id += 1
        
        return tasks

class DoubleBinaryTreePass(IRPass):
    
    @property
    def name(self) -> str:
        return "DoubleBinaryTreePass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLREDUCE:
            return ir
        
        device_ids = list(ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        if n_devices < 2:
            return ir
        
        ir.task_map.tasks.clear()
        new_tasks = {}
        task_id = 0
        total_size = int(ir.data_size_gb * 1024 * 1024 * 1024)
        
        for i in range(n_devices):
            if i > 0:
                parent = (i - 1) // 2
                if parent < n_devices:
                    device = ir.cluster.get_device(device_ids[i])
                    parent_device = ir.cluster.get_device(device_ids[parent])
                    
                    memory_region = LocalMemory(device, 0, total_size)
                    remote_memory = RemoteMemory(parent_device, 0, total_size)
                    
                    primitive = CommunicationPrimitive(
                        initiator=device,
                        op_type=PrimitiveOpType.REDUCE,
                        memory_regions=[memory_region, remote_memory]
                    )
                    
                    task = Task(task_id, [primitive])
                    new_tasks[task_id] = task
                    task_id += 1
        
        for i in range(n_devices):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            if left_child < n_devices:
                device = ir.cluster.get_device(device_ids[i])
                child_device = ir.cluster.get_device(device_ids[left_child])
                
                memory_region = LocalMemory(child_device, 0, total_size)
                remote_memory = RemoteMemory(device, 0, total_size)
                
                primitive = CommunicationPrimitive(
                    initiator=device,
                    op_type=PrimitiveOpType.COPY,
                    memory_regions=[remote_memory, memory_region]
                )
                
                task = Task(task_id, [primitive])
                new_tasks[task_id] = task
                task_id += 1
                
            if right_child < n_devices:
                device = ir.cluster.get_device(device_ids[i])
                child_device = ir.cluster.get_device(device_ids[right_child])
                
                memory_region = LocalMemory(child_device, 0, total_size)
                remote_memory = RemoteMemory(device, 0, total_size)
                
                primitive = CommunicationPrimitive(
                    initiator=device,
                    op_type=PrimitiveOpType.COPY,
                    memory_regions=[remote_memory, memory_region]
                )
                
                task = Task(task_id, [primitive])
                new_tasks[task_id] = task
                task_id += 1
        
        ir.task_map.tasks = new_tasks
        return ir

class HalvingDoublingPass(IRPass):
    
    @property
    def name(self) -> str:
        return "HalvingDoublingPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLREDUCE:
            return ir
        
        device_ids = list(ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        if n_devices < 2:
            return ir
        
        ir.task_map.tasks.clear()
        new_tasks = {}
        task_id = 0
        total_size = int(ir.data_size_gb * 1024 * 1024 * 1024)
        
        steps = (n_devices - 1).bit_length()
        
        for step in range(steps):
            stride = 1 << step
            for rank in range(n_devices):
                partner = rank ^ stride
                if partner < n_devices and rank < partner:
                    rank_device = ir.cluster.get_device(device_ids[rank])
                    partner_device = ir.cluster.get_device(device_ids[partner])
                    
                    if rank & stride:
                        src_memory = RemoteMemory(partner_device, 0, total_size)
                        dst_memory = LocalMemory(rank_device, 0, total_size)
                    else:
                        src_memory = RemoteMemory(rank_device, 0, total_size)
                        dst_memory = LocalMemory(partner_device, 0, total_size)
                    
                    reduce_primitive = CommunicationPrimitive(
                        initiator=rank_device,
                        op_type=PrimitiveOpType.REDUCE,
                        memory_regions=[dst_memory, src_memory]
                    )
                    
                    task = Task(task_id, [reduce_primitive])
                    new_tasks[task_id] = task
                    task_id += 1
        
        for step in reversed(range(steps)):
            stride = 1 << step
            for rank in range(n_devices):
                partner = rank ^ stride
                if partner < n_devices and rank < partner:
                    rank_device = ir.cluster.get_device(device_ids[rank])
                    partner_device = ir.cluster.get_device(device_ids[partner])
                    
                    if rank & stride:
                        src_memory = RemoteMemory(partner_device, 0, total_size)
                        dst_memory = LocalMemory(rank_device, 0, total_size)
                    else:
                        src_memory = RemoteMemory(rank_device, 0, total_size)
                        dst_memory = LocalMemory(partner_device, 0, total_size)
                    
                    copy_primitive = CommunicationPrimitive(
                        initiator=rank_device,
                        op_type=PrimitiveOpType.COPY,
                        memory_regions=[src_memory, dst_memory]
                    )
                    
                    task = Task(task_id, [copy_primitive])
                    new_tasks[task_id] = task
                    task_id += 1
        
        ir.task_map.tasks = new_tasks
        return ir

class AllReduceAlgorithmSelectionPass(IRPass):
    
    def __init__(self, threshold_for_ring: int = 8):
        self.threshold_for_ring = threshold_for_ring
    
    @property
    def name(self) -> str:
        return "AllReduceAlgorithmSelectionPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.ALLREDUCE:
            return ir
        
        device_ids = list(ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        if n_devices <= 4:
            pass_obj = DoubleBinaryTreePass()
        elif n_devices <= self.threshold_for_ring:
            pass_obj = HalvingDoublingPass()
        else:
            pass_obj = RingAllReducePass()
        
        return pass_obj.run(ir)
