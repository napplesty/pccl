from ....core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory, RemoteMemory
from ....core.enums import CollectiveOpType, PrimitiveOpType
from ...base import IRPass

class TreeBroadcastPass(IRPass):
    """树形Broadcast算法优化"""
    
    @property
    def name(self) -> str:
        return "TreeBroadcastPass"
    
    def run(self, ir: CollectiveIR) -> CollectiveIR:
        if ir.collective_op != CollectiveOpType.BROADCAST:
            return ir
        
        device_ids = list(ir.cluster.devices_by_id.keys())
        root_device = device_ids[0]  # 假设第一个设备是root
        
        ir.task_map.tasks.clear()
        
        new_tasks = {}
        task_id = 0
        
        def build_broadcast_tree(parent: int, children: List[int], current_task_id: int):
            tasks = {}
            
            for child in children:
                parent_device = ir.cluster.get_device(parent)
                child_device = ir.cluster.get_device(child)
                
                memory_region = LocalMemory(child_device, 0, int(ir.data_size_gb * 1024 * 1024 * 1024))
                remote_memory = RemoteMemory(parent_device, 0, int(ir.data_size_gb * 1024 * 1024 * 1024))
                
                primitive = CommunicationPrimitive(
                    initiator=parent_device,
                    op_type=PrimitiveOpType.COPY,
                    memory_regions=[remote_memory, memory_region]
                )
                
                task = Task(current_task_id, [primitive])
                tasks[current_task_id] = task
                current_task_id += 1
            
            return tasks, current_task_id
        
        children = device_ids[1:]
        tasks, task_id = build_broadcast_tree(root_device, children, task_id)
        new_tasks.update(tasks)
        
        level_children = children
        while len(level_children) > 1:
            next_level = []
            for i in range(0, len(level_children), 2):
                parent = level_children[i]
                if i + 1 < len(level_children):
                    children_group = [level_children[i + 1]]
                    tasks, task_id = build_broadcast_tree(parent, children_group, task_id)
                    new_tasks.update(tasks)
                    next_level.append(parent)
            
            level_children = next_level
        
        ir.task_map.tasks = new_tasks
        return ir
