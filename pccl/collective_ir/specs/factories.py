from typing import List, Set, Dict
from ..core.ir import CollectiveSpec, Precondition, Postcondition, Chunk, ChunkState, CollectiveOpType
from ..core.ir import Device, Host, Switch, ClusterMesh, CollectiveIR, TaskMap, CommunicationPrimitive, LocalMemory, RemoteMemory, Task
from ..core.enums import CollectiveOpType, PrimitiveOpType
from .base import CollectiveSpecBuilder, SpecValidator

def create_allreduce_spec(device_ids: List[int], data_size_gb: float) -> CollectiveSpec:
    builder = CollectiveSpecBuilder(CollectiveOpType.ALLREDUCE)
    all_devices = set(device_ids)
    n_devices = len(device_ids)

    for i, dev_id in enumerate(device_ids):
        chunk_states = {(dev_id, i): ChunkState.VALID}
        reduced_rank_requirements = {(dev_id, i): {dev_id}}
        
        precondition = Precondition(
            chunk_states=chunk_states,
            required_devices={dev_id},
            required_chunks={i},
            reduced_rank_requirements=reduced_rank_requirements
        )
        builder.add_precondition(precondition)

    for dev_id in device_ids:
        all_chunk_states = {(dev_id, j): ChunkState.REDUCED for j in range(n_devices)}
        reduced_rank_updates = {(dev_id, j): all_devices for j in range(n_devices)}
        
        postcondition = Postcondition(
            chunk_states=all_chunk_states,
            produced_devices={dev_id},
            produced_chunks=set(range(n_devices)),
            reduced_rank_updates=reduced_rank_updates
        )
        builder.add_postcondition(postcondition)

    return builder.set_devices(all_devices).set_data_size(data_size_gb).build()

def create_reduce_spec(root_device: int, device_ids: List[int], data_size_gb: float) -> CollectiveSpec:
    builder = CollectiveSpecBuilder(CollectiveOpType.REDUCE)
    all_devices = set(device_ids)

    for dev_id in device_ids:
        chunk_states = {(dev_id, 0): ChunkState.VALID}
        precondition = Precondition(
            chunk_states=chunk_states,
            required_devices={dev_id},
            required_chunks={0}
        )
        builder.add_precondition(precondition)

    root_chunk_states = {(root_device, 0): ChunkState.REDUCED}
    reduced_rank_updates = {(root_device, 0): all_devices}
    postcondition = Postcondition(
        chunk_states=root_chunk_states,
        produced_devices={root_device},
        produced_chunks={0},
        reduced_rank_updates=reduced_rank_updates
    )
    builder.add_postcondition(postcondition)

    return builder.set_devices(all_devices).set_data_size(data_size_gb).build()

def create_broadcast_spec(root_device: int, device_ids: List[int], data_size_gb: float) -> CollectiveSpec:
    builder = CollectiveSpecBuilder(CollectiveOpType.BROADCAST)
    all_devices = set(device_ids)

    root_chunk_states = {(root_device, 0): ChunkState.VALID}
    precondition = Precondition(
        chunk_states=root_chunk_states,
        required_devices={root_device},
        required_chunks={0}
    )
    builder.add_precondition(precondition)

    for dev_id in device_ids:
        chunk_states = {(dev_id, 0): ChunkState.VALID}
        postcondition = Postcondition(
            chunk_states=chunk_states,
            produced_devices={dev_id},
            produced_chunks={0}
        )
        builder.add_postcondition(postcondition)

    return builder.set_devices(all_devices).set_data_size(data_size_gb).build()

def create_alltoall_spec(device_ids: List[int], data_size_gb: float) -> CollectiveSpec:
    builder = CollectiveSpecBuilder(CollectiveOpType.ALLTOALL)
    all_devices = set(device_ids)
    n_devices = len(device_ids)

    for src_dev_id in device_ids:
        chunk_states = {(src_dev_id, j): ChunkState.VALID for j in range(n_devices)}
        precondition = Precondition(
            chunk_states=chunk_states,
            required_devices={src_dev_id},
            required_chunks=set(range(n_devices))
        )
        builder.add_precondition(precondition)

    for dst_dev_id in device_ids:
        chunk_states = {(dst_dev_id, j): ChunkState.VALID for j in range(n_devices)}
        postcondition = Postcondition(
            chunk_states=chunk_states,
            produced_devices={dst_dev_id},
            produced_chunks=set(range(n_devices))
        )
        builder.add_postcondition(postcondition)

    return builder.set_devices(all_devices).set_data_size(data_size_gb).build()

def create_scatter_spec(root_device: int, device_ids: List[int], data_size_gb: float) -> CollectiveSpec:
    builder = CollectiveSpecBuilder(CollectiveOpType.SCATTER)
    all_devices = set(device_ids)
    n_devices = len(device_ids)

    root_chunk_states = {(root_device, j): ChunkState.VALID for j in range(n_devices)}
    precondition = Precondition(
        chunk_states=root_chunk_states,
        required_devices={root_device},
        required_chunks=set(range(n_devices))
    )
    builder.add_precondition(precondition)

    for i, dev_id in enumerate(device_ids):
        chunk_states = {(dev_id, 0): ChunkState.VALID}
        postcondition = Postcondition(
            chunk_states=chunk_states,
            produced_devices={dev_id},
            produced_chunks={0}
        )
        builder.add_postcondition(postcondition)

    return builder.set_devices(all_devices).set_data_size(data_size_gb).build()

def create_gather_spec(root_device: int, device_ids: List[int], data_size_gb: float) -> CollectiveSpec:
    builder = CollectiveSpecBuilder(CollectiveOpType.GATHER)
    all_devices = set(device_ids)
    n_devices = len(device_ids)

    for i, dev_id in enumerate(device_ids):
        chunk_states = {(dev_id, 0): ChunkState.VALID}
        precondition = Precondition(
            chunk_states=chunk_states,
            required_devices={dev_id},
            required_chunks={0}
        )
        builder.add_precondition(precondition)

    root_chunk_states = {(root_device, j): ChunkState.VALID for j in range(n_devices)}
    postcondition = Postcondition(
        chunk_states=root_chunk_states,
        produced_devices={root_device},
        produced_chunks=set(range(n_devices))
    )
    builder.add_postcondition(postcondition)

    return builder.set_devices(all_devices).set_data_size(data_size_gb).build()

def create_allgather_spec(device_ids: List[int], data_size_gb: float) -> CollectiveSpec:
    builder = CollectiveSpecBuilder(CollectiveOpType.ALLGATHER)
    all_devices = set(device_ids)
    n_devices = len(device_ids)

    for dev_id in device_ids:
        chunk_states = {(dev_id, 0): ChunkState.VALID}
        precondition = Precondition(
            chunk_states=chunk_states,
            required_devices={dev_id},
            required_chunks={0}
        )
        builder.add_precondition(precondition)

    for dev_id in device_ids:
        chunk_states = {(dev_id, j): ChunkState.VALID for j in range(n_devices)}
        postcondition = Postcondition(
            chunk_states=chunk_states,
            produced_devices={dev_id},
            produced_chunks=set(range(n_devices))
        )
        builder.add_postcondition(postcondition)

    return builder.set_devices(all_devices).set_data_size(data_size_gb).build()

def create_reducescatter_spec(device_ids: List[int], data_size_gb: float) -> CollectiveSpec:
    builder = CollectiveSpecBuilder(CollectiveOpType.REDUCESCATTER)
    all_devices = set(device_ids)
    n_devices = len(device_ids)

    for i, dev_id in enumerate(device_ids):
        chunk_states = {(dev_id, j): ChunkState.VALID for j in range(n_devices)}
        precondition = Precondition(
            chunk_states=chunk_states,
            required_devices={dev_id},
            required_chunks=set(range(n_devices))
        )
        builder.add_precondition(precondition)

    for i, dev_id in enumerate(device_ids):
        chunk_states = {(dev_id, i): ChunkState.REDUCED}
        reduced_rank_updates = {(dev_id, i): all_devices}
        postcondition = Postcondition(
            chunk_states=chunk_states,
            produced_devices={dev_id},
            produced_chunks={i},
            reduced_rank_updates=reduced_rank_updates
        )
        builder.add_postcondition(postcondition)

    return builder.set_devices(all_devices).set_data_size(data_size_gb).build()

def create_simple_allreduce_ir(device_ids: List[int], data_size_gb: float = 1.0) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    host = Host(0, devices)
    cluster = ClusterMesh(hosts=[host])

    collective_spec = create_allreduce_spec(device_ids, data_size_gb)

    tasks = {}
    task_id = 0

    for i, device in enumerate(devices):
        primitive = CommunicationPrimitive(
            device,
            PrimitiveOpType.COPY,
            [LocalMemory(device, 0, int(data_size_gb * 1024 * 1024 * 1024))],
            chunk_updates={(device.device_id, i): set(device_ids)}
        )
        task = Task(task_id, [primitive])
        tasks[task_id] = task
        task_id += 1

    task_map = TaskMap(CollectiveOpType.ALLREDUCE, data_size_gb, tasks)

    return CollectiveIR(cluster, CollectiveOpType.ALLREDUCE, data_size_gb, task_map, collective_spec)

def create_simple_reduce_ir(root_device: int, device_ids: List[int], data_size_gb: float = 1.0) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    host = Host(0, devices)
    cluster = ClusterMesh(hosts=[host])

    collective_spec = create_reduce_spec(root_device, device_ids, data_size_gb)

    tasks = {}
    task_id = 0

    for i, device in enumerate(devices):
        if device.device_id != root_device:
            primitive = CommunicationPrimitive(
                device,
                PrimitiveOpType.REDUCE,
                [LocalMemory(device, 0, int(data_size_gb * 1024 * 1024 * 1024))],
                chunk_updates={(root_device, 0): {device.device_id}}
            )
            task = Task(task_id, [primitive])
            tasks[task_id] = task
            task_id += 1

    task_map = TaskMap(CollectiveOpType.REDUCE, data_size_gb, tasks)

    return CollectiveIR(cluster, CollectiveOpType.REDUCE, data_size_gb, task_map, collective_spec)

def create_simple_broadcast_ir(root_device: int, device_ids: List[int], data_size_gb: float = 1.0) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    host = Host(0, devices)
    cluster = ClusterMesh(hosts=[host])

    collective_spec = create_broadcast_spec(root_device, device_ids, data_size_gb)

    tasks = {}
    task_id = 0

    for i, device in enumerate(devices):
        if device.device_id != root_device:
            primitive = CommunicationPrimitive(
                device,
                PrimitiveOpType.COPY,
                [LocalMemory(device, 0, int(data_size_gb * 1024 * 1024 * 1024))],
                chunk_updates={(device.device_id, 0): {root_device}}
            )
            task = Task(task_id, [primitive])
            tasks[task_id] = task
            task_id += 1

    task_map = TaskMap(CollectiveOpType.BROADCAST, data_size_gb, tasks)

    return CollectiveIR(cluster, CollectiveOpType.BROADCAST, data_size_gb, task_map, collective_spec)

def create_simple_alltoall_ir(device_ids: List[int], data_size_gb: float = 1.0) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    host = Host(0, devices)
    cluster = ClusterMesh(hosts=[host])

    collective_spec = create_alltoall_spec(device_ids, data_size_gb)

    tasks = {}
    task_id = 0

    for i, src_device in enumerate(devices):
        primitives = []
        for j, dst_device in enumerate(devices):
            if i != j:
                chunk_size = int(data_size_gb * 1024 * 1024 * 1024 / len(device_ids))
                src_memory = LocalMemory(src_device, j * chunk_size, chunk_size)
                dst_memory = RemoteMemory(dst_device, i * chunk_size, chunk_size)

                primitive = CommunicationPrimitive(
                    src_device,
                    PrimitiveOpType.COPY,
                    [src_memory, dst_memory]
                )
                primitives.append(primitive)

        task = Task(task_id, primitives)
        tasks[task_id] = task
        task_id += 1

    task_map = TaskMap(CollectiveOpType.ALLTOALL, data_size_gb, tasks)

    return CollectiveIR(cluster, CollectiveOpType.ALLTOALL, data_size_gb, task_map, collective_spec)

def create_simple_scatter_ir(root_device: int, device_ids: List[int], data_size_gb: float = 1.0) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    host = Host(0, devices)
    cluster = ClusterMesh(hosts=[host])

    collective_spec = create_scatter_spec(root_device, device_ids, data_size_gb)

    tasks = {}
    task_id = 0

    for i, device in enumerate(devices):
        if device.device_id != root_device:
            chunk_size = int(data_size_gb * 1024 * 1024 * 1024 / len(device_ids))
            src_memory = LocalMemory(devices[root_device], i * chunk_size, chunk_size)
            dst_memory = RemoteMemory(device, 0, chunk_size)

            primitive = CommunicationPrimitive(
                devices[root_device],
                PrimitiveOpType.COPY,
                [src_memory, dst_memory]
            )
            task = Task(task_id, [primitive])
            tasks[task_id] = task
            task_id += 1

    task_map = TaskMap(CollectiveOpType.SCATTER, data_size_gb, tasks)

    return CollectiveIR(cluster, CollectiveOpType.SCATTER, data_size_gb, task_map, collective_spec)

def create_simple_gather_ir(root_device: int, device_ids: List[int], data_size_gb: float = 1.0) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    host = Host(0, devices)
    cluster = ClusterMesh(hosts=[host])

    collective_spec = create_gather_spec(root_device, device_ids, data_size_gb)

    tasks = {}
    task_id = 0

    for i, device in enumerate(devices):
        if device.device_id != root_device:
            chunk_size = int(data_size_gb * 1024 * 1024 * 1024 / len(device_ids))
            src_memory = LocalMemory(device, 0, chunk_size)
            dst_memory = RemoteMemory(devices[root_device], i * chunk_size, chunk_size)

            primitive = CommunicationPrimitive(
                device,
                PrimitiveOpType.COPY,
                [src_memory, dst_memory]
            )
            task = Task(task_id, [primitive])
            tasks[task_id] = task
            task_id += 1

    task_map = TaskMap(CollectiveOpType.GATHER, data_size_gb, tasks)

    return CollectiveIR(cluster, CollectiveOpType.GATHER, data_size_gb, task_map, collective_spec)

def create_simple_allgather_ir(device_ids: List[int], data_size_gb: float = 1.0) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    host = Host(0, devices)
    cluster = ClusterMesh(hosts=[host])

    collective_spec = create_allgather_spec(device_ids, data_size_gb)

    tasks = {}
    task_id = 0

    for i, src_device in enumerate(devices):
        primitives = []
        for j, dst_device in enumerate(devices):
            if i != j:
                chunk_size = int(data_size_gb * 1024 * 1024 * 1024 / len(device_ids))
                src_memory = LocalMemory(src_device, 0, chunk_size)
                dst_memory = RemoteMemory(dst_device, i * chunk_size, chunk_size)

                primitive = CommunicationPrimitive(
                    src_device,
                    PrimitiveOpType.COPY,
                    [src_memory, dst_memory]
                )
                primitives.append(primitive)

        task = Task(task_id, primitives)
        tasks[task_id] = task
        task_id += 1

    task_map = TaskMap(CollectiveOpType.ALLGATHER, data_size_gb, tasks)

    return CollectiveIR(cluster, CollectiveOpType.ALLGATHER, data_size_gb, task_map, collective_spec)

def create_simple_reducescatter_ir(device_ids: List[int], data_size_gb: float = 1.0) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    host = Host(0, devices)
    cluster = ClusterMesh(hosts=[host])

    collective_spec = create_reducescatter_spec(device_ids, data_size_gb)

    tasks = {}
    task_id = 0

    n_devices = len(device_ids)
    chunk_size = int(data_size_gb * 1024 * 1024 * 1024 / n_devices)

    for i, device in enumerate(devices):
        primitives = []
        for j in range(n_devices):
            if j != i:
                src_memory = RemoteMemory(devices[j], i * chunk_size, chunk_size)
                dst_memory = LocalMemory(device, j * chunk_size, chunk_size)

                primitive = CommunicationPrimitive(
                    device,
                    PrimitiveOpType.REDUCE,
                    [dst_memory, src_memory],
                    chunk_updates={(device.device_id, j): {device.device_id, devices[j].device_id}}
                )
                primitives.append(primitive)

        task = Task(task_id, primitives)
        tasks[task_id] = task
        task_id += 1

    task_map = TaskMap(CollectiveOpType.REDUCESCATTER, data_size_gb, tasks)

    return CollectiveIR(cluster, CollectiveOpType.REDUCESCATTER, data_size_gb, task_map, collective_spec)

def create_complex_topology_ir(device_ids: List[int], data_size_gb: float = 1.0, op_type: CollectiveOpType = CollectiveOpType.ALLREDUCE) -> CollectiveIR:
    devices = [Device(did, "GPU", memory_capacity_gb=16.0, bandwidth_gbs=25.0) for did in device_ids]
    
    host1 = Host(1, devices[:len(devices)//2])
    host2 = Host(2, devices[len(devices)//2:])
    
    switch1 = Switch(1, bandwidth_gbs=100.0, connected_hosts=[host1])
    switch2 = Switch(2, bandwidth_gbs=100.0, connected_hosts=[host2])
    switch1.connected_switches.append(switch2)
    switch2.connected_switches.append(switch1)
    
    cluster = ClusterMesh(hosts=[host1, host2], switches=[switch1, switch2])
    
    bandwidth_matrix = {}
    latency_matrix = {}
    
    for i, dev1 in enumerate(devices):
        for j, dev2 in enumerate(devices):
            if i == j:
                bandwidth_matrix[(dev1.device_id, dev2.device_id)] = float('inf')
                latency_matrix[(dev1.device_id, dev2.device_id)] = 0.0
            else:
                host1_id = 1 if i < len(devices)//2 else 2
                host2_id = 1 if j < len(devices)//2 else 2
                
                if host1_id == host2_id:
                    bandwidth_matrix[(dev1.device_id, dev2.device_id)] = 50.0
                    latency_matrix[(dev1.device_id, dev2.device_id)] = 0.1
                else:
                    bandwidth_matrix[(dev1.device_id, dev2.device_id)] = 25.0
                    latency_matrix[(dev1.device_id, dev2.device_id)] = 0.5
    
    cluster.network_topology.bandwidth_matrix = bandwidth_matrix
    cluster.network_topology.latency_matrix = latency_matrix
    
    spec_creators = {
        CollectiveOpType.ALLREDUCE: create_allreduce_spec,
        CollectiveOpType.REDUCE: lambda devs, size: create_reduce_spec(devs[0], devs, size),
        CollectiveOpType.BROADCAST: lambda devs, size: create_broadcast_spec(devs[0], devs, size),
        CollectiveOpType.ALLTOALL: create_alltoall_spec,
        CollectiveOpType.SCATTER: lambda devs, size: create_scatter_spec(devs[0], devs, size),
        CollectiveOpType.GATHER: lambda devs, size: create_gather_spec(devs[0], devs, size),
        CollectiveOpType.ALLGATHER: create_allgather_spec,
        CollectiveOpType.REDUCESCATTER: create_reducescatter_spec,
    }
    
    collective_spec = spec_creators[op_type](device_ids, data_size_gb)
    
    ir_creators = {
        CollectiveOpType.ALLREDUCE: lambda devs, size: create_simple_allreduce_ir(devs, size),
        CollectiveOpType.REDUCE: lambda devs, size: create_simple_reduce_ir(devs[0], devs, size),
        CollectiveOpType.BROADCAST: lambda devs, size: create_simple_broadcast_ir(devs[0], devs, size),
        CollectiveOpType.ALLTOALL: lambda devs, size: create_simple_alltoall_ir(devs, size),
        CollectiveOpType.SCATTER: lambda devs, size: create_simple_scatter_ir(devs[0], devs, size),
        CollectiveOpType.GATHER: lambda devs, size: create_simple_gather_ir(devs[0], devs, size),
        CollectiveOpType.ALLGATHER: lambda devs, size: create_simple_allgather_ir(devs, size),
        CollectiveOpType.REDUCESCATTER: lambda devs, size: create_simple_reducescatter_ir(devs, size),
    }
    
    base_ir = ir_creators[op_type](device_ids, data_size_gb)
    
    return CollectiveIR(cluster, op_type, data_size_gb, base_ir.task_map, collective_spec)

def create_ir_from_spec(spec: CollectiveSpec, cluster: ClusterMesh) -> CollectiveIR:
    device_ids = list(spec.involved_devices)
    
    tasks = {}
    task_id = 0
    
    for dev_id in device_ids:
        device = cluster.get_device(dev_id)
        primitive = CommunicationPrimitive(
            device,
            PrimitiveOpType.COPY,
            [LocalMemory(device, 0, int(spec.data_size_gb * 1024 * 1024 * 1024))]
        )
        task = Task(task_id, [primitive])
        tasks[task_id] = task
        task_id += 1
    
    task_map = TaskMap(spec.op_type, spec.data_size_gb, tasks)
    
    return CollectiveIR(cluster, spec.op_type, spec.data_size_gb, task_map, spec)

def validate_ir_against_spec(ir: CollectiveIR) -> bool:
    spec = ir.collective_spec
    current_states = ir.get_current_chunk_states()

    if not current_states:
        for task in ir.task_map.tasks.values():
            for primitive in task.primitives:
                for mem_region in primitive.memory_regions:
                    chunk_key = (mem_region.device.device_id, id(mem_region))
                    current_states[chunk_key] = Chunk(
                        chunk_id=id(mem_region),
                        device_id=mem_region.device.device_id,
                        state=ChunkState.VALID,
                        data_size=mem_region.size,
                        offset=mem_region.address
                    )
    
    for precondition in spec.preconditions:
        if precondition.is_satisfied(current_states):
            return True
    
    return False
