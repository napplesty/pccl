from ..ir.core import CollectiveIR, Chunk, Device, Link, DeviceType
from typing import List, Set

def generate_allreduce_spec(devices: List[Device], links: List[Link], 
                          ranks: List[int], data_size: int) -> CollectiveIR:
    ir = CollectiveIR("allreduce")
    
    for device in devices:
        ir.add_device(device)
    
    for link in links:
        ir.add_link(link)
    
    precondition = []
    postcondition = []
    
    for rank in ranks:
        precondition_chunk = Chunk(
            reduced_ranks={rank},
            cur_device_id=rank,
            data_size=data_size,
            offset=0
        )
        precondition.append(precondition_chunk)
        
        postcondition_chunk = Chunk(
            reduced_ranks=set(ranks),
            cur_device_id=rank,
            data_size=data_size,
            offset=0
        )
        postcondition.append(postcondition_chunk)
    
    ir.set_precondition(precondition)
    ir.set_postcondition(postcondition)
    
    return ir

def generate_allgather_spec(devices: List[Device], links: List[Link],
                          ranks: List[int], chunk_size: int) -> CollectiveIR:
    ir = CollectiveIR("allgather")
    
    for device in devices:
        ir.add_device(device)
    
    for link in links:
        ir.add_link(link)
    
    precondition = []
    postcondition = []
    total_size = chunk_size * len(ranks)
    
    for i, rank in enumerate(ranks):
        precondition_chunk = Chunk(
            reduced_ranks={rank},
            cur_device_id=rank,
            data_size=chunk_size,
            offset=i * chunk_size
        )
        precondition.append(precondition_chunk)
        
        postcondition_chunk = Chunk(
            reduced_ranks=set(ranks),
            cur_device_id=rank,
            data_size=total_size,
            offset=0
        )
        postcondition.append(postcondition_chunk)
    
    ir.set_precondition(precondition)
    ir.set_postcondition(postcondition)
    
    return ir

def generate_reduce_scatter_spec(devices: List[Device], links: List[Link],
                               ranks: List[int], total_size: int) -> CollectiveIR:
    ir = CollectiveIR("reduce_scatter")
    
    for device in devices:
        ir.add_device(device)
    
    for link in links:
        ir.add_link(link)
    
    precondition = []
    postcondition = []
    chunk_size = total_size // len(ranks)
    
    for rank in ranks:
        precondition_chunk = Chunk(
            reduced_ranks={rank},
            cur_device_id=rank,
            data_size=total_size,
            offset=0
        )
        precondition.append(precondition_chunk)
    
    for i, rank in enumerate(ranks):
        postcondition_chunk = Chunk(
            reduced_ranks=set(ranks),
            cur_device_id=rank,
            data_size=chunk_size,
            offset=i * chunk_size
        )
        postcondition.append(postcondition_chunk)
    
    ir.set_precondition(precondition)
    ir.set_postcondition(postcondition)
    
    return ir

def generate_broadcast_spec(devices: List[Device], links: List[Link],
                          ranks: List[int], root_rank: int, data_size: int) -> CollectiveIR:
    ir = CollectiveIR("broadcast")
    
    for device in devices:
        ir.add_device(device)
    
    for link in links:
        ir.add_link(link)
    
    precondition = []
    postcondition = []
    
    for rank in ranks:
        if rank == root_rank:
            precondition_chunk = Chunk(
                reduced_ranks={rank},
                cur_device_id=rank,
                data_size=data_size,
                offset=0
            )
        else:
            precondition_chunk = Chunk(
                reduced_ranks=set(),
                cur_device_id=rank,
                data_size=data_size,
                offset=0
            )
        
        precondition.append(precondition_chunk)
        
        postcondition_chunk = Chunk(
            reduced_ranks={root_rank},
            cur_device_id=rank,
            data_size=data_size,
            offset=0
        )
        postcondition.append(postcondition_chunk)
    
    ir.set_precondition(precondition)
    ir.set_postcondition(postcondition)
    
    return ir
