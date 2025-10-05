import numpy as np
from typing import Dict, List, Tuple, Set
import networkx as nx
from scipy.optimize import linprog

try:
    import infomap
    INFOMAP_AVAILABLE = True
except ImportError:
    infomap = None
    INFOMAP_AVAILABLE = False

from ...core.ir import CollectiveIR, Task, CommunicationPrimitive, LocalMemory, RemoteMemory
from ...core.enums import PrimitiveOpType
from .base import SolverBasedOptimizationPass

class BottleneckAwareMSTPass(SolverBasedOptimizationPass):
    def __init__(self, time_limit: int = 60, optimization_level: int = 1):
        super().__init__(time_limit, optimization_level)
        self.device_tree = None
        self.communication_trees = {}
        self.optimal_weights = {}
        self.min_max_normalized_flow = float('inf')
    
    @property
    def name(self) -> str:
        return f"BottleneckAwareMSTPass(level={self.optimization_level})"
    
    def _solve(self, ir: CollectiveIR) -> Dict[int, Task]:
        device_ids = list(ir.cluster.devices_by_id.keys())
        n_devices = len(device_ids)
        
        if n_devices < 2:
            return self._generate_fallback_tasks(ir)
        
        network_graph = self._build_network_graph(ir, device_ids)
        
        if not INFOMAP_AVAILABLE or n_devices < 4:
            self.device_tree = self._build_mst_fallback(network_graph)
        else:
            self.device_tree = self._build_device_tree_infomap(network_graph)
        
        self._build_communication_trees(ir, device_ids, network_graph)
        
        if n_devices <= 100:
            success = self._solve_optimization_problem(device_ids)
            if success:
                return self._generate_optimized_tasks(ir, device_ids)
        
        return self._generate_mst_based_tasks(ir, device_ids)
    
    def _build_network_graph(self, ir: CollectiveIR, device_ids: List[int]) -> nx.Graph:
        G = nx.Graph()
        topology = ir.cluster.network_topology
        
        for i, dev1_id in enumerate(device_ids):
            G.add_node(dev1_id)
            for j, dev2_id in enumerate(device_ids):
                if i < j:
                    bandwidth = topology.get_bandwidth(dev1_id, dev2_id)
                    latency = topology.get_latency(dev1_id, dev2_id)
                    if bandwidth > 0:
                        G.add_edge(dev1_id, dev2_id, 
                                 bandwidth=bandwidth, 
                                 latency=latency,
                                 weight=1.0/latency if latency > 0 else 1.0)
        return G
    
    def _build_device_tree_infomap(self, network_graph: nx.Graph) -> nx.Graph:
        if not INFOMAP_AVAILABLE:
            return self._build_mst_fallback(network_graph)
        
        try:
            infomap_network = infomap.Network()
            
            node_id_map = {}
            for i, node in enumerate(network_graph.nodes()):
                node_id_map[node] = i
                infomap_network.addNode(i)
            
            for u, v, data in network_graph.edges(data=True):
                weight = data.get('weight', 1.0)
                infomap_network.addLink(node_id_map[u], node_id_map[v], weight)
            
            # Create Infomap instance and run
            infomap_instance = infomap.Infomap("--two-level --silent")
            infomap_instance.network(infomap_network)
            infomap_instance.run()
            
            communities = {}
            for node in network_graph.nodes():
                module_index = infomap_instance.getModules()[node_id_map[node]]
                if module_index not in communities:
                    communities[module_index] = []
                communities[module_index].append(node)
            
            device_tree = nx.Graph()
            
            def build_tree_recursive(nodes):
                if len(nodes) == 1:
                    device_tree.add_node(nodes[0])
                    return nodes[0]
                
                subgraph = network_graph.subgraph(nodes)
                infomap_sub = infomap.Network()
                
                local_node_map = {}
                for i, node in enumerate(nodes):
                    local_node_map[node] = i
                    infomap_sub.addNode(i)
                
                for u, v, data in subgraph.edges(data=True):
                    weight = data.get('weight', 1.0)
                    infomap_sub.addLink(local_node_map[u], local_node_map[v], weight)
                
                infomap_sub_instance = infomap.Infomap("--two-level --silent")
                infomap_sub_instance.network(infomap_sub)
                infomap_sub_instance.run()
                
                sub_communities = {}
                for node in nodes:
                    module_index = infomap_sub_instance.getModules()[local_node_map[node]]
                    if module_index not in sub_communities:
                        sub_communities[module_index] = []
                    sub_communities[module_index].append(node)
                
                representative_nodes = []
                for comm_nodes in sub_communities.values():
                    rep_node = build_tree_recursive(comm_nodes)
                    representative_nodes.append(rep_node)
                
                if len(representative_nodes) > 1:
                    complete_subgraph = network_graph.subgraph(representative_nodes)
                    mst = nx.minimum_spanning_tree(complete_subgraph, weight='latency')
                    
                    for u, v, data in mst.edges(data=True):
                        bandwidth = data.get('bandwidth', 1.0)
                        device_tree.add_edge(u, v, bandwidth=bandwidth)
                
                return representative_nodes[0]
            
            root_rep = build_tree_recursive(list(network_graph.nodes()))
            return device_tree
            
        except Exception as e:
            print(f"  Infomap failed: {e}, using fallback MST")
            return self._build_mst_fallback(network_graph)
    
    def _build_mst_fallback(self, network_graph: nx.Graph) -> nx.Graph:
        return nx.minimum_spanning_tree(network_graph, weight='latency')
    
    def _build_communication_trees(self, ir: CollectiveIR, device_ids: List[int], network_graph: nx.Graph):
        topology = ir.cluster.network_topology
        
        for device_id in device_ids:
            shortest_paths = nx.single_source_dijkstra_path(network_graph, device_id, weight='latency')
            comm_tree = nx.Graph()
            
            for target_id, path in shortest_paths.items():
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    bandwidth = topology.get_bandwidth(u, v)
                    comm_tree.add_edge(u, v, bandwidth=bandwidth)
            
            self.communication_trees[device_id] = comm_tree
    
    def _solve_optimization_problem(self, device_ids: List[int]) -> bool:
        n_devices = len(device_ids)
        
        A_matrix = []
        b_vector = []
        
        device_tree_edges = list(self.device_tree.edges())
        
        for edge in device_tree_edges:
            u, v = edge
            bandwidth = self.device_tree[u][v].get('bandwidth', 1.0)
            
            a_row = np.zeros(n_devices)
            for i, device_id in enumerate(device_ids):
                comm_tree = self.communication_trees[device_id]
                if self._is_edge_in_tree(comm_tree, u, v):
                    a_row[i] = 1.0
            
            A_matrix.append(a_row / bandwidth)
            b_vector.append(0.0)
        
        A_ub = np.array(A_matrix)
        b_ub = np.array(b_vector)
        
        A_eq = np.ones((1, n_devices))
        b_eq = np.array([1.0])
        
        bounds = [(0.0, 1.0) for _ in range(n_devices)]
        
        c = np.zeros(n_devices + 1)
        c[-1] = 1.0
        
        A_extended = []
        for i in range(len(A_ub)):
            row = np.concatenate([A_ub[i], [-1.0]])
            A_extended.append(row)
        
        A_extended = np.array(A_extended)
        
        try:
            result = linprog(c, A_ub=A_extended, b_ub=b_ub, 
                           A_eq=np.hstack([A_eq, np.zeros((1, 1))]), 
                           b_eq=b_eq, bounds=bounds + [(None, None)])
            
            if result.success:
                solution = result.x
                self.min_max_normalized_flow = solution[-1]
                
                for i, device_id in enumerate(device_ids):
                    self.optimal_weights[device_id] = solution[i]
                
                return True
        except:
            pass
        
        return False
    
    def _is_edge_in_tree(self, tree: nx.Graph, u: int, v: int) -> bool:
        return tree.has_edge(u, v) or tree.has_edge(v, u)
    
    def _generate_optimized_tasks(self, ir: CollectiveIR, device_ids: List[int]) -> Dict[int, Task]:
        tasks = {}
        task_id = 0
        n_devices = len(device_ids)
        
        total_data_size = int(ir.data_size_gb * 1024 * 1024 * 1024)
        
        for round_num in range(n_devices - 1):
            for i, src_id in enumerate(device_ids):
                dst_id = (src_id + round_num + 1) % n_devices
                if src_id == dst_id:
                    continue
                
                src_weight = self.optimal_weights.get(src_id, 1.0/n_devices)
                dst_weight = self.optimal_weights.get(dst_id, 1.0/n_devices)
                
                data_size_src = int(total_data_size * src_weight)
                data_size_dst = int(total_data_size * dst_weight)
                
                src_device = ir.cluster.get_device(src_id)
                dst_device = ir.cluster.get_device(dst_id)
                
                if data_size_src > 0:
                    src_memory = LocalMemory(src_device, 0, data_size_src)
                    dst_memory = RemoteMemory(dst_device, 0, data_size_src)
                    
                    primitive = CommunicationPrimitive(
                        initiator=src_device,
                        op_type=PrimitiveOpType.COPY,
                        memory_regions=[src_memory, dst_memory]
                    )
                    
                    task = Task(task_id, [primitive])
                    tasks[task_id] = task
                    task_id += 1
                
                if data_size_dst > 0:
                    src_memory_dst = LocalMemory(dst_device, 0, data_size_dst)
                    dst_memory_src = RemoteMemory(src_device, 0, data_size_dst)
                    
                    primitive = CommunicationPrimitive(
                        initiator=dst_device,
                        op_type=PrimitiveOpType.COPY,
                        memory_regions=[src_memory_dst, dst_memory_src]
                    )
                    
                    task = Task(task_id, [primitive])
                    tasks[task_id] = task
                    task_id += 1
        
        return tasks
    
    def _generate_mst_based_tasks(self, ir: CollectiveIR, device_ids: List[int]) -> Dict[int, Task]:
        tasks = {}
        task_id = 0
        
        total_data_size = int(ir.data_size_gb * 1024 * 1024 * 1024)
        avg_data_size = total_data_size // len(device_ids)
        
        if self.device_tree is not None:
            for u, v in self.device_tree.edges():
                src_device = ir.cluster.get_device(u)
                dst_device = ir.cluster.get_device(v)
                
                src_memory = LocalMemory(src_device, 0, avg_data_size)
                dst_memory = RemoteMemory(dst_device, 0, avg_data_size)
                
                primitive1 = CommunicationPrimitive(
                    initiator=src_device,
                    op_type=PrimitiveOpType.COPY,
                    memory_regions=[src_memory, dst_memory]
                )
                
                primitive2 = CommunicationPrimitive(
                    initiator=dst_device,
                    op_type=PrimitiveOpType.COPY,
                    memory_regions=[dst_memory, src_memory]
                )
                
                task1 = Task(task_id, [primitive1])
                tasks[task_id] = task1
                task_id += 1
                
                task2 = Task(task_id, [primitive2])
                tasks[task_id] = task2
                task_id += 1
        
        return tasks
    
    def _generate_fallback_tasks(self, ir: CollectiveIR) -> Dict[int, Task]:
        device_ids = list(ir.cluster.devices_by_id.keys())
        tasks = {}
        task_id = 0
        
        total_data_size = int(ir.data_size_gb * 1024 * 1024 * 1024)
        avg_data_size = total_data_size // max(1, len(device_ids))
        
        for i, src_id in enumerate(device_ids):
            for j, dst_id in enumerate(device_ids):
                if i != j:
                    src_device = ir.cluster.get_device(src_id)
                    dst_device = ir.cluster.get_device(dst_id)
                    
                    src_memory = LocalMemory(src_device, 0, avg_data_size)
                    dst_memory = RemoteMemory(dst_device, 0, avg_data_size)
                    
                    primitive = CommunicationPrimitive(
                        initiator=src_device,
                        op_type=PrimitiveOpType.COPY,
                        memory_regions=[src_memory, dst_memory]
                    )
                    
                    task = Task(task_id, [primitive])
                    tasks[task_id] = task
                    task_id += 1
        
        return tasks
    
    def get_optimization_stats(self) -> Dict[str, float]:
        return {
            'min_max_normalized_flow': self.min_max_normalized_flow,
            'optimal_weights_sum': sum(self.optimal_weights.values()) if self.optimal_weights else 0.0,
            'device_tree_edges': len(self.device_tree.edges()) if self.device_tree else 0,
            'communication_trees_count': len(self.communication_trees)
        }
