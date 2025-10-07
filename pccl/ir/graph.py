from typing import List, Dict, Set, Any, Optional
from abc import ABC, abstractmethod

class GraphNode:
    def __init__(self, node_id: int, data: Any = None):
        self.node_id = node_id
        self.data = data
        self.in_edges = []
        self.out_edges = []

class GraphEdge:
    def __init__(self, src_id: int, dst_id: int, data: Any = None):
        self.src_id = src_id
        self.dst_id = dst_id
        self.data = data

class DirectedGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._next_node_id = 0
    
    def add_node(self, data: Any = None) -> int:
        node_id = self._next_node_id
        self.nodes[node_id] = GraphNode(node_id, data)
        self._next_node_id += 1
        return node_id
    
    def add_edge(self, src_id: int, dst_id: int, data: Any = None):
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise ValueError("Invalid node IDs")
        
        edge = GraphEdge(src_id, dst_id, data)
        self.edges.append(edge)
        self.nodes[src_id].out_edges.append(edge)
        self.nodes[dst_id].in_edges.append(edge)
    
    def remove_edge(self, src_id: int, dst_id: int):
        edge_to_remove = None
        for edge in self.edges:
            if edge.src_id == src_id and edge.dst_id == dst_id:
                edge_to_remove = edge
                break
        
        if edge_to_remove:
            self.edges.remove(edge_to_remove)
            self.nodes[src_id].out_edges.remove(edge_to_remove)
            self.nodes[dst_id].in_edges.remove(edge_to_remove)
    
    def get_node(self, node_id: int) -> Optional[GraphNode]:
        return self.nodes.get(node_id)
    
    def get_edges(self) -> List[GraphEdge]:
        return self.edges.copy()
    
    def get_predecessors(self, node_id: int) -> List[int]:
        node = self.get_node(node_id)
        if not node:
            return []
        return [edge.src_id for edge in node.in_edges]
    
    def get_successors(self, node_id: int) -> List[int]:
        node = self.get_node(node_id)
        if not node:
            return []
        return [edge.dst_id for edge in node.out_edges]
    
    def is_acyclic(self) -> bool:
        visited = set()
        recursion_stack = set()
        
        def dfs(node_id):
            if node_id in recursion_stack:
                return False
            if node_id in visited:
                return True
            
            visited.add(node_id)
            recursion_stack.add(node_id)
            
            for succ_id in self.get_successors(node_id):
                if not dfs(succ_id):
                    return False
            
            recursion_stack.remove(node_id)
            return True
        
        for node_id in self.nodes:
            if node_id not in visited:
                if not dfs(node_id):
                    return False
        return True
    
    def topological_sort(self) -> List[int]:
        if not self.is_acyclic():
            raise ValueError("Graph contains cycles")
        
        visited = set()
        result = []
        
        def dfs(node_id):
            visited.add(node_id)
            for succ_id in self.get_successors(node_id):
                if succ_id not in visited:
                    dfs(succ_id)
            result.append(node_id)
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return result[::-1]
    
    def get_connected_components(self) -> List[List[int]]:
        visited = set()
        components = []
        
        def dfs(node_id, component):
            visited.add(node_id)
            component.append(node_id)
            for succ_id in self.get_successors(node_id):
                if succ_id not in visited:
                    dfs(succ_id, component)
            for pred_id in self.get_predecessors(node_id):
                if pred_id not in visited:
                    dfs(pred_id, component)
        
        for node_id in self.nodes:
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                components.append(component)
        
        return components
