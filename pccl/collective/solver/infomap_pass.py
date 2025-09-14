from ir.pipeline import Pass, Diagnostic, DiagnosticLevel
from ir import *
import math
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple


class SimpleInfomap:
    """自研简化版Infomap：基于随机游走的社区检测"""
    def __init__(self, graph: Dict[int, Dict[int, float]], max_iter: int = 100, tolerance: float = 1e-6):
        self.graph = graph
        self.nodes = list(graph.keys())
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.modules = {n: n for n in self.nodes}
        self.module_nodes = defaultdict(list)
        for n in self.nodes:
            self.module_nodes[n].append(n)
        self.transition = self._compute_transition_probs()

    def _compute_transition_probs(self) -> Dict[int, Dict[int, float]]:
        trans = defaultdict(dict)
        for u in self.nodes:
            total = sum(self.graph[u].values())
            if total == 0:
                continue
            for v, bw in self.graph[u].items():
                trans[u][v] = bw / total
        return trans

    def _encode_length(self) -> float:
        visits = self._node_visit_probs()
        module_exit = defaultdict(float)
        
        for u in self.nodes:
            exit_prob = sum(self.transition.get(u, {}).get(v, 0) for v in self.graph.get(u, {}) if self.modules[v] != self.modules[u])
            module_exit[self.modules[u]] += visits[u] * exit_prob

        length = 0.0
        for mod, nodes in self.module_nodes.items():
            exit_prob = module_exit.get(mod, 0)
            if exit_prob == 0:
                continue
            length += exit_prob * math.log2(1 / exit_prob)
            for u in nodes:
                visit_prob = visits.get(u, 0)
                if visit_prob == 0:
                    continue
                length += visit_prob * math.log2(exit_prob / visit_prob)
        return length

    def _node_visit_probs(self) -> Dict[int, float]:
        visits = {n: 1.0 / len(self.nodes) for n in self.nodes}
        for _ in range(10):
            new_visits = defaultdict(float)
            for u in self.nodes:
                for v, p in self.transition.get(u, {}).items():
                    new_visits[v] += visits[u] * p
            visits = new_visits
        total = sum(visits.values())
        return {n: v / total for n, v in visits.items()} if total else visits

    def _move_node(self, node: int, new_mod: int):
        old_mod = self.modules[node]
        self.module_nodes[old_mod].remove(node)
        if not self.module_nodes[old_mod]:
            del self.module_nodes[old_mod]
        self.modules[node] = new_mod
        self.module_nodes[new_mod].append(node)

    def run(self) -> Dict[int, int]:
        current_len = self._encode_length()
        for _ in range(self.max_iter):
            improved = False
            for node in self.nodes:
                original_mod = self.modules[node]
                best_mod = original_mod
                best_len = current_len

                for neighbor in self.graph.get(node, {}):
                    candidate_mod = self.modules[neighbor]
                    if candidate_mod == original_mod:
                        continue
                    self._move_node(node, candidate_mod)
                    new_len = self._encode_length()
                    if new_len < best_len - self.tolerance:
                        best_len = new_len
                        best_mod = candidate_mod
                    self._move_node(node, original_mod)

                if best_mod != original_mod:
                    self._move_node(node, best_mod)
                    current_len = best_len
                    improved = True
            if not improved:
                break
        return self.modules


class InfomapOptimizationPass(Pass):
    """Infomap优化Pass：社区划分+通信路径优化"""
    def run(self, ir: CollectiveIR, diags: List[Diagnostic]) -> CollectiveIR:
        # 1. 构建设备带宽图
        bw_graph = self._build_bandwidth_graph(ir)
        if not bw_graph:
            diags.append(Diagnostic(DiagnosticLevel.WARNING, "Empty bandwidth graph", loc="InfomapOptimizationPass"))
            return ir

        # 2. 社区划分
        infomap = SimpleInfomap(bw_graph)
        community = infomap.run()
        communities = self._group_by_community(community)
        diags.append(Diagnostic(DiagnosticLevel.NOTE, f"Found {len(communities)} communities: {communities}", loc="InfomapOptimizationPass"))

        # 3. 计算优化前后关键路径时间
        pre_time = self._calculate_critical_path(ir)
        optimized_ir = self._optimize_ir(ir, communities, bw_graph)
        post_time = self._calculate_critical_path(optimized_ir)

        # 4. 输出效率提升
        if pre_time > 1e-9:
            gain = ((pre_time - post_time) / pre_time) * 100
            diags.append(Diagnostic(DiagnosticLevel.NOTE, f"Infomap优化: {pre_time:.4f}s → {post_time:.4f}s (提升{gain:.2f}%)", loc="InfomapOptimizationPass"))
        else:
            diags.append(Diagnostic(DiagnosticLevel.WARNING, "优化前关键路径时间为0", loc="InfomapOptimizationPass"))

        return optimized_ir

    def _build_bandwidth_graph(self, ir: CollectiveIR) -> Dict[int, Dict[int, float]]:
        graph = defaultdict(dict)
        devices = list(ir.cluster.devices_by_id.values())
        for u in devices:
            for v in devices:
                if u == v:
                    continue
                path = ir.cluster.get_optimal_path(u, v)
                graph[u.device_id][v.device_id] = path["min_bandwidth"]
        return graph

    def _group_by_community(self, community: Dict[int, int]) -> Dict[int, List[int]]:
        groups = defaultdict(list)
        for dev_id, comm_id in community.items():
            groups[comm_id].append(dev_id)
        return groups

    def _calculate_critical_path(self, ir: CollectiveIR) -> float:
        if not ir.task_map.tasks:
            return 0.0

        G = nx.DiGraph()
        task_times = {}
        for task in ir.task_map.tasks.values():
            task_time = max(self._primitive_time(p, ir) for p in task.primitives) if task.primitives else 0.0
            task_times[task.task_id] = task_time
            G.add_node(task.task_id, weight=task_time)
            for dep in task.dependencies:
                G.add_edge(dep.task_id, task.task_id)

        if not nx.is_directed_acyclic_graph(G):
            return float("inf")
        return sum(task_times[t] for t in nx.dag_longest_path(G, weight="weight"))

    def _primitive_time(self, prim: CommunicationPrimitive, ir: CollectiveIR) -> float:
        local_mem = next(m for m in prim.memory_regions if isinstance(m, LocalMemory))
        remote_mem = next(m for m in prim.memory_regions if isinstance(m, RemoteMemory))
        path = ir.cluster.get_optimal_path(local_mem.device, remote_mem.device)
        return path["total_latency"] + (local_mem.size / path["min_bandwidth"])

    def _optimize_ir(self, ir: CollectiveIR, communities: Dict[int, List[int]], bw_graph: Dict[int, Dict[int, float]]) -> CollectiveIR:
        optimized_task_map = TaskMap(ir.collective_op, ir.data_size_gb)
        devices = ir.cluster.devices_by_id

        # 1. 社区内生成最小生成树（MST）
        community_mst = {}
        for comm_id, dev_ids in communities.items():
            subgraph = nx.Graph()
            for u in dev_ids:
                for v in dev_ids:
                    if u != v:
                        subgraph.add_edge(u, v, weight=1/bw_graph[u][v])
            community_mst[comm_id] = nx.minimum_spanning_tree(subgraph)

        # 2. 优化任务原语（社区内MST通信，跨社区高带宽链路）
        for task_id, task in ir.task_map.tasks.items():
            optimized_primitives = []
            for prim in task.primitives:
                local_dev = prim.initiator
                local_mem = next(m for m in prim.memory_regions if isinstance(m, LocalMemory))
                remote_mem = next(m for m in prim.memory_regions if isinstance(m, RemoteMemory))
                remote_dev = remote_mem.device

                # 社区内优化：MST路径替代原始路径
                local_comm = next(c for c, ds in communities.items() if local_dev.device_id in ds)
                remote_comm = next(c for c, ds in communities.items() if remote_dev.device_id in ds)

                if local_comm == remote_comm:
                    mst = community_mst[local_comm]
                    if local_dev.device_id in mst and remote_dev.device_id in mst:
                        path = nx.shortest_path(mst, local_dev.device_id, remote_dev.device_id)
                        mst_bw = min(bw_graph[u][v] for u, v in zip(path[:-1], path[1:]))
                        mst_latency = sum(ir.cluster.get_optimal_path(devices[u], devices[v])["total_latency"] for u, v in zip(path[:-1], path[1:]))
                        new_prim = CommunicationPrimitive(local_dev, prim.op_type, [local_mem, remote_mem])
                    else:
                        new_prim = prim
                # 跨社区优化：选择最高带宽链路
                else:
                    cross_edges = sorted(
                        [(u, v) for u in communities[local_comm] for v in communities[remote_comm]],
                        key=lambda x: -bw_graph[x[0]][x[1]]
                    )
                    new_prim = prim
                    if cross_edges:
                        best_u, best_v = cross_edges[0]
                        new_remote_dev = devices[best_v]
                        new_remote_mem = RemoteMemory(new_remote_dev, remote_mem.address, remote_mem.size)
                        new_prim = CommunicationPrimitive(local_dev, prim.op_type, [local_mem, new_remote_mem])

                optimized_primitives.append(new_prim)

            # 保留任务依赖
            optimized_task = Task(task_id, optimized_primitives, task.dependencies, task.status)
            optimized_task_map.add_task(optimized_task)

        return CollectiveIR(ir.cluster, ir.collective_op, ir.data_size_gb, optimized_task_map)
    