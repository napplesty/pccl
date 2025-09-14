import networkx as nx
from .passes import Pass, Diagnostic, DiagnosticLevel
from ir import CollectiveIR, Task, TaskMap, LocalMemory
from typing import List

class IRValidationPass(Pass):
    def run(self, ir: CollectiveIR, diagnostics: List[Diagnostic]) -> CollectiveIR:
        try:
            CollectiveIR(
                cluster=ir.cluster,
                collective_op=ir.collective_op,
                data_size_gb=ir.data_size_gb,
                task_map=ir.task_map
            )
        except ValueError as e:
            diagnostics.append(Diagnostic(DiagnosticLevel.ERROR, str(e), location="CollectiveIR"))

        try:
            self._check_task_dag(ir.task_map, diagnostics)
        except Exception as e:
            diagnostics.append(Diagnostic(DiagnosticLevel.ERROR, str(e), location="TaskMap"))

        self._check_primitive_local_memory(ir.task_map, diagnostics)

        self._check_device_memory(ir, diagnostics)

        return ir

    def _check_task_dag(self, task_map: TaskMap, diagnostics: List[Diagnostic]):
        G = nx.DiGraph()
        for task in task_map.tasks.values():
            G.add_node(task.task_id)
            for dep in task.dependencies:
                G.add_edge(dep.task_id, task.task_id)
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            cycle_str = " → ".join(map(str, cycles[0]))
            diagnostics.append(Diagnostic(
                DiagnosticLevel.ERROR,
                f"Task graph has cycle: {cycle_str}",
                location="TaskMap.dependencies"
            ))

    def _check_primitive_local_memory(self, task_map: TaskMap, diagnostics: List[Diagnostic]):
        for task in task_map.tasks.values():
            for prim in task.primitives:
                if not any(isinstance(reg, LocalMemory) for reg in prim.memory_regions):
                    diagnostics.append(Diagnostic(
                        DiagnosticLevel.ERROR,
                        f"Primitive in task {task.task_id} has no local memory",
                        location=f"Task {task.task_id}.Primitive"
                    ))

    def _check_device_memory(self, ir: CollectiveIR, diagnostics: List[Diagnostic]):
        for dev in ir.cluster.devices_by_id.values():
            mem_size = dev.properties.get("memory", 0.0)  # 假设内存以GB为单位
            for task in ir.task_map.tasks.values():
                for prim in task.primitives:
                    if prim.initiator != dev:
                        continue
                    for reg in prim.memory_regions:
                        if isinstance(reg, LocalMemory) and reg.size > mem_size:
                            diagnostics.append(Diagnostic(
                                DiagnosticLevel.WARNING,
                                f"Device {dev.device_id} has insufficient memory for local region (needs {reg.size} GB, has {mem_size} GB)",
                                location=f"Device {dev.device_id}.LocalMemory"
                            ))