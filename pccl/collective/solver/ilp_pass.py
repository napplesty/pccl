from ir.pipeline import Pass, Diagnostic, DiagnosticLevel
from ir import *
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, value
from typing import Dict, List


class ILPOptimizationPass(Pass):
    """ILP优化Pass：任务调度与设备分配优化"""
    def run(self, ir: CollectiveIR, diags: List[Diagnostic]) -> CollectiveIR:
        if not ir.task_map.tasks:
            diags.append(Diagnostic(DiagnosticLevel.WARNING, "TaskMap为空", loc="ILPOptimizationPass"))
            return ir

        # 1. 计算优化前关键路径时间
        pre_time = self._calculate_critical_path(ir)
        if pre_time == float("inf"):
            diags.append(Diagnostic(DiagnosticLevel.ERROR, "任务图非DAG", loc="ILPOptimizationPass"))
            return ir

        # 2. 运行ILP调度器
        scheduler = ILPScheduler(ir)
        schedule = scheduler.solve()
        if not schedule:
            diags.append(Diagnostic(DiagnosticLevel.ERROR, "ILP求解失败", loc="ILPOptimizationPass"))
            return ir

        # 3. 应用调度结果优化IR
        optimized_ir = self._apply_schedule(ir, schedule)
        post_time = self._calculate_critical_path(optimized_ir)

        # 4. 输出效率提升
        if pre_time > 1e-9:
            gain = ((pre_time - post_time) / pre_time) * 100
            diags.append(Diagnostic(DiagnosticLevel.NOTE, f"ILP优化: {pre_time:.4f}s → {post_time:.4f}s (提升{gain:.2f}%)", loc="ILPOptimizationPass"))
        else:
            diags.append(Diagnostic(DiagnosticLevel.WARNING, "优化前关键路径时间为0", loc="ILPOptimizationPass"))

        return optimized_ir

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

    def _apply_schedule(self, ir: CollectiveIR, schedule: Dict[int, Dict]) -> CollectiveIR:
        optimized_task_map = TaskMap(ir.collective_op, ir.data_size_gb)
        devices = ir.cluster.devices_by_id

        for task_id, task in ir.task_map.tasks.items():
            if task_id not in schedule:
                optimized_task = task
            else:
                sched = schedule[task_id]
                target_dev = devices[sched["device_id"]]
                optimized_primitives = []
                for prim in task.primitives:
                    # 将原语迁移到目标设备
                    if prim.initiator != target_dev:
                        new_local_mem = LocalMemory(target_dev, prim.memory_regions[0].address, prim.memory_regions[0].size)
                        new_remote_mem = prim.memory_regions[1]
                        new_prim = CommunicationPrimitive(target_dev, prim.op_type, [new_local_mem, new_remote_mem])
                        optimized_primitives.append(new_prim)
                    else:
                        optimized_primitives.append(prim)
                optimized_task = Task(task_id, optimized_primitives, task.dependencies, task.status)
            optimized_task_map.add_task(optimized_task)

        return CollectiveIR(ir.cluster, ir.collective_op, ir.data_size_gb, optimized_task_map)


class ILPScheduler:
    """ILP任务调度核心逻辑"""
    def __init__(self, ir: CollectiveIR):
        self.ir = ir
        self.tasks = list(ir.task_map.tasks.values())
        self.devices = list(ir.cluster.devices_by_id.values())
        self.durations = self._calculate_task_durations()
        self.model = LpProblem("TaskScheduling", LpMinimize)
        self._define_variables()
        self._add_constraints()
        self._set_objective()

    def _calculate_task_durations(self) -> Dict[Task, float]:
        durations = {}
        for task in self.tasks:
            task_time = max(self._primitive_time(p) for p in task.primitives) if task.primitives else 0.0
            durations[task] = task_time
        return durations

    def _primitive_time(self, prim: CommunicationPrimitive) -> float:
        local_mem = next(m for m in prim.memory_regions if isinstance(m, LocalMemory))
        remote_mem = next(m for m in prim.memory_regions if isinstance(m, RemoteMemory))
        path = self.ir.cluster.get_optimal_path(local_mem.device, remote_mem.device)
        return path["total_latency"] + (local_mem.size / path["min_bandwidth"])

    def _define_variables(self):
        self.start_time = LpVariable.dicts(
            "start_time",
            [(t.task_id, d.device_id) for t in self.tasks for d in self.devices],
            lowBound=0,
            cat=LpInteger
        )
        self.order = LpVariable.dicts(
            "order",
            [(t1.task_id, t2.task_id, d.device_id) for t1 in self.tasks for t2 in self.tasks if t1 != t2 for d in self.devices],
            cat=LpInteger,
            lowBound=0,
            upBound=1
        )

    def _add_constraints(self):
        # 1. 任务仅在原语发起设备上执行
        for task in self.tasks:
            for dev in self.devices:
                if dev not in [prim.initiator for prim in task.primitives]:
                    self.model += self.start_time[task.task_id, dev.device_id] >= 1e9

        # 2. 任务依赖约束
        for task in self.tasks:
            for dep in task.dependencies:
                for dev in self.devices:
                    self.model += self.start_time[task.task_id, dev.device_id] >= self.start_time[dep.task_id, dev.device_id] + self.durations[dep]

        # 3. 同一设备任务顺序约束
        for dev in self.devices:
            for t1 in self.tasks:
                for t2 in self.tasks:
                    if t1 != t2:
                        self.model += self.order[t1.task_id, t2.task_id, dev.device_id] + self.order[t2.task_id, t1.task_id, dev.device_id] == 1
                        self.model += self.start_time[t2.task_id, dev.device_id] >= self.start_time[t1.task_id, dev.device_id] + self.durations[t1] * self.order[t1.task_id, t2.task_id, dev.device_id]
                        self.model += self.start_time[t1.task_id, dev.device_id] >= self.start_time[t2.task_id, dev.device_id] + self.durations[t2] * self.order[t2.task_id, t1.task_id, dev.device_id]

    def _set_objective(self):
        self.makespan = LpVariable("makespan", lowBound=0)
        for task in self.tasks:
            for dev in self.devices:
                self.model += self.makespan >= self.start_time[task.task_id, dev.device_id] + self.durations[task]
        self.model += self.makespan

    def solve(self) -> Dict[int, Dict]:
        self.model.solve()
        if self.model.status != 1:
            return {}
        schedule = {}
        for task in self.tasks:
            for dev in self.devices:
                start = value(self.start_time[task.task_id, dev.device_id])
                if start < 1e9:
                    schedule[task.task_id] = {
                        "device_id": dev.device_id,
                        "start_time": start,
                        "duration": self.durations[task]
                    }
        return schedule
    