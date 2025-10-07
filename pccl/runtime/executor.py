from typing import List, Dict, Optional, Any, Union
import logging
import json

from ..low_ir.core import LowLevelIR
from .core import Runtime

logger = logging.getLogger(__name__)

class Executor:
    """执行器，封装图执行逻辑"""
    
    def __init__(self, runtime: Runtime):
        """初始化执行器
        
        Args:
            runtime: 运行时实例
        """
        if not runtime.is_initialized:
            raise RuntimeError("Runtime未初始化")
        
        self._runtime = runtime
        self._current_graph: Optional[LowLevelIR] = None
        self._execution_history: List[Dict[str, Any]] = []
    
    def load_graph(self, graph: Union[LowLevelIR, str]):
        """加载图
        
        Args:
            graph: LowLevelIR实例或已注册的图名称
        """
        if isinstance(graph, str):
            # 从运行时获取已注册的图
            self._current_graph = self._runtime.get_graph(graph)
        else:
            # 直接使用LowLevelIR实例
            self._current_graph = graph
        
        logger.info(f"加载图: {self._current_graph.name}")
    
    def execute(self, input_tensors: List[Any] = None, output_tensors: List[Any] = None, 
                participants: List[int] = None) -> Dict[str, Any]:
        """执行当前加载的图
        
        Args:
            input_tensors: 输入张量列表，如果为None则使用模拟张量
            output_tensors: 输出张量列表，如果为None则创建模拟张量
            participants: 参与执行的执行器列表
            
        Returns:
            执行结果
        """
        if self._current_graph is None:
            raise RuntimeError("未加载任何图")
        
        # 验证图
        if not self._current_graph.validate():
            raise ValueError("图验证失败")
        
        # 验证输入输出张量
        self._validate_tensors(input_tensors, output_tensors)
        
        # 执行图
        result = self._runtime.execute_graph(self._current_graph.name, participants)
        
        # 记录执行历史
        execution_record = {
            'graph_name': self._current_graph.name,
            'timestamp': self._get_timestamp(),
            'input_tensors_count': len(input_tensors) if input_tensors else 0,
            'output_tensors_count': len(output_tensors) if output_tensors else 0,
            'participants': participants or list(range(self._runtime.config.world_size)),
            'result': result
        }
        self._execution_history.append(execution_record)
        
        logger.info(f"图执行完成: {self._current_graph.name}")
        return result
    
    def _validate_tensors(self, input_tensors: List[Any], output_tensors: List[Any]):
        """验证输入输出张量
        
        注意：由于用户要求暂时不修改C++代码，这里只进行基本的验证
        未来集成PyTorch后，这里应该验证张量形状、数据类型和设备
        """
        if input_tensors is not None:
            expected_inputs = len(self._current_graph.input_tensors)
            actual_inputs = len(input_tensors)
            if actual_inputs != expected_inputs:
                raise ValueError(f"输入张量数量不匹配: 期望{expected_inputs}, 实际{actual_inputs}")
        
        if output_tensors is not None:
            expected_outputs = len(self._current_graph.output_tensors)
            actual_outputs = len(output_tensors)
            if actual_outputs != expected_outputs:
                raise ValueError(f"输出张量数量不匹配: 期望{expected_outputs}, 实际{actual_outputs}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def execute_with_tensors(self, graph: Union[LowLevelIR, str], 
                           input_tensors: List[Any], 
                           output_tensors: List[Any] = None,
                           participants: List[int] = None) -> Dict[str, Any]:
        """加载并执行图（便捷方法）
        
        Args:
            graph: LowLevelIR实例或已注册的图名称
            input_tensors: 输入张量列表
            output_tensors: 输出张量列表，如果为None则创建输出张量
            participants: 参与执行的执行器列表
            
        Returns:
            执行结果
        """
        self.load_graph(graph)
        return self.execute(input_tensors, output_tensors, participants)
    
    def profile(self, graph: Union[LowLevelIR, str], iterations: int = 10,
               input_tensors: List[Any] = None, participants: List[int] = None) -> Dict[str, Any]:
        """性能分析
        
        Args:
            graph: LowLevelIR实例或已注册的图名称
            iterations: 迭代次数
            input_tensors: 输入张量列表
            participants: 参与执行的执行器列表
            
        Returns:
            性能分析结果
        """
        import time
        
        self.load_graph(graph)
        
        # 准备输出张量
        output_tensors = None
        
        # 预热
        logger.info("预热执行...")
        self.execute(input_tensors, output_tensors, participants)
        
        # 性能测试
        logger.info(f"开始性能测试，迭代次数: {iterations}")
        execution_times = []
        
        for i in range(iterations):
            start_time = time.time()
            result = self.execute(input_tensors, output_tensors, participants)
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            logger.debug(f"迭代 {i+1}/{iterations}: {execution_time:.6f}秒")
        
        # 计算统计信息
        min_time = min(execution_times)
        max_time = max(execution_times)
        avg_time = sum(execution_times) / len(execution_times)
        
        profile_result = {
            'iterations': iterations,
            'min_time': min_time,
            'max_time': max_time,
            'avg_time': avg_time,
            'execution_times': execution_times,
            'throughput': len(self._current_graph.operators) / avg_time if avg_time > 0 else 0
        }
        
        logger.info(f"性能测试完成: 平均时间={avg_time:.6f}秒, 吞吐量={profile_result['throughput']:.2f} ops/秒")
        return profile_result
    
    def export_execution_history(self, filepath: str):
        """导出执行历史到文件
        
        Args:
            filepath: 输出文件路径
        """
        with open(filepath, 'w') as f:
            json.dump(self._execution_history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"执行历史已导出到: {filepath}")
    
    def clear_history(self):
        """清空执行历史"""
        self._execution_history.clear()
        logger.info("执行历史已清空")
    
    @property
    def current_graph(self) -> Optional[LowLevelIR]:
        """返回当前加载的图"""
        return self._current_graph
    
    @property
    def execution_count(self) -> int:
        """返回执行次数"""
        return len(self._execution_history)
    
    @property
    def execution_history(self) -> List[Dict[str, Any]]:
        """返回执行历史"""
        return self._execution_history.copy()
    
    def __str__(self) -> str:
        """返回执行器状态字符串"""
        status = f"Executor(当前图: {self._current_graph.name if self._current_graph else '无'}, "
        status += f"执行次数: {self.execution_count})"
        return status
