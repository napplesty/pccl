from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
import json

from ..low_ir.core import LowLevelIR
from ..low_ir.types import ExecutorType

logger = logging.getLogger(__name__)

@dataclass
class RuntimeConfig:
    """运行时配置"""
    local_rank: int = 0
    world_size: int = 1
    buffers_per_executor: int = 8
    default_buffer_sizes: List[int] = field(default_factory=lambda: [1024 * 1024] * 8)
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'local_rank': self.local_rank,
            'world_size': self.world_size,
            'buffers_per_executor': self.buffers_per_executor,
            'default_buffer_sizes': self.default_buffer_sizes,
            'extra_config': self.extra_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuntimeConfig':
        """从字典创建RuntimeConfig"""
        return cls(
            local_rank=data.get('local_rank', 0),
            world_size=data.get('world_size', 1),
            buffers_per_executor=data.get('buffers_per_executor', 8),
            default_buffer_sizes=data.get('default_buffer_sizes', [1024 * 1024] * 8),
            extra_config=data.get('extra_config', {})
        )

class Runtime:
    """PCCL运行时管理器"""
    
    def __init__(self):
        self._initialized = False
        self._config: Optional[RuntimeConfig] = None
        self._executors: Dict[int, Any] = {}  # 执行器实例
        self._graphs: Dict[str, LowLevelIR] = {}  # 注册的图
        self._active_graphs: List[str] = []  # 活跃的图
    
    def initialize(self, config: Optional[RuntimeConfig] = None):
        """初始化运行时
        
        Args:
            config: 运行时配置，如果为None则使用默认配置
        """
        if self._initialized:
            logger.warning("Runtime已经初始化")
            return
        
        self._config = config or RuntimeConfig()
        logger.info(f"初始化Runtime: rank={self._config.local_rank}, world_size={self._config.world_size}")
        
        # 这里应该调用C++后端的initializeRuntime函数
        # 但由于用户要求暂时不修改C++代码，我们只做Python端的初始化
        self._initialized = True
        
        # 创建执行器
        self._create_executors()
        
        logger.info("Runtime初始化完成")
    
    def _create_executors(self):
        """创建执行器实例"""
        # 这里应该根据配置创建对应的执行器
        # 暂时创建虚拟执行器
        for i in range(self._config.world_size):
            self._executors[i] = {
                'type': ExecutorType.CUDA if i < 4 else ExecutorType.CPU,  # 模拟GPU和CPU执行器
                'available': True
            }
        
        logger.info(f"创建了 {len(self._executors)} 个执行器")
    
    def register_graph(self, graph: LowLevelIR, name: Optional[str] = None):
        """注册图到运行时
        
        Args:
            graph: 低级IR图
            name: 图名称，如果为None则使用graph.name
        """
        if not self._initialized:
            raise RuntimeError("Runtime未初始化")
        
        graph_name = name or graph.name
        if not graph_name:
            raise ValueError("图名称不能为空")
        
        self._graphs[graph_name] = graph
        logger.info(f"注册图: {graph_name}")
    
    def get_graph(self, name: str) -> LowLevelIR:
        """获取注册的图
        
        Args:
            name: 图名称
            
        Returns:
            低级IR图
        """
        if name not in self._graphs:
            raise KeyError(f"图 '{name}' 未注册")
        return self._graphs[name]
    
    def list_graphs(self) -> List[str]:
        """列出所有注册的图名称"""
        return list(self._graphs.keys())
    
    def execute_graph(self, graph_name: str, participants: List[int] = None):
        """执行图
        
        Args:
            graph_name: 图名称
            participants: 参与执行的执行器列表，如果为None则使用所有执行器
            
        Returns:
            执行结果
        """
        if not self._initialized:
            raise RuntimeError("Runtime未初始化")
        
        if graph_name not in self._graphs:
            raise KeyError(f"图 '{graph_name}' 未注册")
        
        graph = self._graphs[graph_name]
        
        # 验证图
        if not graph.validate():
            raise ValueError(f"图 '{graph_name}' 验证失败")
        
        # 选择执行器
        if participants is None:
            participants = list(range(self._config.world_size))
        
        logger.info(f"执行图: {graph_name}, 参与者: {participants}")
        
        # 这里应该调用C++后端的executeGraph函数
        # 但由于用户要求暂时不修改C++代码，我们只做Python端的模拟执行
        result = self._simulate_execution(graph, participants)
        
        self._active_graphs.append(graph_name)
        logger.info(f"图执行完成: {graph_name}")
        
        return result
    
    def _simulate_execution(self, graph: LowLevelIR, participants: List[int]) -> Dict[str, Any]:
        """模拟图执行（用于测试）"""
        # 模拟执行过程
        execution_log = []
        
        # 获取拓扑排序的操作序列
        # 这里简化处理，按操作ID顺序执行
        for op_id, op_config in graph.operators.items():
            execution_log.append({
                'op_id': op_id,
                'type': op_config.type.name,
                'src_buffer': op_config.src_buffer_idx,
                'dst_buffer': op_config.dst_buffer_idx,
                'data_size': op_config.data_size
            })
        
        return {
            'success': True,
            'execution_time': 0.1,  # 模拟执行时间
            'participants': participants,
            'operations_executed': len(execution_log),
            'execution_log': execution_log
        }
    
    def shutdown(self):
        """关闭运行时"""
        if not self._initialized:
            return
        
        logger.info("关闭Runtime")
        
        # 清理活跃的图
        self._active_graphs.clear()
        
        # 清理执行器
        self._executors.clear()
        
        # 这里应该调用C++后端的shutdownRuntime函数
        self._initialized = False
        
        logger.info("Runtime关闭完成")
    
    @property
    def is_initialized(self) -> bool:
        """返回运行时是否已初始化"""
        return self._initialized
    
    @property
    def config(self) -> Optional[RuntimeConfig]:
        """返回运行时配置"""
        return self._config
    
    @property
    def active_graphs(self) -> List[str]:
        """返回活跃的图列表"""
        return self._active_graphs.copy()
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
