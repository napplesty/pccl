import unittest
import tempfile
import os
from pccl.runtime import Runtime, RuntimeConfig, Executor
from pccl.low_ir.core import LowLevelIR, PrimitiveConfig, BufferConfig
from pccl.low_ir.types import PrimitiveType, DataType, ExecutorType, TensorInfo


class TestRuntimeConfig(unittest.TestCase):
    """测试运行时配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RuntimeConfig()
        
        self.assertEqual(config.local_rank, 0)
        self.assertEqual(config.world_size, 1)
        self.assertEqual(config.buffers_per_executor, 8)
        self.assertEqual(len(config.default_buffer_sizes), 8)
        self.assertEqual(config.extra_config, {})
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = RuntimeConfig(
            local_rank=1,
            world_size=4,
            buffers_per_executor=16,
            default_buffer_sizes=[1024] * 16,
            extra_config={'debug': True, 'log_level': 'INFO'}
        )
        
        self.assertEqual(config.local_rank, 1)
        self.assertEqual(config.world_size, 4)
        self.assertEqual(config.buffers_per_executor, 16)
        self.assertEqual(len(config.default_buffer_sizes), 16)
        self.assertEqual(config.extra_config['debug'], True)
        self.assertEqual(config.extra_config['log_level'], 'INFO')
    
    def test_serialization(self):
        """测试配置序列化"""
        config = RuntimeConfig(
            local_rank=1,
            world_size=4,
            buffers_per_executor=16,
            default_buffer_sizes=[1024] * 16,
            extra_config={'debug': True}
        )
        
        # 序列化
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['local_rank'], 1)
        self.assertEqual(config_dict['world_size'], 4)
        self.assertEqual(config_dict['buffers_per_executor'], 16)
        self.assertEqual(len(config_dict['default_buffer_sizes']), 16)
        self.assertEqual(config_dict['extra_config']['debug'], True)
        
        # 反序列化
        config2 = RuntimeConfig.from_dict(config_dict)
        
        self.assertEqual(config2.local_rank, 1)
        self.assertEqual(config2.world_size, 4)
        self.assertEqual(config2.buffers_per_executor, 16)
        self.assertEqual(len(config2.default_buffer_sizes), 16)
        self.assertEqual(config2.extra_config['debug'], True)


class TestRuntime(unittest.TestCase):
    """测试运行时"""
    
    def setUp(self):
        """测试前准备"""
        self.runtime = Runtime()
        self.config = RuntimeConfig(world_size=2)
        
        # 创建测试图
        self.test_graph = LowLevelIR("test_graph")
        buffer_id = self.test_graph.add_buffer(DataType.F32, 1024, ExecutorType.CUDA)
        op_config = PrimitiveConfig(PrimitiveType.COPY, DataType.F32)
        self.test_graph.add_operator(op_config)
        tensor_info = TensorInfo(shape=(256,), dtype=DataType.F32, device_type=ExecutorType.CUDA)
        self.test_graph.add_input_tensor(tensor_info)
        self.test_graph.add_output_tensor(tensor_info)
    
    def test_initialization(self):
        """测试运行时初始化"""
        self.assertFalse(self.runtime.is_initialized)
        
        self.runtime.initialize(self.config)
        
        self.assertTrue(self.runtime.is_initialized)
        self.assertEqual(self.runtime.config.world_size, 2)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with Runtime() as runtime:
            self.assertTrue(runtime.is_initialized)
        
        # 离开上下文后应该自动关闭
        self.assertFalse(runtime.is_initialized)
    
    def test_graph_registration(self):
        """测试图注册"""
        self.runtime.initialize(self.config)
        
        # 注册图
        self.runtime.register_graph(self.test_graph)
        
        # 验证图已注册
        self.assertIn("test_graph", self.runtime.list_graphs())
        
        # 获取注册的图
        graph = self.runtime.get_graph("test_graph")
        self.assertEqual(graph.name, "test_graph")
    
    def test_graph_execution(self):
        """测试图执行"""
        self.runtime.initialize(self.config)
        self.runtime.register_graph(self.test_graph)
        
        # 执行图
        result = self.runtime.execute_graph("test_graph")
        
        # 验证执行结果
        self.assertTrue(result['success'])
        self.assertIn('execution_time', result)
        self.assertIn('participants', result)
        self.assertIn('operations_executed', result)
        self.assertIn('execution_log', result)
        
        # 验证图被标记为活跃
        self.assertIn("test_graph", self.runtime.active_graphs)
    
    def test_graph_execution_with_participants(self):
        """测试带参与者的图执行"""
        self.runtime.initialize(RuntimeConfig(world_size=4))
        self.runtime.register_graph(self.test_graph)
        
        # 指定参与者
        participants = [0, 2]
        result = self.runtime.execute_graph("test_graph", participants)
        
        # 验证参与者正确设置
        self.assertEqual(result['participants'], participants)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 未初始化时执行图应该报错
        with self.assertRaises(RuntimeError):
            self.runtime.execute_graph("test_graph")
        
        # 注册未初始化的图应该报错
        with self.assertRaises(RuntimeError):
            self.runtime.register_graph(self.test_graph)
        
        self.runtime.initialize(self.config)
        
        # 执行未注册的图应该报错
        with self.assertRaises(KeyError):
            self.runtime.execute_graph("nonexistent_graph")
    
    def test_shutdown(self):
        """测试关闭运行时"""
        self.runtime.initialize(self.config)
        self.runtime.register_graph(self.test_graph)
        self.runtime.execute_graph("test_graph")
        
        # 验证运行时已初始化且有活跃图
        self.assertTrue(self.runtime.is_initialized)
        self.assertEqual(len(self.runtime.active_graphs), 1)
        
        # 关闭运行时
        self.runtime.shutdown()
        
        # 验证运行时已关闭
        self.assertFalse(self.runtime.is_initialized)
        self.assertEqual(len(self.runtime.active_graphs), 0)


class TestExecutor(unittest.TestCase):
    """测试执行器"""
    
    def setUp(self):
        """测试前准备"""
        self.runtime = Runtime()
        self.runtime.initialize(RuntimeConfig(world_size=2))
        
        self.executor = Executor(self.runtime)
        
        # 创建测试图
        self.test_graph = LowLevelIR("test_graph")
        buffer_id = self.test_graph.add_buffer(DataType.F32, 1024, ExecutorType.CUDA)
        op_config = PrimitiveConfig(PrimitiveType.COPY, DataType.F32)
        self.test_graph.add_operator(op_config)
        tensor_info = TensorInfo(shape=(256,), dtype=DataType.F32, device_type=ExecutorType.CUDA)
        self.test_graph.add_input_tensor(tensor_info)
        self.test_graph.add_output_tensor(tensor_info)
        
        # 注册图到运行时
        self.runtime.register_graph(self.test_graph)
    
    def test_initialization(self):
        """测试执行器初始化"""
        self.assertEqual(self.executor.execution_count, 0)
        self.assertEqual(len(self.executor.execution_history), 0)
        self.assertIsNone(self.executor.current_graph)
    
    def test_load_graph_by_name(self):
        """测试通过名称加载图"""
        self.executor.load_graph("test_graph")
        
        self.assertIsNotNone(self.executor.current_graph)
        self.assertEqual(self.executor.current_graph.name, "test_graph")
    
    def test_load_graph_by_instance(self):
        """测试通过实例加载图"""
        self.executor.load_graph(self.test_graph)
        
        self.assertIsNotNone(self.executor.current_graph)
        self.assertEqual(self.executor.current_graph.name, "test_graph")
    
    def test_execute_graph(self):
        """测试执行图"""
        self.executor.load_graph("test_graph")
        
        # 执行图
        result = self.executor.execute()
        
        # 验证执行结果
        self.assertTrue(result['success'])
        
        # 验证执行历史
        self.assertEqual(self.executor.execution_count, 1)
        self.assertEqual(len(self.executor.execution_history), 1)
        
        history_record = self.executor.execution_history[0]
        self.assertEqual(history_record['graph_name'], "test_graph")
        self.assertIn('timestamp', history_record)
        self.assertEqual(history_record['input_tensors_count'], 0)  # 没有提供输入张量
        self.assertEqual(history_record['output_tensors_count'], 0)  # 没有提供输出张量
    
    def test_execute_with_tensors(self):
        """测试使用张量执行图"""
        self.executor.load_graph("test_graph")
        
        # 模拟输入输出张量
        input_tensors = [None]  # 模拟一个输入张量
        output_tensors = [None]  # 模拟一个输出张量
        
        # 执行图
        result = self.executor.execute(input_tensors, output_tensors)
        
        # 验证执行历史
        history_record = self.executor.execution_history[0]
        self.assertEqual(history_record['input_tensors_count'], 1)
        self.assertEqual(history_record['output_tensors_count'], 1)
    
    def test_execute_with_tensors_validation(self):
        """测试张量验证"""
        self.executor.load_graph("test_graph")
        
        # 测试输入张量数量不匹配
        input_tensors = [None, None]  # 期望1个，实际2个
        with self.assertRaises(ValueError):
            self.executor.execute(input_tensors)
        
        # 测试输出张量数量不匹配
        output_tensors = [None, None]  # 期望1个，实际2个
        with self.assertRaises(ValueError):
            self.executor.execute(None, output_tensors)
    
    def test_execute_with_tensors_convenience(self):
        """测试便捷执行方法"""
        # 使用便捷方法执行
        result = self.executor.execute_with_tensors(
            "test_graph", 
            input_tensors=[None], 
            output_tensors=[None]
        )
        
        # 验证图已加载
        self.assertIsNotNone(self.executor.current_graph)
        
        # 验证执行结果
        self.assertTrue(result['success'])
        self.assertEqual(self.executor.execution_count, 1)
    
    def test_profile_execution(self):
        """测试性能分析"""
        # 性能分析
        profile_result = self.executor.profile("test_graph", iterations=3)
        
        # 验证性能分析结果
        self.assertEqual(profile_result['iterations'], 3)
        self.assertIn('min_time', profile_result)
        self.assertIn('max_time', profile_result)
        self.assertIn('avg_time', profile_result)
        self.assertIn('execution_times', profile_result)
        self.assertIn('throughput', profile_result)
        
        self.assertEqual(len(profile_result['execution_times']), 3)
    
    def test_export_execution_history(self):
        """测试导出执行历史"""
        # 执行几次图
        self.executor.load_graph("test_graph")
        self.executor.execute()
        self.executor.execute()
        
        # 导出执行历史
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.executor.export_execution_history(temp_file)
            
            # 验证文件存在且不为空
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 0)
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_clear_history(self):
        """测试清空执行历史"""
        # 执行几次图
        self.executor.load_graph("test_graph")
        self.executor.execute()
        self.executor.execute()
        
        # 验证执行历史
        self.assertEqual(self.executor.execution_count, 2)
        
        # 清空历史
        self.executor.clear_history()
        
        # 验证历史已清空
        self.assertEqual(self.executor.execution_count, 0)
        self.assertEqual(len(self.executor.execution_history), 0)
    
    def test_string_representation(self):
        """测试字符串表示"""
        # 未加载图时的字符串表示
        self.assertEqual(
            str(self.executor),
            "Executor(当前图: 无, 执行次数: 0)"
        )
        
        # 加载图后的字符串表示
        self.executor.load_graph("test_graph")
        self.assertEqual(
            str(self.executor),
            "Executor(当前图: test_graph, 执行次数: 0)"
        )
        
        # 执行后的字符串表示
        self.executor.execute()
        self.assertEqual(
            str(self.executor),
            "Executor(当前图: test_graph, 执行次数: 1)"
        )


if __name__ == '__main__':
    unittest.main()
