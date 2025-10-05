import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pccl.pipeline import CollectiveExecutionPipeline, PipelineConfig, ExecutionStrategy, OptimizationLevel
from pccl.collective_ir.specs.factories import create_simple_allreduce_ir

class TestPipelineConfig(unittest.TestCase):
    def test_config_creation(self):
        config = PipelineConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.optimization_level, OptimizationLevel.STANDARD)
        self.assertEqual(config.execution_strategy, ExecutionStrategy.PARALLEL)
        self.assertTrue(config.enable_verification)

    def test_config_custom_values(self):
        config = PipelineConfig(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            execution_strategy=ExecutionStrategy.SEQUENTIAL,
            enable_verification=False,
            enable_profiling=True,
            timeout_ms=60000
        )
        
        self.assertEqual(config.optimization_level, OptimizationLevel.AGGRESSIVE)
        self.assertEqual(config.execution_strategy, ExecutionStrategy.SEQUENTIAL)
        self.assertFalse(config.enable_verification)
        self.assertTrue(config.enable_profiling)
        self.assertEqual(config.timeout_ms, 60000)

    def test_config_to_dict(self):
        config = PipelineConfig(
            optimization_level=OptimizationLevel.ADVANCED,
            execution_strategy=ExecutionStrategy.PIPELINED
        )
        
        config_dict = config.to_dict()
        self.assertEqual(config_dict['optimization_level'], 2)
        self.assertEqual(config_dict['execution_strategy'], 'pipelined')
        self.assertTrue(config_dict['enable_verification'])

    def test_config_from_dict(self):
        config_dict = {
            'optimization_level': 3,
            'execution_strategy': 'sequential',
            'enable_verification': False,
            'enable_profiling': True,
            'timeout_ms': 45000
        }
        
        config = PipelineConfig.from_dict(config_dict)
        self.assertEqual(config.optimization_level, OptimizationLevel.AGGRESSIVE)
        self.assertEqual(config.execution_strategy, ExecutionStrategy.SEQUENTIAL)
        self.assertFalse(config.enable_verification)
        self.assertTrue(config.enable_profiling)
        self.assertEqual(config.timeout_ms, 45000)

class TestCollectiveExecutionPipeline(unittest.TestCase):
    def setUp(self):
        self.device_ids = [0, 1, 2, 3]
        self.data_size_gb = 0.1
        self.collective_ir = create_simple_allreduce_ir(self.device_ids, self.data_size_gb)

    def test_pipeline_creation(self):
        config = PipelineConfig(
            optimization_level=OptimizationLevel.STANDARD,
            execution_strategy=ExecutionStrategy.PARALLEL
        )
        pipeline = CollectiveExecutionPipeline(config)
        
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.config.optimization_level, OptimizationLevel.STANDARD)
        self.assertEqual(pipeline.config.execution_strategy, ExecutionStrategy.PARALLEL)
        self.assertFalse(pipeline.runtime_initialized)

    def test_pipeline_default_config(self):
        pipeline = CollectiveExecutionPipeline()
        self.assertIsNotNone(pipeline.config)
        self.assertEqual(pipeline.config.optimization_level, OptimizationLevel.STANDARD)

    def test_pipeline_execution(self):
        pipeline = CollectiveExecutionPipeline()
        result = pipeline.execute(self.collective_ir)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'execution_time_ms'))
        self.assertTrue(hasattr(result, 'results'))
        
        pipeline.shutdown()

    def test_pipeline_execution_stats(self):
        pipeline = CollectiveExecutionPipeline()
        stats = pipeline.get_execution_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('config', stats)
        self.assertIn('runtime_initialized', stats)
        self.assertIn('pass_manager_stats', stats)
        self.assertIn('execution_stats', stats)

    def test_pipeline_shutdown(self):
        pipeline = CollectiveExecutionPipeline()
        pipeline.shutdown()
        self.assertFalse(pipeline.runtime_initialized)

    def test_different_optimization_levels(self):
        for level in [OptimizationLevel.BASIC, OptimizationLevel.STANDARD, OptimizationLevel.ADVANCED]:
            config = PipelineConfig(optimization_level=level)
            pipeline = CollectiveExecutionPipeline(config)
            
            result = pipeline.execute(self.collective_ir)
            self.assertTrue(hasattr(result, 'success'))
            
            pipeline.shutdown()

    def test_different_execution_strategies(self):
        for strategy in [ExecutionStrategy.SEQUENTIAL, ExecutionStrategy.PARALLEL]:
            config = PipelineConfig(execution_strategy=strategy)
            pipeline = CollectiveExecutionPipeline(config)
            
            result = pipeline.execute(self.collective_ir)
            self.assertTrue(hasattr(result, 'success'))
            
            pipeline.shutdown()

    def test_pipeline_with_verification_disabled(self):
        config = PipelineConfig(enable_verification=False)
        pipeline = CollectiveExecutionPipeline(config)
        
        result = pipeline.execute(self.collective_ir)
        self.assertTrue(hasattr(result, 'success'))
        
        pipeline.shutdown()

    def test_pipeline_error_handling(self):
        from pccl.collective_ir.core.ir import CollectiveIR, ClusterMesh, TaskMap, CollectiveSpec
        from pccl.collective_ir.core.enums import CollectiveOpType
        
        invalid_ir = CollectiveIR(
            cluster=ClusterMesh(),
            collective_op=CollectiveOpType.ALLREDUCE,
            data_size_gb=1.0,
            task_map=TaskMap(CollectiveOpType.ALLREDUCE, 1.0, {}),
            collective_spec=CollectiveSpec(
                op_type=CollectiveOpType.ALLREDUCE,
                preconditions=[],
                postconditions=[],
                data_size_gb=1.0,
                involved_devices=set()
            )
        )
        
        pipeline = CollectiveExecutionPipeline()
        result = pipeline.execute(invalid_ir)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        
        pipeline.shutdown()

class TestExecutionResult(unittest.TestCase):
    def test_result_creation(self):
        from pccl.pipeline.execution_pipeline import ExecutionResult
        
        result = ExecutionResult(True, 100.0, {'key': 'value'})
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time_ms, 100.0)
        self.assertEqual(result.results, {'key': 'value'})
        self.assertIsNone(result.error)

    def test_result_with_error(self):
        from pccl.pipeline.execution_pipeline import ExecutionResult
        
        result = ExecutionResult(False, 50.0, error="Test error")
        self.assertFalse(result.success)
        self.assertEqual(result.execution_time_ms, 50.0)
        self.assertEqual(result.error, "Test error")

    def test_result_to_dict(self):
        from pccl.pipeline.execution_pipeline import ExecutionResult
        
        result = ExecutionResult(True, 75.5, {'data': [1, 2, 3]})
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['success'], True)
        self.assertEqual(result_dict['execution_time_ms'], 75.5)
        self.assertEqual(result_dict['results'], {'data': [1, 2, 3]})
        self.assertIsNone(result_dict['error'])

class TestPipelineFactoryFunctions(unittest.TestCase):
    def test_create_default_pipeline(self):
        from pccl.pipeline.execution_pipeline import create_default_pipeline
        
        pipeline = create_default_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.config.optimization_level, OptimizationLevel.STANDARD)
        self.assertEqual(pipeline.config.execution_strategy, ExecutionStrategy.PARALLEL)

    def test_create_high_performance_pipeline(self):
        from pccl.pipeline.execution_pipeline import create_high_performance_pipeline
        
        pipeline = create_high_performance_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.config.optimization_level, OptimizationLevel.AGGRESSIVE)
        self.assertEqual(pipeline.config.execution_strategy, ExecutionStrategy.PIPELINED)

if __name__ == '__main__':
    unittest.main()
