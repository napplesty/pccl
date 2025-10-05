import unittest
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pccl.primitive_ir.primitive_ir import (
    PrimitiveGraph, PrimitiveType, DataType, ComputeType, ExecutorType,
    BufferConfig, ExecutorConfig, PrimitiveConfig
)

class TestPrimitiveIR(unittest.TestCase):
    def test_primitive_type_enum(self):
        self.assertEqual(PrimitiveType.COPY.value, 1)
        self.assertEqual(PrimitiveType.COMPUTE.value, 2)
        self.assertEqual(PrimitiveType.WRITE.value, 3)

    def test_data_type_enum(self):
        self.assertEqual(DataType.F32.value, 1)
        self.assertEqual(DataType.F16.value, 2)
        self.assertEqual(DataType.BF16.value, 3)

    def test_compute_type_enum(self):
        self.assertEqual(ComputeType.SUM.value, 1)
        self.assertEqual(ComputeType.MAX.value, 2)
        self.assertEqual(ComputeType.MIN.value, 3)

    def test_executor_type_enum(self):
        self.assertEqual(ExecutorType.CPU.value, 1)
        self.assertEqual(ExecutorType.CUDA.value, 2)

class TestBufferConfig(unittest.TestCase):
    def test_buffer_config_creation(self):
        buffer_config = BufferConfig(0, DataType.F32, 1024, ExecutorType.CUDA)
        
        self.assertEqual(buffer_config.buffer_idx, 0)
        self.assertEqual(buffer_config.dtype, DataType.F32)
        self.assertEqual(buffer_config.size, 1024)
        self.assertEqual(buffer_config.executor_type, ExecutorType.CUDA)

    def test_buffer_config_to_dict(self):
        buffer_config = BufferConfig(1, DataType.F16, 2048, ExecutorType.CPU)
        config_dict = buffer_config.to_dict()
        
        self.assertEqual(config_dict['buffer_idx'], 1)
        self.assertEqual(config_dict['dtype'], 'F16')
        self.assertEqual(config_dict['size'], 2048)
        self.assertEqual(config_dict['executor_type'], 'CPU')

    def test_buffer_config_from_dict(self):
        config_dict = {
            'buffer_idx': 2,
            'dtype': 'BF16',
            'size': 4096,
            'executor_type': 'CUDA'
        }
        
        buffer_config = BufferConfig.from_dict(config_dict)
        self.assertEqual(buffer_config.buffer_idx, 2)
        self.assertEqual(buffer_config.dtype, DataType.BF16)
        self.assertEqual(buffer_config.size, 4096)
        self.assertEqual(buffer_config.executor_type, ExecutorType.CUDA)

class TestExecutorConfig(unittest.TestCase):
    def test_executor_config_creation(self):
        executor_config = ExecutorConfig(ExecutorType.CUDA, 4)
        
        self.assertEqual(executor_config.executor_type, ExecutorType.CUDA)
        self.assertEqual(executor_config.num_total_executors, 4)

    def test_executor_config_to_dict(self):
        executor_config = ExecutorConfig(ExecutorType.CPU, 8)
        config_dict = executor_config.to_dict()
        
        self.assertEqual(config_dict['executor_type'], 'CPU')
        self.assertEqual(config_dict['num_total_executors'], 8)

    def test_executor_config_from_dict(self):
        config_dict = {
            'executor_type': 'CUDA',
            'num_total_executors': 16
        }
        
        executor_config = ExecutorConfig.from_dict(config_dict)
        self.assertEqual(executor_config.executor_type, ExecutorType.CUDA)
        self.assertEqual(executor_config.num_total_executors, 16)

class TestPrimitiveConfig(unittest.TestCase):
    def test_primitive_config_creation(self):
        primitive_config = PrimitiveConfig(
            type=PrimitiveType.COPY,
            dtype=DataType.F32,
            target_rank=0,
            src_buffer_idx=0,
            dst_buffer_idx=1,
            compute_op=ComputeType.SUM,
            executor_type=ExecutorType.CUDA,
            num_executors=1,
            data_size=1024,
            signal_value=0,
            num_dependencies=0
        )
        
        self.assertEqual(primitive_config.type, PrimitiveType.COPY)
        self.assertEqual(primitive_config.dtype, DataType.F32)
        self.assertEqual(primitive_config.target_rank, 0)
        self.assertEqual(primitive_config.src_buffer_idx, 0)
        self.assertEqual(primitive_config.dst_buffer_idx, 1)
        self.assertEqual(primitive_config.compute_op, ComputeType.SUM)
        self.assertEqual(primitive_config.executor_type, ExecutorType.CUDA)
        self.assertEqual(primitive_config.num_executors, 1)
        self.assertEqual(primitive_config.data_size, 1024)
        self.assertEqual(primitive_config.signal_value, 0)
        self.assertEqual(primitive_config.num_dependencies, 0)

    def test_primitive_config_to_dict(self):
        primitive_config = PrimitiveConfig(
            type=PrimitiveType.COMPUTE,
            dtype=DataType.F16,
            target_rank=1,
            src_buffer_idx=2,
            dst_buffer_idx=3,
            compute_op=ComputeType.MAX,
            executor_type=ExecutorType.CPU,
            num_executors=2,
            data_size=2048,
            signal_value=1,
            num_dependencies=1,
            followers=[4, 5],
            num_followers=2
        )
        
        config_dict = primitive_config.to_dict()
        self.assertEqual(config_dict['type'], 'COMPUTE')
        self.assertEqual(config_dict['dtype'], 'F16')
        self.assertEqual(config_dict['target_rank'], 1)
        self.assertEqual(config_dict['src_buffer_idx'], 2)
        self.assertEqual(config_dict['dst_buffer_idx'], 3)
        self.assertEqual(config_dict['compute_op'], 'MAX')
        self.assertEqual(config_dict['executor_type'], 'CPU')
        self.assertEqual(config_dict['num_executors'], 2)
        self.assertEqual(config_dict['data_size'], 2048)
        self.assertEqual(config_dict['signal_value'], 1)
        self.assertEqual(config_dict['num_dependencies'], 1)
        self.assertEqual(config_dict['followers'], [4, 5])
        self.assertEqual(config_dict['num_followers'], 2)

    def test_primitive_config_from_dict(self):
        config_dict = {
            'type': 'WRITE',
            'dtype': 'BF16',
            'target_rank': 2,
            'src_buffer_idx': 3,
            'dst_buffer_idx': 4,
            'compute_op': 'MIN',
            'executor_type': 'CUDA',
            'num_executors': 4,
            'data_size': 4096,
            'signal_value': 2,
            'num_dependencies': 2,
            'followers': [5, 6],
            'num_followers': 2
        }
        
        primitive_config = PrimitiveConfig.from_dict(config_dict)
        self.assertEqual(primitive_config.type, PrimitiveType.WRITE)
        self.assertEqual(primitive_config.dtype, DataType.BF16)
        self.assertEqual(primitive_config.target_rank, 2)
        self.assertEqual(primitive_config.src_buffer_idx, 3)
        self.assertEqual(primitive_config.dst_buffer_idx, 4)
        self.assertEqual(primitive_config.compute_op, ComputeType.MIN)
        self.assertEqual(primitive_config.executor_type, ExecutorType.CUDA)
        self.assertEqual(primitive_config.num_executors, 4)
        self.assertEqual(primitive_config.data_size, 4096)
        self.assertEqual(primitive_config.signal_value, 2)
        self.assertEqual(primitive_config.num_dependencies, 2)
        self.assertEqual(primitive_config.followers, [5, 6])
        self.assertEqual(primitive_config.num_followers, 2)

class TestPrimitiveGraph(unittest.TestCase):
    def setUp(self):
        self.graph = PrimitiveGraph(rank=0)

    def test_primitive_graph_creation(self):
        self.assertEqual(self.graph.rank, 0)
        self.assertEqual(len(self.graph.buffers), 0)
        self.assertEqual(len(self.graph.executors), 0)
        self.assertEqual(len(self.graph.operators), 0)
        self.assertEqual(len(self.graph.dependencies), 0)

    def test_add_buffer(self):
        buffer_config = self.graph.add_buffer(0, DataType.F32, 1024, ExecutorType.CUDA)
        
        self.assertEqual(len(self.graph.buffers), 1)
        self.assertEqual(buffer_config.buffer_idx, 0)
        self.assertEqual(buffer_config.dtype, DataType.F32)
        self.assertEqual(buffer_config.size, 1024)
        self.assertEqual(buffer_config.executor_type, ExecutorType.CUDA)

    def test_add_operator(self):
        primitive_config = PrimitiveConfig(
            type=PrimitiveType.COPY,
            dtype=DataType.F32,
            target_rank=0,
            src_buffer_idx=0,
            dst_buffer_idx=1,
            compute_op=ComputeType.SUM,
            executor_type=ExecutorType.CUDA,
            num_executors=1,
            data_size=1024,
            signal_value=0,
            num_dependencies=0
        )
        
        operator_id = self.graph.add_operator(primitive_config)
        self.assertEqual(operator_id, 0)
        self.assertEqual(len(self.graph.operators), 1)

    def test_add_dependency(self):
        primitive_config1 = PrimitiveConfig(
            type=PrimitiveType.COPY,
            dtype=DataType.F32,
            target_rank=0,
            src_buffer_idx=0,
            dst_buffer_idx=1,
            compute_op=ComputeType.SUM,
            executor_type=ExecutorType.CUDA,
            num_executors=1,
            data_size=1024,
            signal_value=0,
            num_dependencies=0
        )
        
        primitive_config2 = PrimitiveConfig(
            type=PrimitiveType.COMPUTE,
            dtype=DataType.F32,
            target_rank=0,
            src_buffer_idx=1,
            dst_buffer_idx=2,
            compute_op=ComputeType.SUM,
            executor_type=ExecutorType.CUDA,
            num_executors=1,
            data_size=1024,
            signal_value=0,
            num_dependencies=0
        )
        
        op1_id = self.graph.add_operator(primitive_config1)
        op2_id = self.graph.add_operator(primitive_config2)
        
        self.graph.add_dependency(op1_id, op2_id)
        self.assertEqual(len(self.graph.dependencies), 1)
        self.assertEqual(self.graph.dependencies[0], (op1_id, op2_id))

    def test_primitive_graph_validation(self):
        self.assertTrue(self.graph.validate())
        
        self.graph.add_buffer(0, DataType.F32, 1024, ExecutorType.CUDA)
        self.assertTrue(self.graph.validate())

    def test_primitive_graph_to_dict(self):
        self.graph.add_buffer(0, DataType.F32, 1024, ExecutorType.CUDA)
        
        primitive_config = PrimitiveConfig(
            type=PrimitiveType.COPY,
            dtype=DataType.F32,
            target_rank=0,
            src_buffer_idx=0,
            dst_buffer_idx=0,
            compute_op=ComputeType.SUM,
            executor_type=ExecutorType.CUDA,
            num_executors=1,
            data_size=1024,
            signal_value=0,
            num_dependencies=0
        )
        
        self.graph.add_operator(primitive_config)
        self.graph.add_dependency(0, 1)
        
        graph_dict = self.graph.to_dict()
        self.assertEqual(graph_dict['rank'], 0)
        self.assertEqual(len(graph_dict['buffers']), 1)
        self.assertEqual(len(graph_dict['operators']), 1)
        self.assertEqual(len(graph_dict['dependencies']), 1)

    def test_primitive_graph_serialization(self):
        self.graph.add_buffer(0, DataType.F32, 1024, ExecutorType.CUDA)
        
        graph_json = self.graph.to_json()
        self.assertIsInstance(graph_json, str)
        
        reconstructed_graph = PrimitiveGraph.from_json(graph_json)
        self.assertEqual(reconstructed_graph.rank, 0)
        self.assertEqual(len(reconstructed_graph.buffers), 1)

    def test_get_executor_types(self):
        self.graph.executors.append(ExecutorConfig(ExecutorType.CUDA, 4))
        self.graph.executors.append(ExecutorConfig(ExecutorType.CPU, 2))
        
        executor_types = self.graph.get_executor_types()
        self.assertEqual(len(executor_types), 2)
        self.assertIn(ExecutorType.CUDA, executor_types)
        self.assertIn(ExecutorType.CPU, executor_types)

if __name__ == '__main__':
    unittest.main()
