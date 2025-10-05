import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pccl.primitive_ir.transformer import CollectiveToPrimitiveTransformer, convert_collective_to_primitive
from pccl.collective_ir.specs.factories import create_simple_allreduce_ir, create_simple_broadcast_ir
from pccl.primitive_ir.primitive_ir import PrimitiveType, DataType, ComputeType, ExecutorType

class TestCollectiveToPrimitiveTransformer(unittest.TestCase):
    def setUp(self):
        self.device_ids = [0, 1, 2, 3]
        self.data_size_gb = 1.0
        self.collective_ir = create_simple_allreduce_ir(self.device_ids, self.data_size_gb)
        self.transformer = CollectiveToPrimitiveTransformer()

    def test_transformer_initialization(self):
        self.assertIsNotNone(self.transformer)
        self.assertEqual(self.transformer.buffer_counter, 0)
        self.assertEqual(self.transformer.operator_counter, 0)
        self.assertEqual(len(self.transformer.buffer_map), 0)
        self.assertEqual(len(self.transformer.device_executor_map), 0)

    def test_transform_method(self):
        for rank in self.device_ids:
            primitive_graph = self.transformer.transform(self.collective_ir, rank)
            self.assertIsNotNone(primitive_graph)
            self.assertEqual(primitive_graph.rank, rank)

    def test_transform_allreduce_ir(self):
        rank = 0
        primitive_graph = self.transformer.transform(self.collective_ir, rank)
        
        self.assertIsNotNone(primitive_graph)
        self.assertEqual(primitive_graph.rank, rank)
        self.assertGreater(len(primitive_graph.buffers), 0)
        self.assertGreater(len(primitive_graph.executors), 0)
        self.assertGreater(len(primitive_graph.operators), 0)

    def test_primitive_graph_structure(self):
        rank = 0
        primitive_graph = self.transformer.transform(self.collective_ir, rank)
        
        self.assertTrue(primitive_graph.validate())
        
        for buffer_config in primitive_graph.buffers:
            self.assertIn(buffer_config.dtype, [DataType.F32])
            self.assertIn(buffer_config.executor_type, [ExecutorType.CUDA, ExecutorType.CPU])
            self.assertGreater(buffer_config.size, 0)
        
        for operator in primitive_graph.operators:
            self.assertIn(operator.type, [PrimitiveType.COPY, PrimitiveType.COMPUTE, PrimitiveType.WRITE])
            self.assertEqual(operator.dtype, DataType.F32)
            self.assertIn(operator.compute_op, [ComputeType.SUM])
            self.assertIn(operator.executor_type, [ExecutorType.CUDA])
            self.assertGreater(operator.data_size, 0)

    def test_buffer_creation(self):
        rank = 0
        primitive_graph = self.transformer.transform(self.collective_ir, rank)
        
        buffer_count = len(primitive_graph.buffers)
        self.assertGreater(buffer_count, 0)
        
        input_buffer = None
        output_buffer = None
        for buffer in primitive_graph.buffers:
            if buffer.buffer_idx == 0:
                input_buffer = buffer
            elif buffer.buffer_idx == 1:
                output_buffer = buffer
        
        self.assertIsNotNone(input_buffer)
        self.assertIsNotNone(output_buffer)
        self.assertEqual(input_buffer.dtype, DataType.F32)
        self.assertEqual(output_buffer.dtype, DataType.F32)

    def test_operator_creation(self):
        rank = 0
        primitive_graph = self.transformer.transform(self.collective_ir, rank)
        
        operator_count = len(primitive_graph.operators)
        self.assertGreater(operator_count, 0)
        
        for operator in primitive_graph.operators:
            self.assertIn(operator.type, [PrimitiveType.COPY, PrimitiveType.COMPUTE, PrimitiveType.WRITE])
            self.assertGreater(operator.data_size, 0)
            self.assertEqual(operator.num_executors, 1)

    def test_dependency_creation(self):
        rank = 0
        primitive_graph = self.transformer.transform(self.collective_ir, rank)
        
        if len(primitive_graph.operators) > 1:
            self.assertGreater(len(primitive_graph.dependencies), 0)
            
            for from_op, to_op in primitive_graph.dependencies:
                self.assertLess(from_op, len(primitive_graph.operators))
                self.assertLess(to_op, len(primitive_graph.operators))
                self.assertLess(from_op, to_op)

    def test_converter_function(self):
        for rank in self.device_ids:
            primitive_graph = convert_collective_to_primitive(self.collective_ir, rank)
            self.assertIsNotNone(primitive_graph)
            self.assertEqual(primitive_graph.rank, rank)

    def test_different_collective_operations(self):
        test_cases = [
            (create_simple_broadcast_ir(0, self.device_ids, self.data_size_gb), "broadcast"),
        ]
        
        for collective_ir, operation_name in test_cases:
            for rank in self.device_ids:
                primitive_graph = convert_collective_to_primitive(collective_ir, rank)
                self.assertIsNotNone(primitive_graph, f"Failed for {operation_name} at rank {rank}")

    def test_empty_ir_handling(self):
        from pccl.collective_ir.core.ir import CollectiveIR, ClusterMesh, TaskMap, CollectiveSpec
        from pccl.collective_ir.core.enums import CollectiveOpType
        
        empty_ir = CollectiveIR(
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
        
        primitive_graph = convert_collective_to_primitive(empty_ir, 0)
        self.assertIsNotNone(primitive_graph)

    def test_serialization(self):
        rank = 0
        primitive_graph = convert_collective_to_primitive(self.collective_ir, rank)
        
        graph_dict = primitive_graph.to_dict()
        self.assertIsInstance(graph_dict, dict)
        self.assertEqual(graph_dict['rank'], rank)
        
        graph_json = primitive_graph.to_json()
        self.assertIsInstance(graph_json, str)
        self.assertGreater(len(graph_json), 0)
        
        reconstructed_graph = primitive_graph.from_json(graph_json)
        self.assertIsNotNone(reconstructed_graph)

    def test_invalid_rank_handling(self):
        invalid_rank = 999
        primitive_graph = convert_collective_to_primitive(self.collective_ir, invalid_rank)
        self.assertIsNotNone(primitive_graph)
        self.assertEqual(primitive_graph.rank, invalid_rank)

    def test_memory_region_handling(self):
        rank = 0
        primitive_graph = convert_collective_to_primitive(self.collective_ir, rank)
        
        for operator in primitive_graph.operators:
            if operator.type == PrimitiveType.COPY:
                self.assertNotEqual(operator.src_buffer_idx, -1)
                self.assertNotEqual(operator.dst_buffer_idx, -1)
            elif operator.type == PrimitiveType.COMPUTE:
                self.assertNotEqual(operator.src_buffer_idx, -1)
                self.assertNotEqual(operator.dst_buffer_idx, -1)

class TestConverterFunction(unittest.TestCase):
    def setUp(self):
        self.device_ids = [0, 1, 2, 3]
        self.data_size_gb = 1.0
        self.collective_ir = create_simple_allreduce_ir(self.device_ids, self.data_size_gb)

    def test_convert_collective_to_primitive(self):
        for rank in self.device_ids:
            primitive_graph = convert_collective_to_primitive(self.collective_ir, rank)
            self.assertIsNotNone(primitive_graph)
            self.assertEqual(primitive_graph.rank, rank)

    def test_converter_with_different_ranks(self):
        for rank in [0, 1, 2]:
            primitive_graph = convert_collective_to_primitive(self.collective_ir, rank)
            self.assertEqual(primitive_graph.rank, rank)

    def test_converter_returns_valid_graph(self):
        primitive_graph = convert_collective_to_primitive(self.collective_ir, 0)
        self.assertTrue(primitive_graph.validate())

if __name__ == '__main__':
    unittest.main()
