import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pccl.api import CollectiveAPI, create_collective_api

class TestCollectiveAPI(unittest.TestCase):
    def setUp(self):
        self.device_ids = [0, 1, 2, 3]
        self.data_size_gb = 0.1

    def test_api_creation(self):
        api = CollectiveAPI()
        self.assertIsNotNone(api)
        self.assertIsNotNone(api.pipeline)
        api.shutdown()

    def test_api_with_custom_pipeline(self):
        from pccl.pipeline import CollectiveExecutionPipeline
        pipeline = CollectiveExecutionPipeline()
        api = CollectiveAPI(pipeline)
        
        self.assertIsNotNone(api)
        self.assertEqual(api.pipeline, pipeline)
        api.shutdown()

    def test_allreduce_operation(self):
        api = create_collective_api("standard")
        result = api.allreduce(self.device_ids, self.data_size_gb)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'execution_time_ms'))
        api.shutdown()

    def test_broadcast_operation(self):
        api = create_collective_api("basic")
        result = api.broadcast(0, self.device_ids, self.data_size_gb)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        api.shutdown()

    def test_allgather_operation(self):
        api = create_collective_api("standard")
        result = api.allgather(self.device_ids, self.data_size_gb)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        api.shutdown()

    def test_alltoall_operation(self):
        api = create_collective_api("standard")
        result = api.alltoall(self.device_ids, self.data_size_gb)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        api.shutdown()

    def test_reducescatter_operation(self):
        api = create_collective_api("standard")
        result = api.reducescatter(self.device_ids, self.data_size_gb)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        api.shutdown()

    def test_execute_custom_operation(self):
        from pccl.collective_ir.specs.factories import create_simple_allreduce_ir
        
        api = create_collective_api("standard")
        collective_ir = create_simple_allreduce_ir(self.device_ids, self.data_size_gb)
        result = api.execute_custom(collective_ir)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        api.shutdown()

    def test_api_shutdown(self):
        api = create_collective_api("standard")
        api.shutdown()
        
        self.assertIsNotNone(api)
        self.assertIsNotNone(api.pipeline)

class TestCollectiveAPIFactory(unittest.TestCase):
    def setUp(self):
        self.device_ids = [0, 1, 2, 3]
        self.data_size_gb = 0.1

    def test_create_collective_api_basic(self):
        api = create_collective_api("basic")
        self.assertIsNotNone(api)
        api.shutdown()

    def test_create_collective_api_standard(self):
        api = create_collective_api("standard")
        self.assertIsNotNone(api)
        api.shutdown()

    def test_create_collective_api_advanced(self):
        api = create_collective_api("advanced")
        self.assertIsNotNone(api)
        api.shutdown()

    def test_create_collective_api_aggressive(self):
        api = create_collective_api("aggressive")
        self.assertIsNotNone(api)
        api.shutdown()

    def test_create_collective_api_invalid_level(self):
        api = create_collective_api("invalid_level")
        self.assertIsNotNone(api)
        api.shutdown()

    def test_api_operations_with_different_optimization_levels(self):
        optimization_levels = ["basic", "standard", "advanced", "aggressive"]
        
        for level in optimization_levels:
            api = create_collective_api(level)
            result = api.allreduce(self.device_ids, self.data_size_gb)
            
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, 'success'))
            api.shutdown()

    def test_multiple_operations_with_same_api(self):
        api = create_collective_api("standard")
        
        results = []
        operations = [
            lambda: api.allreduce(self.device_ids, self.data_size_gb),
            lambda: api.broadcast(0, self.device_ids, self.data_size_gb),
            lambda: api.allgather(self.device_ids, self.data_size_gb)
        ]
        
        for operation in operations:
            result = operation()
            results.append(result)
            self.assertIsNotNone(result)
        
        self.assertEqual(len(results), 3)
        api.shutdown()

if __name__ == '__main__':
    unittest.main()
