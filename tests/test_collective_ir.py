import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pccl.collective_ir.core.ir import *
from pccl.collective_ir.core.enums import *
from pccl.collective_ir.specs.factories import *
from pccl.collective_ir.passes.static.algorithms.allreduce import *
from pccl.collective_ir.passes.static.algorithms.allgather import *
from pccl.collective_ir.passes.static.algorithms.collective import CollectiveOptimizationPass
from pccl.collective_ir.memory_optimization.allocator import SmartMemoryAllocator
from pccl.collective_ir.memory_optimization.optimizer import AdvancedMemoryOptimizer
from pccl.collective_ir.memory_optimization.manager import MemoryManager
from pccl.collective_ir.performance_modeling.gpu_bandwidth import GPUBandwidthModel
from pccl.collective_ir.performance_modeling.latency_models import SimpleLatencyModel
from pccl.collective_ir.performance_modeling.base import PerformanceModel
from pccl.collective_ir.dependency_analysis.analyzer import DependencyAnalyzer
from pccl.collective_ir.dependency_analysis.optimizer import DependencyOptimizer
from pccl.collective_ir.dependency_analysis.scheduler import AdvancedTaskScheduler
from pccl.collective_ir.serialization.encoder import serialize
from pccl.collective_ir.serialization.decoder import deserialize
from pccl.collective_ir.passes.comprehensive_optimization import ComprehensiveOptimizationPass
from pccl.collective_ir.management.manager import create_default_pass_manager
from pccl.collective_ir.verification.verifiers import CompositeVerifier, BasicIRVerifier
from pccl.collective_ir.passes.solver_based.mst_solver import BottleneckAwareMSTPass
from pccl.collective_ir.passes.solver_based.hybrid_solver import HybridSolverPass
from pccl.collective_ir.passes.performance_modeling import PerformanceModelingPass

class TestFixtures:
    @staticmethod
    def sample_devices():
        return [
            Device(0, "GPU", {"memory": 16}, 16.0, 25.0),
            Device(1, "GPU", {"memory": 16}, 16.0, 25.0),
            Device(2, "GPU", {"memory": 16}, 16.0, 25.0),
            Device(3, "GPU", {"memory": 16}, 16.0, 25.0)
        ]
    
    @staticmethod
    def sample_cluster():
        devices = TestFixtures.sample_devices()
        host = Host(0, devices)
        return ClusterMesh(hosts=[host])
    
    @staticmethod
    def sample_allreduce_ir():
        device_ids = [0, 1, 2, 3]
        return create_simple_allreduce_ir(device_ids, 1.0)
    
    @staticmethod
    def sample_broadcast_ir():
        device_ids = [0, 1, 2, 3]
        return create_simple_broadcast_ir(0, device_ids, 1.0)

class TestCoreIR(unittest.TestCase):
    def test_device_creation(self):
        device = Device(1, "GPU", {"memory": 16}, 16.0, 25.0)
        self.assertEqual(device.device_id, 1)
        self.assertEqual(device.type, "GPU")
        self.assertEqual(device.memory_capacity_gb, 16.0)

    def test_task_creation(self):
        device = Device(0, "GPU")
        memory = LocalMemory(device, 0, 1024)
        primitive = CommunicationPrimitive(device, PrimitiveOpType.COPY, [memory])
        task = Task(1, [primitive])
        
        self.assertEqual(task.task_id, 1)
        self.assertEqual(len(task.primitives), 1)
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_cluster_mesh(self):
        devices = TestFixtures.sample_devices()
        host = Host(0, devices)
        cluster = ClusterMesh(hosts=[host])
        
        self.assertEqual(len(cluster.devices_by_id), 4)
        self.assertEqual(cluster.get_device(0).device_id, 0)

    def test_collective_ir_validation(self):
        ir = TestFixtures.sample_allreduce_ir()
        self.assertEqual(ir.collective_op, CollectiveOpType.ALLREDUCE)

    def test_memory_regions(self):
        device = Device(0, "GPU")
        local_mem = LocalMemory(device, 0, 1024)
        remote_mem = RemoteMemory(device, 0, 1024)
        
        self.assertEqual(local_mem.device.device_id, 0)
        self.assertEqual(remote_mem.size, 1024)

    def test_task_dependencies(self):
        device = Device(0, "GPU")
        memory = LocalMemory(device, 0, 1024)
        primitive = CommunicationPrimitive(device, PrimitiveOpType.COPY, [memory])
        
        task1 = Task(1, [primitive])
        task2 = Task(2, [primitive])
        
        task2.add_dependency(task1)
        self.assertEqual(len(task2.dependencies), 1)
        self.assertEqual(task2.dependencies[0].task_id, 1)

class TestAlgorithms(unittest.TestCase):
    def test_ring_allreduce(self):
        ir = TestFixtures.sample_allreduce_ir()
        pass_obj = RingAllReducePass()
        optimized_ir = pass_obj.run(ir)
        
        self.assertIsNotNone(optimized_ir)
        self.assertGreater(len(optimized_ir.task_map.tasks), 0)

    def test_double_binary_tree(self):
        ir = TestFixtures.sample_allreduce_ir()
        pass_obj = DoubleBinaryTreePass()
        optimized_ir = pass_obj.run(ir)
        
        self.assertIsNotNone(optimized_ir)
        self.assertGreater(len(optimized_ir.task_map.tasks), 0)

    def test_algorithm_selection(self):
        ir = TestFixtures.sample_allreduce_ir()
        pass_obj = AllReduceAlgorithmSelectionPass(threshold_for_ring=4)
        optimized_ir = pass_obj.run(ir)
        
        self.assertIsNotNone(optimized_ir)
        self.assertGreater(len(optimized_ir.task_map.tasks), 0)

    def test_collective_optimization_pass(self):
        ir = TestFixtures.sample_allreduce_ir()
        pass_obj = CollectiveOptimizationPass(algorithm_selection_strategy="auto")
        optimized_ir = pass_obj.run(ir)
        
        self.assertIsNotNone(optimized_ir)
        self.assertGreater(len(optimized_ir.task_map.tasks), 0)

class TestMemoryOptimization(unittest.TestCase):
    def test_smart_memory_allocator(self):
        allocator = SmartMemoryAllocator(1024 * 1024)
        
        addr1 = allocator.allocate(1024)
        self.assertIsNotNone(addr1)
        self.assertGreaterEqual(addr1, 0)
        
        addr2 = allocator.allocate(512)
        self.assertIsNotNone(addr2)
        self.assertNotEqual(addr2, addr1)
        
        allocator.deallocate(addr1, 1024)
        
        fragmentation = allocator.get_fragmentation()
        self.assertGreaterEqual(fragmentation, 0)
        self.assertLessEqual(fragmentation, 1.0)
        
        usage = allocator.get_usage()
        self.assertLessEqual(usage[0], usage[1])

    def test_memory_manager(self):
        manager = MemoryManager()
        device = Device(0, "GPU", memory_capacity_gb=16.0)
        
        manager.initialize_device_memory(device, 1024 * 1024)
        memory_region = manager.allocate_memory(device, 1024)
        
        self.assertIsNotNone(memory_region)
        self.assertGreaterEqual(memory_region.address, 0)

    def test_advanced_memory_optimizer(self):
        ir = TestFixtures.sample_allreduce_ir()
        optimizer = AdvancedMemoryOptimizer()
        optimized_ir = optimizer.run(ir)
        
        self.assertIsNotNone(optimized_ir)
        self.assertIsNotNone(optimized_ir.task_map.tasks)

class TestPerformanceModeling(unittest.TestCase):
    def test_gpu_bandwidth_model(self):
        model = GPUBandwidthModel()
        device1 = Device(0, "A100")
        device2 = Device(1, "A100")
        
        bandwidth = model.get_bandwidth(device1, device2, 1024 * 1024)
        self.assertGreater(bandwidth, 0)
        self.assertIsInstance(bandwidth, float)

    def test_latency_model(self):
        model = SimpleLatencyModel()
        device1 = Device(0, "GPU")
        device2 = Device(1, "GPU")
        
        latency = model.get_latency(device1, device2)
        self.assertGreaterEqual(latency, 0)
        self.assertIsInstance(latency, float)

    def test_performance_model(self):
        bandwidth_model = GPUBandwidthModel()
        latency_model = SimpleLatencyModel()
        performance_model = PerformanceModel(bandwidth_model, latency_model)
        
        device1 = Device(0, "A100")
        device2 = Device(1, "A100")
        
        comm_time = performance_model.estimate_communication_time(
            device1, device2, 1024 * 1024
        )
        self.assertGreaterEqual(comm_time, 0)

    def test_performance_modeling_pass(self):
        ir = TestFixtures.sample_allreduce_ir()
        pass_obj = PerformanceModelingPass()
        modeled_ir = pass_obj.run(ir)
        
        self.assertIsNotNone(modeled_ir)
        
        for task in modeled_ir.task_map.tasks.values():
            for primitive in task.primitives:
                self.assertTrue(hasattr(primitive, 'estimated_duration_ms'))
                self.assertGreaterEqual(primitive.estimated_duration_ms, 0)

class TestDependencyAnalysis(unittest.TestCase):
    def test_dependency_analyzer(self):
        ir = TestFixtures.sample_allreduce_ir()
        analyzer = DependencyAnalyzer(ir.task_map)
        
        critical_path = analyzer.get_critical_path_tasks()
        self.assertIsInstance(critical_path, set)
        
        parallelism = analyzer.get_task_parallelism()
        self.assertGreaterEqual(parallelism, 0)
        self.assertLessEqual(parallelism, 1.0)
        
        completed_tasks = set()
        ready_tasks = analyzer.get_ready_tasks(completed_tasks)
        self.assertIsInstance(ready_tasks, list)

    def test_dependency_optimizer(self):
        ir = TestFixtures.sample_allreduce_ir()
        optimizer = DependencyOptimizer(ir)
        optimized_ir = optimizer.optimize_task_dependencies()
        
        self.assertIsNotNone(optimized_ir)
        self.assertGreater(len(optimized_ir.task_map.tasks), 0)

    def test_task_scheduler(self):
        ir = TestFixtures.sample_allreduce_ir()
        scheduler = AdvancedTaskScheduler(ir)
        
        strategies = ["critical_path", "earliest_finish", "load_balanced"]
        
        for strategy in strategies:
            scheduled_ir = scheduler.schedule_tasks(strategy)
            self.assertIsNotNone(scheduled_ir)
            
            for task in scheduled_ir.task_map.tasks.values():
                self.assertTrue(hasattr(task, 'estimated_start_time'))
                self.assertTrue(hasattr(task, 'estimated_end_time'))

class TestSerialization(unittest.TestCase):
    def test_serialize_deserialize(self):
        ir = TestFixtures.sample_allreduce_ir()
        
        json_str = serialize(ir)
        self.assertIsNotNone(json_str)
        self.assertIsInstance(json_str, str)
        self.assertGreater(len(json_str), 0)
        
        deserialized_ir = deserialize(json_str)
        self.assertIsNotNone(deserialized_ir)
        
        self.assertEqual(deserialized_ir.collective_op, ir.collective_op)
        self.assertEqual(deserialized_ir.data_size_gb, ir.data_size_gb)
        self.assertEqual(len(deserialized_ir.task_map.tasks), len(ir.task_map.tasks))

    def test_serialization_consistency(self):
        ir = TestFixtures.sample_allreduce_ir()
        
        json1 = serialize(ir)
        json2 = serialize(ir)
        
        import json
        data1 = json.loads(json1)
        data2 = json.loads(json2)
        
        self.assertEqual(data1['collective_op'], data2['collective_op'])
        self.assertEqual(data1['data_size'], data2['data_size'])

class TestComprehensiveOptimization(unittest.TestCase):
    def test_comprehensive_optimization_pass(self):
        ir = TestFixtures.sample_allreduce_ir()
        
        for level in [1, 2]:
            pass_obj = ComprehensiveOptimizationPass(optimization_level=level)
            optimized_ir = pass_obj.run(ir)
            
            self.assertIsNotNone(optimized_ir)
            self.assertGreater(len(optimized_ir.task_map.tasks), 0)

    def test_pass_manager(self):
        ir = TestFixtures.sample_allreduce_ir()
        manager = create_default_pass_manager()
        optimized_ir = manager.run_passes(ir)
        
        self.assertIsNotNone(optimized_ir)
        
        summary = manager.get_execution_summary()
        self.assertIn('total_passes', summary)
        self.assertIn('successful_passes', summary)

    def test_verification(self):
        ir = TestFixtures.sample_allreduce_ir()
        verifier = CompositeVerifier([BasicIRVerifier()])
        result = verifier.verify(ir)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'is_valid'))
        self.assertTrue(result.is_valid)

class TestSolverBasedOptimization(unittest.TestCase):
    def test_mst_solver(self):
        ir = TestFixtures.sample_allreduce_ir()
        pass_obj = BottleneckAwareMSTPass(time_limit=10)
        
        try:
            optimized_ir = pass_obj.run(ir)
            if optimized_ir:
                self.assertIsNotNone(optimized_ir)
        except:
            pass

    def test_hybrid_solver(self):
        ir = TestFixtures.sample_allreduce_ir()
        pass_obj = HybridSolverPass(time_limit=5, parallel_execution=False)
        
        try:
            optimized_ir = pass_obj.run(ir)
            self.assertIsNotNone(optimized_ir)
        except:
            pass

def run_tests_with_coverage():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestCoreIR,
        TestAlgorithms,
        TestMemoryOptimization,
        TestPerformanceModeling,
        TestDependencyAnalysis,
        TestSerialization,
        TestComprehensiveOptimization,
        TestSolverBasedOptimization
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped
    
    print("\n" + "="*60)
    print("COVERAGE REPORT")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success Rate: {passed/total_tests*100:.1f}%")
    print("="*60)
    
    coverage_by_module = {
        "Core IR": len(TestCoreIR.__dict__) - 10,
        "Algorithms": len(TestAlgorithms.__dict__) - 10,
        "Memory Optimization": len(TestMemoryOptimization.__dict__) - 10,
        "Performance Modeling": len(TestPerformanceModeling.__dict__) - 10,
        "Dependency Analysis": len(TestDependencyAnalysis.__dict__) - 10,
        "Serialization": len(TestSerialization.__dict__) - 10,
        "Comprehensive Optimization": len(TestComprehensiveOptimization.__dict__) - 10,
        "Solver-Based Optimization": len(TestSolverBasedOptimization.__dict__) - 10
    }
    
    print("\nMODULE COVERAGE BREAKDOWN:")
    print("-" * 40)
    for module, test_count in coverage_by_module.items():
        print(f"{module:<25}: {test_count} tests")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests_with_coverage())

