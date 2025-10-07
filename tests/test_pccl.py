import unittest
import sys
import os

from pccl.ir import CollectiveIR, Chunk, Device, Link, PrimitiveOpType, DeviceType
from pccl.spec.generators import generate_allreduce_spec
from pccl.passes import ValidationPass, CanonicalizationPass, AlgorithmGenerationPass, PassManager
from pccl.utils.analysis import analyze_communication_pattern

class TestPCCL(unittest.TestCase):
    
    def setUp(self):
        self.devices = [
            Device(0, DeviceType.CUDA, 100.0, 1.0),
            Device(1, DeviceType.CUDA, 100.0, 1.0),
            Device(2, DeviceType.CUDA, 100.0, 1.0),
            Device(3, DeviceType.CUDA, 100.0, 1.0)
        ]
        
        self.links = [
            Link(0, 1, 10.0, 1.0),
            Link(1, 2, 10.0, 1.0),
            Link(2, 3, 10.0, 1.0),
            Link(3, 0, 10.0, 1.0)
        ]
    
    def test_chunk_creation(self):
        chunk = Chunk(reduced_ranks={0}, cur_device_id=0, data_size=1024, offset=0)
        self.assertEqual(chunk.reduced_ranks, {0})
        self.assertEqual(chunk.cur_device_id, 0)
        self.assertEqual(chunk.data_size, 1024)
        self.assertEqual(chunk.offset, 0)
    
    def test_ir_creation(self):
        ir = CollectiveIR("test")
        self.assertEqual(ir.name, "test")
        self.assertEqual(len(ir.precondition), 0)
        self.assertEqual(len(ir.postcondition), 0)
    
    def test_allreduce_spec_generation(self):
        ir = generate_allreduce_spec(self.devices, self.links, [0, 1, 2, 3], 1024)
        self.assertEqual(len(ir.precondition), 4)
        self.assertEqual(len(ir.postcondition), 4)
        
        for i, chunk in enumerate(ir.precondition):
            self.assertEqual(chunk.reduced_ranks, {i})
            self.assertEqual(chunk.cur_device_id, i)
        
        for chunk in ir.postcondition:
            self.assertEqual(chunk.reduced_ranks, {0, 1, 2, 3})
    
    def test_validation_pass(self):
        ir = generate_allreduce_spec(self.devices, self.links, [0, 1, 2, 3], 1024)
        
        # Run algorithm generation first to create valid operations
        pm = PassManager()
        pm.add_pass(AlgorithmGenerationPass("ring"))
        pm.run(ir)
        
        # Now validate the IR after algorithm generation
        validation_pass = ValidationPass()
        result = validation_pass.run(ir)
        self.assertTrue(result)
    
    def test_algorithm_generation(self):
        ir = generate_allreduce_spec(self.devices, self.links, [0, 1, 2, 3], 1024)
        
        # Only run algorithm generation, validation should be done after algorithm generation
        pm = PassManager()
        pm.add_pass(AlgorithmGenerationPass("ring"))
        
        success = pm.run(ir)
        self.assertTrue(success)
        
        operations = ir.get_operation_sequence()
        self.assertGreater(len(operations), 0)
    
    def test_communication_analysis(self):
        ir = generate_allreduce_spec(self.devices, self.links, [0, 1, 2, 3], 1024)
        
        # Only run algorithm generation for communication analysis
        pm = PassManager()
        pm.add_pass(AlgorithmGenerationPass("ring"))
        pm.run(ir)
        
        analysis = analyze_communication_pattern(ir)
        self.assertIn("total_operations", analysis)
        self.assertIn("communication_volume", analysis)
        self.assertGreater(analysis["total_operations"], 0)

class TestSimulator(unittest.TestCase):
    
    def setUp(self):
        self.devices = [
            Device(0, DeviceType.CUDA, 100.0, 1.0),
            Device(1, DeviceType.CUDA, 100.0, 1.0)
        ]
        
        self.links = [Link(0, 1, 10.0, 1.0)]
    
    def test_simulator_basic(self):
        from pccl.simulator import CollectiveSimulator
        
        ir = generate_allreduce_spec(self.devices, self.links, [0, 1], 1024)
        
        # Only run algorithm generation for simulator test
        pm = PassManager()
        pm.add_pass(AlgorithmGenerationPass("ring"))
        pm.run(ir)
        
        simulator = CollectiveSimulator(ir)
        timeline = simulator.simulate()
        
        self.assertIn("total_time", timeline)
        self.assertIn("operations", timeline)
        self.assertGreater(timeline["total_time"], 0)

if __name__ == '__main__':
    unittest.main()
