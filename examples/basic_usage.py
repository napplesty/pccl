#!/usr/bin/env python3

import sys
import os

from pccl.ir import CollectiveIR, Device, Link, DeviceType
from pccl.spec.generators import generate_allreduce_spec
from pccl.passes import (PassManager, ValidationPass, CanonicalizationPass, 
                         ChunkOptimizationPass, AlgorithmGenerationPass, 
                         PerformanceModelingPass)
from pccl.simulator import CollectiveSimulator
from pccl.utils.analysis import analyze_communication_pattern

def demo_allreduce():
    print("=== PCCL AllReduce Demo ===")
    
    devices = [
        Device(0, DeviceType.CUDA, 100.0, 1.0),
        Device(1, DeviceType.CUDA, 100.0, 1.0),
        Device(2, DeviceType.CUDA, 100.0, 1.0),
        Device(3, DeviceType.CUDA, 100.0, 1.0)
    ]
    
    links = [
        Link(0, 1, 10.0, 1.0),
        Link(1, 2, 10.0, 1.0),
        Link(2, 3, 10.0, 1.0),
        Link(3, 0, 10.0, 1.0)
    ]
    
    ranks = [0, 1, 2, 3]
    data_size = 16 * 1024 * 1024
    
    print(f"Creating AllReduce spec for {len(ranks)} ranks, {data_size} bytes")
    
    ir = generate_allreduce_spec(devices, links, ranks, data_size)
    
    pm = PassManager()
    pm.add_pass(ValidationPass())
    pm.add_pass(ChunkOptimizationPass(chunk_size=1024*1024))
    pm.add_pass(AlgorithmGenerationPass("ring"))
    pm.add_pass(CanonicalizationPass())
    pm.add_pass(PerformanceModelingPass())
    
    print("Running compilation passes...")
    success = pm.run(ir)
    
    if success:
        print("✓ Compilation successful!")
        
        analysis = analyze_communication_pattern(ir)
        print(f"Operations: {analysis['total_operations']}")
        print(f"Communication Volume: {analysis['communication_volume'] / (1024*1024):.2f} MB")
        print(f"Parallelism Factor: {analysis['parallelism_factor']:.2f}")
        
        print("\nRunning simulation...")
        simulator = CollectiveSimulator(ir)
        timeline = simulator.simulate()
        
        print(f"Estimated Time: {timeline['total_time']:.6f} seconds")
        print(f"Throughput: {timeline['throughput'] / (1024*1024):.2f} MB/s")
        
        print("\nDetailed Timeline:")
        simulator.print_timeline_report()
        
    else:
        print("✗ Compilation failed!")
        for pass_name, result in pm.pass_results.items():
            status = "✓" if result else "✗"
            print(f"  {status} {pass_name}")

if __name__ == '__main__':
    demo_allreduce()
