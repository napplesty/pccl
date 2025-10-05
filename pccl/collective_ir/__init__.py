"""
Collective IR - Intermediate Representation for Collective Communication Operations
"""

from .core.ir import *
from .core.enums import *
from .core.exceptions import *
from .verification.verifiers import *
from .passes.base import IRPass
from .passes.static.canonical import CanonicalPass
from .passes.static.scheduling import TaskSchedulingPass
from .passes.static.memory import MemoryOptimizationPass
from .passes.static.algorithms.collective import CollectiveOptimizationPass
from .passes.static.algorithms.allreduce import (
    RingAllReducePass, DoubleBinaryTreePass, HalvingDoublingPass, AllReduceAlgorithmSelectionPass
)
from .passes.static.algorithms.alltoall import AllToAllOptimizationPass, PairwiseExchangeAllToAllPass
from .passes.static.algorithms.broadcast import TreeBroadcastPass
from .passes.static.algorithms.allgather import AllGatherOptimizationPass, RingAllGatherPass, RecursiveDoublingAllGatherPass, BruckAllGatherPass
from .passes.static.algorithms.reducescatter import ReduceScatterOptimizationPass, RingReduceScatterPass, PairwiseReduceScatterPass
from .passes.solver_based.smt_solver import SMTBasedOptimizationPass
from .passes.solver_based.mst_solver import BottleneckAwareMSTPass
from .passes.solver_based.milp_solver import MILPOptimizationPass
from .passes.solver_based.hybrid_solver import HybridSolverPass
from .management.manager import PassManager, create_default_pass_manager
from .serialization.encoder import serialize
from .serialization.decoder import deserialize
from .specs.factories import *
from .performance_modeling import *
from .dependency_analysis import *
from .memory_optimization import *
from .passes.performance_modeling import PerformanceModelingPass
from .passes.comprehensive_optimization import ComprehensiveOptimizationPass

__all__ = [
    # Core
    'CollectiveIR', 'Task', 'TaskMap', 'CommunicationPrimitive', 'Chunk', 'ReducedChunk',
    'Precondition', 'Postcondition', 'CollectiveSpec', 'Device', 'ClusterMesh',
    'CollectiveOpType', 'PrimitiveOpType', 'ChunkState', 'TaskStatus',
    
    # Verification
    'IRVerifier', 'BasicIRVerifier', 'ChunkStateVerifier', 'PerformanceVerifier', 'CompositeVerifier',
    
    # Static Passes
    'IRPass', 'CanonicalPass', 'TaskSchedulingPass', 'MemoryOptimizationPass', 'CollectiveOptimizationPass',
    'RingAllReducePass', 'DoubleBinaryTreePass', 'HalvingDoublingPass', 'AllReduceAlgorithmSelectionPass',
    'AllToAllOptimizationPass', 'PairwiseExchangeAllToAllPass', 'TreeBroadcastPass',
    'AllGatherOptimizationPass', 'RingAllGatherPass', 'RecursiveDoublingAllGatherPass', 'BruckAllGatherPass',
    'ReduceScatterOptimizationPass', 'RingReduceScatterPass', 'PairwiseReduceScatterPass',
    
    # Solver-Based Passes
    'SMTBasedOptimizationPass', 'BottleneckAwareMSTPass', 'MILPOptimizationPass', 'HybridSolverPass',
    
    # Management
    'PassManager', 'create_default_pass_manager',
    
    # Serialization
    'serialize', 'deserialize',
    
    # 基础specs
    'CollectiveSpecBuilder', 'SpecValidator',
    'create_allreduce_spec', 'create_reduce_spec', 'create_broadcast_spec',
    'create_alltoall_spec', 'create_scatter_spec', 'create_gather_spec',
    'create_allgather_spec', 'create_reducescatter_spec',
    'create_simple_allreduce_ir', 'create_simple_reduce_ir', 'create_simple_broadcast_ir',
    'create_simple_alltoall_ir', 'create_simple_scatter_ir', 'create_simple_gather_ir',
    'create_simple_allgather_ir', 'create_simple_reducescatter_ir',
    'create_complex_topology_ir', 'create_ir_from_spec', 'validate_ir_against_spec',
    

    # 性能建模
    'PerformanceModel', 'BandwidthModel', 'LatencyModel',
    'GPUBandwidthModel', 'SimpleLatencyModel', 'ComputationCostModel',
    'MemoryBandwidthModel', 'PerformanceModelingPass',
    
    # 依赖关系分析
    'DependencyAnalyzer', 'DependencyOptimizer', 'AdvancedTaskScheduler',
    
    # 内存优化
    'MemoryManager', 'SmartMemoryAllocator', 'MemoryPool', 'AdvancedMemoryOptimizer',
    
    # 综合优化
    'ComprehensiveOptimizationPass',
]
