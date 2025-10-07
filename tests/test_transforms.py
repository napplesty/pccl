import unittest
from pccl.ir.core import CollectiveIR, Device, Chunk, PrimitiveOp
from pccl.ir.types import PrimitiveOpType, DeviceType
from pccl.transforms import HighToLowConverter
from pccl.low_ir.core import LowLevelIR
from pccl.low_ir.types import PrimitiveType, DataType, ComputeType, ExecutorType


class TestHighToLowConverter(unittest.TestCase):
    """测试高级IR到低级IR转换器"""
    
    def setUp(self):
        """测试前准备"""
        self.converter = HighToLowConverter()
        self.high_ir = CollectiveIR("test_allreduce")
        
        # 添加设备
        self.device1 = Device(0, DeviceType.CUDA, 1000.0, 8.0)
        self.device2 = Device(1, DeviceType.CUDA, 1000.0, 8.0)
        self.high_ir.add_device(self.device1)
        self.high_ir.add_device(self.device2)
        
        # 添加数据块
        self.chunk1 = Chunk(reduced_ranks={0}, cur_device_id=0, data_size=1024, offset=0)
        self.chunk2 = Chunk(reduced_ranks={1}, cur_device_id=1, data_size=1024, offset=0)
        self.result_chunk = Chunk(reduced_ranks={0, 1}, cur_device_id=0, data_size=1024, offset=0)
    
    def test_convert_empty_ir(self):
        """测试转换空IR"""
        empty_ir = CollectiveIR("empty")
        low_ir = self.converter.convert(empty_ir)
        
        self.assertIsInstance(low_ir, LowLevelIR)
        self.assertEqual(low_ir.name, "empty")
        self.assertEqual(len(low_ir.buffers), 0)
        self.assertEqual(len(low_ir.operators), 0)
        self.assertTrue(low_ir.validate())
    
    def test_convert_copy_operation(self):
        """测试转换COPY操作"""
        # 添加COPY操作
        op_id = self.high_ir.create_operation(
            op_type=PrimitiveOpType.COPY,
            src_chunk=self.chunk1,
            tgt_chunk=self.chunk2,
            src_device=0,
            tgt_device=1
        )
        
        # 转换
        low_ir = self.converter.convert(self.high_ir)
        
        # 验证转换结果
        self.assertEqual(len(low_ir.operators), 1)
        op_config = list(low_ir.operators.values())[0]
        self.assertEqual(op_config.type, PrimitiveType.COPY)
        self.assertEqual(op_config.dtype, DataType.F32)
        self.assertNotEqual(op_config.src_buffer_idx, -1)
        self.assertNotEqual(op_config.dst_buffer_idx, -1)
        self.assertEqual(op_config.data_size, 1024)
        self.assertTrue(low_ir.validate())
    
    def test_convert_reduce_operation(self):
        """测试转换REDUCE操作"""
        # 添加REDUCE操作
        op_id = self.high_ir.create_operation(
            op_type=PrimitiveOpType.REDUCE,
            src_chunk=self.chunk1,
            tgt_chunk=self.result_chunk,
            src_device=0,
            tgt_device=0,
            metadata={'compute_op': 'sum'}
        )
        
        # 转换
        low_ir = self.converter.convert(self.high_ir)
        
        # 验证转换结果
        self.assertEqual(len(low_ir.operators), 1)
        op_config = list(low_ir.operators.values())[0]
        self.assertEqual(op_config.type, PrimitiveType.COMPUTE)
        self.assertEqual(op_config.compute_op, ComputeType.SUM)
        self.assertEqual(op_config.data_size, 1024)
        self.assertTrue(low_ir.validate())
    
    def test_convert_notify_operations(self):
        """测试转换NOTIFY和GET_NOTIFIED操作"""
        # 添加NOTIFY操作
        notify_id = self.high_ir.create_operation(
            op_type=PrimitiveOpType.NOTIFY,
            src_device=0,
            tgt_device=1
        )
        
        # 添加GET_NOTIFIED操作
        get_notified_id = self.high_ir.create_operation(
            op_type=PrimitiveOpType.GET_NOTIFIED,
            src_device=1,
            tgt_device=1,
            dependencies=[notify_id]
        )
        
        # 转换
        low_ir = self.converter.convert(self.high_ir)
        
        # 验证转换结果
        self.assertEqual(len(low_ir.operators), 2)
        
        # 检查操作类型映射
        op_types = [op.type for op in low_ir.operators.values()]
        self.assertIn(PrimitiveType.SIGNAL, op_types)
        self.assertIn(PrimitiveType.WAITSIGNAL, op_types)
        
        # 检查依赖关系
        self.assertEqual(len(low_ir.dependencies), 1)
        self.assertTrue(low_ir.validate())
    
    def test_convert_with_dependencies(self):
        """测试转换带依赖关系的操作序列"""
        # 创建操作序列：COPY -> REDUCE
        copy_id = self.high_ir.create_operation(
            op_type=PrimitiveOpType.COPY,
            src_chunk=self.chunk1,
            tgt_chunk=self.chunk2,
            src_device=0,
            tgt_device=1
        )
        
        reduce_id = self.high_ir.create_operation(
            op_type=PrimitiveOpType.REDUCE,
            src_chunk=self.chunk2,
            tgt_chunk=self.result_chunk,
            src_device=1,
            tgt_device=0,
            dependencies=[copy_id]
        )
        
        # 转换
        low_ir = self.converter.convert(self.high_ir)
        
        # 验证转换结果
        self.assertEqual(len(low_ir.operators), 2)
        self.assertEqual(len(low_ir.dependencies), 1)
        
        # 检查依赖关系是否正确转换
        for op_id, deps in low_ir.dependencies.items():
            self.assertEqual(len(deps), 1)  # 应该有一个依赖
        
        self.assertTrue(low_ir.validate())
    
    def test_convert_with_pre_post_conditions(self):
        """测试转换带前后置条件的IR"""
        # 设置前置条件和后置条件
        self.high_ir.set_precondition([self.chunk1, self.chunk2])
        self.high_ir.set_postcondition([self.result_chunk])
        
        # 添加操作
        self.high_ir.create_operation(
            op_type=PrimitiveOpType.REDUCE,
            src_chunk=self.chunk1,
            tgt_chunk=self.result_chunk,
            src_device=0,
            tgt_device=0
        )
        
        # 转换
        low_ir = self.converter.convert(self.high_ir)
        
        # 验证张量信息转换
        self.assertEqual(len(low_ir.input_tensors), 2)  # 两个输入张量
        self.assertEqual(len(low_ir.output_tensors), 1)  # 一个输出张量
        
        for tensor in low_ir.input_tensors + low_ir.output_tensors:
            self.assertEqual(tensor.dtype, DataType.F32)
            self.assertEqual(tensor.device_type, ExecutorType.CUDA)
            self.assertEqual(tensor.total_bytes, 1024)
        
        self.assertTrue(low_ir.validate())
    
    def test_device_type_mapping(self):
        """测试设备类型映射"""
        # 测试CUDA设备映射
        cuda_device = Device(0, DeviceType.CUDA, 1000.0, 8.0)
        self.high_ir.add_device(cuda_device)
        
        # 测试CPU设备映射
        cpu_device = Device(1, DeviceType.CPU, 100.0, 2.0)
        self.high_ir.add_device(cpu_device)
        
        # 转换
        low_ir = self.converter.convert(self.high_ir)
        
        # 验证执行器类型
        executor_types = [executor.executor_type for executor in low_ir.executors.values()]
        self.assertIn(ExecutorType.CUDA, executor_types)
        self.assertIn(ExecutorType.CPU, executor_types)
    
    def test_compute_op_mapping(self):
        """测试计算操作映射"""
        compute_ops = ['sum', 'max', 'min', 'prod']
        
        for compute_op in compute_ops:
            high_ir = CollectiveIR(f"test_{compute_op}")
            high_ir.add_device(self.device1)
            
            # 添加REDUCE操作
            high_ir.create_operation(
                op_type=PrimitiveOpType.REDUCE,
                src_chunk=self.chunk1,
                tgt_chunk=self.result_chunk,
                src_device=0,
                tgt_device=0,
                metadata={'compute_op': compute_op}
            )
            
            # 转换
            low_ir = self.converter.convert(high_ir)
            
            # 验证计算操作映射
            op_config = list(low_ir.operators.values())[0]
            expected_compute_op = getattr(ComputeType, compute_op.upper())
            self.assertEqual(op_config.compute_op, expected_compute_op)
    
    def test_serialization_roundtrip(self):
        """测试序列化往返"""
        # 创建复杂的IR
        self.high_ir.set_precondition([self.chunk1, self.chunk2])
        self.high_ir.set_postcondition([self.result_chunk])
        
        copy_id = self.high_ir.create_operation(
            op_type=PrimitiveOpType.COPY,
            src_chunk=self.chunk1,
            tgt_chunk=self.chunk2,
            src_device=0,
            tgt_device=1
        )
        
        reduce_id = self.high_ir.create_operation(
            op_type=PrimitiveOpType.REDUCE,
            src_chunk=self.chunk2,
            tgt_chunk=self.result_chunk,
            src_device=1,
            tgt_device=0,
            dependencies=[copy_id]
        )
        
        # 转换
        low_ir = self.converter.convert(self.high_ir)
        
        # 序列化
        json_str = low_ir.to_json()
        
        # 反序列化
        low_ir2 = LowLevelIR.from_json(json_str)
        
        # 验证往返一致性
        self.assertEqual(low_ir.name, low_ir2.name)
        self.assertEqual(len(low_ir.buffers), len(low_ir2.buffers))
        self.assertEqual(len(low_ir.operators), len(low_ir2.operators))
        self.assertEqual(len(low_ir.dependencies), len(low_ir2.dependencies))
        self.assertEqual(len(low_ir.input_tensors), len(low_ir2.input_tensors))
        self.assertEqual(len(low_ir.output_tensors), len(low_ir2.output_tensors))
        
        self.assertTrue(low_ir2.validate())


if __name__ == '__main__':
    unittest.main()
