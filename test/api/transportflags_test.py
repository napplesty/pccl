import unittest
import pccl

class TestTransportFlags(unittest.TestCase):
    
    def test_basic_operations(self):
        # 测试单个Transport标志
        ib_flag = pccl.TransportFlags(pccl.Transport.IB)
        self.assertTrue(ib_flag.has(pccl.Transport.IB))
        self.assertFalse(ib_flag.has(pccl.Transport.Ethernet))
        
        # 测试组合标志
        combo = ib_flag | pccl.Transport.Ethernet
        self.assertTrue(combo.has(pccl.Transport.IB))
        self.assertTrue(combo.has(pccl.Transport.Ethernet))
        
        # 测试清除标志
        cleared = combo & ~pccl.Transport.Ethernet
        self.assertFalse(cleared.has(pccl.Transport.Ethernet))
    
    def test_string_conversion(self):
        # 测试二进制字符串转换（假设Transport顺序为：HostIpc=0, CudaIpc=1, IB=2...）
        flags = pccl.TransportFlags.from_string("101")  # 二进制表示
        self.assertTrue(flags.has(pccl.Transport.HostIpc))
        self.assertFalse(flags.has(pccl.Transport.CudaIpc))
        self.assertTrue(flags.has(pccl.Transport.NVLS))
    
    def test_operator_priority(self):
        # 测试运算符优先级
        flags = pccl.Transport.IB | pccl.Transport.Ethernet & pccl.Transport.NVLS
        self.assertTrue(flags.has(pccl.Transport.IB))
        self.assertFalse(flags.has(pccl.Transport.Ethernet))
    
    def test_edge_cases(self):
        # 测试空标志
        empty = pccl.TransportFlags()
        self.assertTrue(empty.none())
        
        # 测试全标志
        full = ~empty
        self.assertTrue(full.all())

if __name__ == '__main__':
    unittest.main()