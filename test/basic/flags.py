import pccl

def main():
    # 测试 FunctionTypeFlags
    a = pccl.FunctionTypeFlags("0b1010")
    b = pccl.FunctionTypeFlags("1100")
    print(f"FunctionTypeFlags示例:")
    print(f"初始值 a: {a}, b: {b}")
    print(f"AND操作: {a & b}")
    print(f"OR操作:  {a | b}")
    print(f"XOR操作: {a ^ b}")
    print(f"取反操作: {~a}")
    a.set(0, True)
    print(f"设置后 a: {a}, 测试第0位: {a.test(0)}")
    print("大小:", a.size())
    print("-" * 40)

    # 测试 OperationTypeFlags
    op1 = pccl.OperationTypeFlags("0b0101")
    op2 = pccl.OperationTypeFlags("0b0011")
    print(f"OperationTypeFlags示例:")
    print(f"相等比较: {op1 == op2}, 不等比较: {op1 != op2}")
    print(f"组合操作: {op1 | op2}")
    print(f"异或后取反: {~(op1 ^ op2)}")
    print("-" * 40)

    # 测试 ComponentTypeFlags
    comp = pccl.ComponentTypeFlags()
    print(f"ComponentTypeFlags空值: {comp}")
    comp.set(3, True)
    print(f"设置第3位后: {comp}, 测试第3位: {comp.test(3)}")
    print("-" * 40)

    # 测试 PluginTypeFlags
    plugin = pccl.PluginTypeFlags("1010")
    print(f"PluginTypeFlags字符串初始化: {plugin}")
    print(f"十六进制表示: {hex(int(str(plugin), 2))}")
    print(f"大小: {plugin.size()}")

if __name__ == "__main__":
    main()