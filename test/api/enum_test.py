import pccl
from textwrap import indent

def print_enum_info(enum_class):
    """打印单个枚举类的信息"""
    print(f"\n{enum_class.__name__}:")
    max_name_length = max(len(name) for name, _ in enum_class.__members__.items())
    
    for name, member in enum_class.__members__.items():
        value = member.value
        print(f"  {name:<{max_name_length}} : {value}")

def print_all_enums():
    """打印所有导出的枚举类型"""
    print("PCCL 导出枚举类型列表:")
    print("=" * 40)
    
    # 获取所有导出的枚举类型
    enums = [
        pccl.ChannelType,
        pccl.DeviceType,
        pccl.BufferType, 
        pccl.NetworkType,
        pccl.DataType,
        pccl.ReduceOpType,
        pccl.Transport,
        pccl.OperationType,
        pccl.PacketType
    ]
    
    for enum in enums:
        print_enum_info(enum)
    
    print("\n枚举使用示例:")
    print(indent("""
from pccl import ChannelType, DataType
print(ChannelType.NVLS)  # 输出: ChannelType.NVLS
print(DataType.FP16)     # 输出: DataType.FP16
""", "  "))

if __name__ == "__main__":
    print_all_enums()
