import pccl
import os
import time

def demonstrate_utils():
    """演示工具函数的使用"""
    # 获取启动时间戳
    print(f"启动时间戳: {pccl.utils.get_start_timestamp()}")
    
    # 创建目录演示
    test_dir = "./test_dir"
    try:
        pccl.utils.create_dir(test_dir)
        print(f"成功创建目录: {os.path.abspath(test_dir)}")
    except Exception as e:
        print(f"目录创建失败: {str(e)}")
    
    # 设置CPU亲和性（示例设置为第一个核心）
    try:
        pccl.utils.set_affinity(0)
        print("已设置CPU亲和性为核心0")
    except Exception as e:
        print(f"设置亲和性失败: {str(e)}")
    
    # 生成哈希值
    print(f"主机哈希: {pccl.utils.host_hash()}")
    print(f"进程哈希: {pccl.utils.pid_hash()}")

if __name__ == "__main__":
    demonstrate_utils()
    # 清理测试目录
    if os.path.exists("./test_dir"):
        os.rmdir("./test_dir")
