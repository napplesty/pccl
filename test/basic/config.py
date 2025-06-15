import pccl

def print_config():
    """打印PCCL配置参数"""
    print("PCCL Runtime Configuration:")
    print(f"{'DEVICE_BUFFER_SIZE':<25}: {pccl.Config.DEVICE_BUFFER_SIZE}")
    print(f"{'HOST_BUFFER_SIZE':<25}: {pccl.Config.HOST_BUFFER_SIZE}")
    print(f"{'SLOT_GRANULARITY':<25}: {pccl.Config.SLOT_GRANULARITY}")
    print(f"{'PROXY_FLUSH_PERIOD':<25}: {pccl.Config.PROXY_FLUSH_PERIOD}")
    print(f"{'PROXY_MAX_FLUSH_SIZE':<25}: {pccl.Config.PROXY_MAX_FLUSH_SIZE}")
    print(f"{'PROXY_CHECK_STOP_PERIOD':<25}: {pccl.Config.PROXY_CHECK_STOP_PERIOD}")

if __name__ == "__main__":
    print_config()
