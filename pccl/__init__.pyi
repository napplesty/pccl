from typing import ClassVar, Final

class Config:
    # Kernel 相关常量
    WARP_SIZE: ClassVar[Final[int]]
    WARP_PER_SM: ClassVar[Final[int]]
    WARP_FOR_SCHEDULE: ClassVar[Final[int]]
    WARP_FOR_PROXY: ClassVar[Final[int]]
    WARP_FOR_MEMORY: ClassVar[Final[int]]
    MAX_SM_COUNT: ClassVar[Final[int]]
    DEVICE_SYNCER_SIZE: ClassVar[Final[int]]
    MAX_OPERATIONS_PER_CAPSULE: ClassVar[Final[int]]
    FIFO_BUFFER_SIZE: ClassVar[Final[int]]
    INTER_SM_FIFO_SIZE: ClassVar[Final[int]]
    
    # 网络连接相关常量
    MAX_CHANNEL_PER_OPERATION: ClassVar[Final[int]]
    MAX_ACTIVE_CONNECTIONS: ClassVar[Final[int]]
    
    # 缓冲区相关常量
    LIB_BUFFER_SIZE: ClassVar[Final[int]]
    DEVICE_BUFFER_SIZE: ClassVar[Final[int]]
    HOST_BUFFER_SIZE: ClassVar[Final[int]]
    NUM_SLOT: ClassVar[Final[int]]
    
    # IB 相关常量
    MAX_CQ_SIZE: ClassVar[Final[int]]
    MAX_CQ_POLL_NUM: ClassVar[Final[int]]
    MAX_SEND_WR: ClassVar[Final[int]]
    MAX_WR_PER_SEND: ClassVar[Final[int]]
    
    # Proxy 相关常量
    PROXY_FLUSH_PERIOD: ClassVar[Final[int]]
    PROXY_CHECK_STOP_PERIOD: ClassVar[Final[int]]

# 暴露模块接口
__all__ = ['Config']
