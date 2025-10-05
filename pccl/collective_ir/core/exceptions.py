class CollectiveIRError(Exception):
    """Collective IR 基础异常类"""
    pass

class DeviceNotFoundError(CollectiveIRError):
    """设备未找到异常"""
    pass

class InvalidTopologyError(CollectiveIRError):
    """无效拓扑异常"""
    pass

class CircularDependencyError(CollectiveIRError):
    """循环依赖异常"""
    pass

class IRVerificationError(CollectiveIRError):
    """IR验证异常"""
    pass

class PassExecutionError(CollectiveIRError):
    """Pass执行异常"""
    pass
