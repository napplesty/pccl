from typing import Dict, Any, Optional
import logging

from .base import Pass
from ..ir.core import CollectiveIR
from ..transforms import HighToLowConverter
from ..low_ir.core import LowLevelIR

logger = logging.getLogger(__name__)

class LoweringPass(Pass):
    """将高级IR转换为低级IR的Pass"""
    
    def __init__(self, converter: Optional[HighToLowConverter] = None):
        """初始化LoweringPass
        
        Args:
            converter: 高级到低级IR转换器，如果为None则创建默认转换器
        """
        self.converter = converter or HighToLowConverter()
        self._result: Optional[LowLevelIR] = None
    
    def run(self, high_ir: CollectiveIR, context: Dict[str, Any] = None) -> LowLevelIR:
        """运行转换Pass
        
        Args:
            high_ir: 高级IR
            context: 上下文信息
            
        Returns:
            转换后的低级IR
        """
        logger.info(f"开始LoweringPass: {high_ir.name}")
        
        # 转换高级IR到低级IR
        low_ir = self.converter.convert(high_ir)
        
        # 保存结果
        self._result = low_ir
        
        # 将结果存入上下文
        if context is not None:
            context['low_ir'] = low_ir
        
        logger.info(f"LoweringPass完成: {high_ir.name} -> {low_ir.name}")
        return low_ir
    
    @property
    def result(self) -> Optional[LowLevelIR]:
        """返回转换结果"""
        return self._result
