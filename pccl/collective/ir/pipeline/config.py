from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PassConfig:
    """单个Pass的配置（名称+参数）"""
    name: str                # Pass名称（需在注册表中存在）
    params: Dict = field(default_factory=dict)  # Pass构造函数参数


@dataclass
class PipelineConfig:
    """管线整体配置（Pass顺序+全局参数）"""
    passes: List[PassConfig] = field(default_factory=list)  # Pass配置列表
    global_params: Dict = field(default_factory=dict)        # 管线全局参数（可选）

  