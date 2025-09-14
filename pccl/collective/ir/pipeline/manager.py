import json
from typing import List
from .passes import Pass
from .registry import PassRegistry
from .config import PipelineConfig, PassConfig


class PipelineConfigManager:
    """管线配置管理器：加载/保存配置、生成管线实例"""
    def __init__(self, registry: PassRegistry = None):
        self.registry = registry or PassRegistry()  # 默认使用单例注册表

    def load_from_json(self, json_str: str) -> PipelineConfig:
        """从JSON字符串加载配置"""
        data = json.loads(json_str)
        return PipelineConfig(
            passes=[PassConfig(**pc) for pc in data.get("passes", [])],
            global_params=data.get("global_params", {})
        )

    def save_to_json(self, config: PipelineConfig) -> str:
        """将配置保存为JSON字符串"""
        return json.dumps({
            "passes": [pc.__dict__ for pc in config.passes],
            "global_params": config.global_params
        }, indent=2)

    def create_pipeline(self, config: PipelineConfig) -> List[Pass]:
        """根据配置生成管线实例（Pass列表）"""
        pipeline = []
        for pc in config.passes:
            # 获取Pass类
            pass_cls = self.registry.get(pc.name)
            # 实例化Pass（传递参数）
            try:
                pass_instance = pass_cls(**pc.params)
            except TypeError as e:
                raise ValueError(f"Invalid parameters for Pass {pc.name}: {e}")
            pipeline.append(pass_instance)
        return pipeline
    