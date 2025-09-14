from typing import Dict, Type
from .passes import Pass


class PassRegistry:
    """单例类，管理所有可用的Pass"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._passes: Dict[str, Type[Pass]] = {}
        return cls._instance

    def register(self, name: str, pass_cls: Type[Pass]) -> None:
        """注册Pass（名称需唯一）"""
        if name in self._passes:
            raise ValueError(f"Pass {name} already registered")
        self._passes[name] = pass_cls

    def get(self, name: str) -> Type[Pass]:
        """根据名称获取Pass类"""
        pass_cls = self._passes.get(name)
        if not pass_cls:
            raise ValueError(f"Pass {name} not found in registry")
        return pass_cls

    def list_passes(self) -> Dict[str, Type[Pass]]:
        """列出所有已注册的Pass"""
        return self._passes.copy()
    