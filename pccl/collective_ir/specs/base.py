from typing import List, Set, Dict, Any
from ..core.ir import CollectiveSpec, Precondition, Postcondition, ChunkState
from ..core.enums import CollectiveOpType

class CollectiveSpecBuilder:
    def __init__(self, op_type: CollectiveOpType):
        self.op_type = op_type
        self.preconditions = []
        self.postconditions = []
        self.involved_devices = set()
        self.data_size_gb = 1.0
        self.metadata = {}

    def add_precondition(self, precondition: Precondition):
        self.preconditions.append(precondition)
        return self

    def add_postcondition(self, postcondition: Postcondition):
        self.postconditions.append(postcondition)
        return self

    def set_devices(self, devices: Set[int]):
        self.involved_devices = devices
        return self

    def set_data_size(self, size_gb: float):
        self.data_size_gb = size_gb
        return self

    def add_metadata(self, key: str, value: Any):
        self.metadata[key] = value
        return self

    def build(self) -> CollectiveSpec:
        return CollectiveSpec(
            op_type=self.op_type,
            preconditions=self.preconditions,
            postconditions=self.postconditions,
            data_size_gb=self.data_size_gb,
            involved_devices=self.involved_devices,
        )

class SpecValidator:
    @staticmethod
    def validate_spec(spec: CollectiveSpec) -> bool:
        if not spec.preconditions or not spec.postconditions:
            return False
        
        if not spec.involved_devices:
            return False
        
        if spec.data_size_gb <= 0:
            return False
        
        return True

    @staticmethod
    def check_spec_consistency(spec: CollectiveSpec) -> List[str]:
        issues = []
        
        pre_devices = set()
        post_devices = set()
        
        for pre in spec.preconditions:
            pre_devices.update(pre.required_devices)
        
        for post in spec.postconditions:
            post_devices.update(post.produced_devices)
        
        if pre_devices != spec.involved_devices:
            issues.append(f"Precondition devices {pre_devices} don't match involved devices {spec.involved_devices}")
        
        if post_devices != spec.involved_devices:
            issues.append(f"Postcondition devices {post_devices} don't match involved devices {spec.involved_devices}")
        
        return issues
