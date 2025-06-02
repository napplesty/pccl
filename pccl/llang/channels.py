from enum import Enum
from typing import List, Optional

class ChannelType(Enum):
    ib = "ib"
    ethernet = "ethernet"
    cudaipc = "cudaipc"
    nvls = "nvls"

class NamedChannel:
    def __init__(self, channelType: ChannelType, name: str):
        self.channelType: ChannelType = channelType
        self.name: str = name
        self.channel: Optional[Channel] = None
        self.connected_ranks: List[int] = []

    def __hash__(self):
        return hash((self.channelType, self.name, tuple(self.connected_ranks)))
    
    def __str__(self):
        return f"{self.channelType.name}[{self.name}]"
    
class Channel:
    def __init__(self, channelType: ChannelType, index: int, max_request_size: int, max_cq_size: int):
        self.channelType: ChannelType = channelType
        self.index: int = index
        self.max_request_size: int = max_request_size
        self.max_cq_size: int = max_cq_size
        self.connected_ranks: List[int] = []

    def __hash__(self):
        return hash((self.channelType, self.index, tuple(self.connected_ranks)))

    def __str__(self):
        return f"{self.channelType.name}[{self.index}]"
    
    def __repr__(self):
        return self.__str__()
