from enum import Enum
from typing import Optional

class BufferType(Enum):
    LIB = 0
    HOST = 1
    DEVICE = 2

class Buffer:
    def __init__(self, bufferType: BufferType, size: int):
        self.bufferType = bufferType
        self.size = size

class BufferSlice:
    def __init__(self, bufferType: BufferType, offset: int, size: int):
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if size <= 0:
            raise ValueError("size must be positive")
        
        self.bufferType = bufferType
        self.offset = offset
        self.size = size

    @property
    def end(self) -> int:
        return self.offset + self.size

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def contains_offset(self, offset: int) -> bool:
        return self.offset <= offset < self.end

    def contains_slice(self, other: 'BufferSlice') -> bool:
        if self.bufferType != other.bufferType:
            return False
        return self.offset <= other.offset and self.end >= other.end

    def intersected(self, other: 'BufferSlice') -> bool:
        if self.bufferType != other.bufferType:
            return False
        return self.offset < other.end and self.end > other.offset

    def intersection(self, other: 'BufferSlice') -> Optional['BufferSlice']:
        if not self.intersected(other):
            return None
        
        start = max(self.offset, other.offset)
        end = min(self.end, other.end)
        return BufferSlice(self.bufferType, start, end - start)

    def adjacent(self, other: 'BufferSlice') -> bool:
        if self.bufferType != other.bufferType:
            return False
        return self.end == other.offset or other.end == self.offset

    def can_merge(self, other: 'BufferSlice') -> bool:
        return self.intersected(other) or self.adjacent(other)

    def merge(self, other: 'BufferSlice') -> 'BufferSlice':
        if not self.can_merge(other):
            raise ValueError("Can not merge non-intersected and non-adjacent slices")
        
        start = min(self.offset, other.offset)
        end = max(self.end, other.end)
        return BufferSlice(self.bufferType, start, end - start)

    def split(self, split_offset: int) -> tuple['BufferSlice', 'BufferSlice']:
        if not self.contains_offset(split_offset):
            raise ValueError("Split offset must be in the range of the slice")
        
        if split_offset == self.offset:
            raise ValueError("Can not split at the start of the slice")
        
        left_size = split_offset - self.offset
        right_size = self.end - split_offset
        
        left = BufferSlice(self.bufferType, self.offset, left_size)
        right = BufferSlice(self.bufferType, split_offset, right_size)
        
        return left, right

    def slice(self, start_offset: int, size: int) -> 'BufferSlice':
        if start_offset < self.offset or start_offset + size > self.end:
            raise ValueError("Slice out of range")
        
        return BufferSlice(self.bufferType, start_offset, size)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BufferSlice):
            return False
        return (self.bufferType == other.bufferType and 
                self.offset == other.offset and 
                self.size == other.size)

    def __hash__(self) -> int:
        return hash((self.bufferType, self.offset, self.size))

    def __str__(self) -> str:
        return f"BufferSlice({self.bufferType.name}, offset={self.offset}, size={self.size})"

    def __repr__(self) -> str:
        return f"BufferSlice(BufferType.{self.bufferType.name}, {self.offset}, {self.size})"
