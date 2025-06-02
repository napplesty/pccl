from typing import List, Dict

class ChunkUIDGen:
    _uid = 0
    @staticmethod
    def get_uid() -> int:
        ChunkUIDGen._uid += 1
        return ChunkUIDGen._uid
    
    @staticmethod
    def refresh():
        ChunkUIDGen._uid = 0

_named_chunk_context = None

class NamedChunkContext:
    _chunks: Dict[int, 'NamedChunk'] = {}
    def __init__(self):
        _named_chunk_context = self
        ChunkUIDGen.refresh()
    
    def get_chunk(self, uid: int) -> 'NamedChunk':
        return self._chunks[uid]
    
    def add_chunk(self, chunk: 'NamedChunk'):
        self._chunks[chunk.uid] = chunk
        
class NamedChunk:
    def __init__(self, src_rank: int, cur_rank: int, chunk_id: List[int]):
        self.uid = ChunkUIDGen.get_uid()
        self.src_rank = src_rank
        self.cur_rank = cur_rank
        self.chunk_id = chunk_id
        self.parents = []
        self.freed = False
        global _named_chunk_context
        _named_chunk_context.add_chunk(self)

    def free(self):
        assert self._is_existing(), f"Try to free a freed chunk: {self}"
        self.freed = True

    def send(self, dst:int) -> 'NamedChunk':
        assert self._is_existing(), f"Try to send a freed chunk: {self}"
        new_chunk = NamedChunk(self.src_rank, dst, self.chunk_id)
        new_chunk.parents.append(self.uid)
        return new_chunk

    def reduce(self, dst: int, chunk: 'NamedChunk') -> 'NamedChunk':
        assert self._is_existing(), f"Try to reduce with a freed chunk: {self}"
        assert chunk._is_existing(), f"Try to reduce with a freed chunk: {chunk}"
        new_chunk = NamedChunk(self.src_rank, dst, self.chunk_id)
        new_chunk.parents.extend([self.uid, chunk.uid])
        return new_chunk

    def _is_existing(self):
        return not self.freed
    
    def __str__(self):
        return f"NamedChunk({self.cur_rank})[{self.src_rank}, {self.chunk_id}]"
    
    def __repr__(self):
        return self.__str__()