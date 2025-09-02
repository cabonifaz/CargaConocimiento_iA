from __future__ import annotations
from typing import List, Tuple

from app.ports.outbound.chunker import ChunkerPort
from app.ports.outbound.tokenizer import TokenCounterPort
from app.domain.services.chunker_global import GlobalTokenChunker, GlobalChunkerConfig, GlobalChunk

class ChunkGlobalAdapter(ChunkerPort):
    """Adapter for basic global chunking strategy."""
    
    def __init__(self, tokenizer: TokenCounterPort, cfg: GlobalChunkerConfig | None = None) -> None:
        """Initialize basic global chunker with tokenizer and configuration."""
        self._chunker = GlobalTokenChunker(tokenizer, cfg)
    
    def chunk_document(self, pages: List[str]) -> Tuple[List[GlobalChunk], str]:
        """Chunk a document into basic pieces.
        
        Args:
            pages: List of text pages to chunk
            
        Returns:
            Tuple of (list of basic chunks, full_text)
        """
        return self._chunker.chunk_document(pages)
    
    def get_config(self) -> GlobalChunkerConfig:
        """Get the current chunker configuration."""
        return self._chunker.cfg