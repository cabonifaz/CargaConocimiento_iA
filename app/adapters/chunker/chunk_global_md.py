from __future__ import annotations
from typing import List, Tuple, Union, Sequence

from app.ports.outbound.chunker import ChunkerPort
from app.ports.outbound.tokenizer import TokenCounterPort
from app.domain.services.chunker_global_md import GlobalTokenChunkerMd, GlobalChunkerConfigMd, GlobalChunkMd
from app.domain.services.chunker_global import GlobalChunk

class ChunkGlobalMdAdapter(ChunkerPort):
    """Adapter for Markdown-aware global chunking strategy."""
    
    def __init__(self, tokenizer: TokenCounterPort, cfg: GlobalChunkerConfigMd | None = None) -> None:
        """Initialize markdown-aware chunker with tokenizer and configuration."""
        self._chunker = GlobalTokenChunkerMd(tokenizer, cfg)
    
    def chunk_document(self, pages: List[str]) -> Tuple[Sequence[GlobalChunkMd], str]:
        """Chunk a document into markdown-aware pieces.
        
        Args:
            pages: List of text pages to chunk
            
        Returns:
            Tuple of (list of markdown chunks, full_text)
        """
        return self._chunker.chunk_document(pages)
    
    def get_config(self) -> GlobalChunkerConfigMd:
        """Get the current chunker configuration."""
        return self._chunker.cfg