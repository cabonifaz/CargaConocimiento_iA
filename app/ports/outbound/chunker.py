from __future__ import annotations
from typing import Protocol, List, Tuple, Union
from dataclasses import dataclass

from app.domain.services.chunker_global import GlobalChunk, GlobalChunkerConfig
from app.domain.services.chunker_global_md import GlobalChunkMd, GlobalChunkerConfigMd
from app.ports.outbound.tokenizer import TokenCounterPort

# Type alias for chunk configurations
ChunkerConfig = Union[GlobalChunkerConfig, GlobalChunkerConfigMd]
# Type alias for chunk types  
ChunkType = Union[GlobalChunk, GlobalChunkMd]

class ChunkerPort(Protocol):
    """Interface for document chunking strategies."""
    
    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None) -> None:
        """Initialize chunker with tokenizer and configuration."""
        ...
    
    def chunk_document(self, pages: List[str]) -> Tuple[List[ChunkType], str]:
        """Chunk a document into pieces.
        
        Args:
            pages: List of text pages to chunk
            
        Returns:
            Tuple of (list of chunks, full_text)
        """
        ...
    
    def get_config(self) -> ChunkerConfig:
        """Get the current chunker configuration."""
        ...