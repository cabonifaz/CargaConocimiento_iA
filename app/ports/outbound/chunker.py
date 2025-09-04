from __future__ import annotations
from typing import Protocol, List, Tuple, Union, Sequence
from dataclasses import dataclass

from app.domain.services.chunker_global import GlobalChunk, GlobalChunkerConfig
from app.domain.services.chunker_global_md import GlobalChunkMd, GlobalChunkerConfigMd
from app.ports.outbound.tokenizer import TokenCounterPort

@dataclass(frozen=True)
class ChunkerConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 50
    # Separador entre páginas en el texto unido (no es visible al usuario final).
    # Usa algo poco probable en el contenido para que el mapeo sea estable.
    page_separator: str = "\n\n\f\n\n"  # \f = form feed, útil como marcador
    
@dataclass(frozen=True)
class ChunkType:
    text: str
    token_count: int
    chunk_id: str
    # offsets en el texto UNIDO
    char_start: int
    char_end: int
    # páginas 1-based que cubre este chunk (derivadas del mapa)
    page_start: int
    page_end: int

class ChunkerPort(Protocol):
    """Interface for document chunking strategies."""
    
    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None) -> None:
        """Initialize chunker with tokenizer and configuration."""
        ...
    
    def chunk_document(self, pages: List[str]) -> Tuple[Sequence[ChunkType], str]:
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