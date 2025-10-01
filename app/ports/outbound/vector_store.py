# src/app/ports/outbound/vector_store.py
from __future__ import annotations
from typing import Protocol, List, Optional, Sequence
from app.ports.outbound.chunker import ChunkType

# Updated to use ChunkType from hexagonal architecture refactoring

class VectorStorePort(Protocol):
    def upsert_chunks(self,
                     doc_id: str,
                     chunks: Sequence[ChunkType],
                     vectors: List[list[float]],
                     company_id: int,
                     company: str,
                     area_id: int,
                     area: str,
                     doc_title: str = "",
                     embedding_model: str = "cohere.embed-multilingual-v3",
                     collection_name: Optional[str] = None) -> int:
        ...
