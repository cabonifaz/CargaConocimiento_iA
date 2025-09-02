# src/app/ports/outbound/vector_store.py
from __future__ import annotations
from typing import Protocol, List, Optional, Sequence
from app.ports.outbound.chunker import ChunkType

# Updated to use ChunkType from hexagonal architecture refactoring

class VectorStorePort(Protocol):
    def upsert_chunks(self, doc_id: str, chunks: Sequence[ChunkType], vectors: List[list[float]], company_id: str = "default_company", area: str = "VENTAS", collection_name: Optional[str] = None) -> int:
        ...
