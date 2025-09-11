# src/app/ports/outbound/vector_store.py
from __future__ import annotations
from typing import Protocol, List, Optional, Sequence
from app.domain.services.chunker_global import GlobalChunk
from app.domain.services.chunker_global_md import GlobalChunkMd

class VectorStorePort(Protocol):
    def upsert_chunks(self, doc_id: str, chunks: Sequence[GlobalChunk] | Sequence[GlobalChunkMd], vectors: List[list[float]], company_id: str = "default_company", area: str = "VENTAS", collection_name: Optional[str] = None) -> int:
        ...
