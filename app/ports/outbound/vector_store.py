# src/app/ports/outbound/vector_store.py
from __future__ import annotations
from typing import Protocol, List, Optional

class VectorStorePort(Protocol):
    def upsert_chunks(self, doc_id: str, chunks: List, vectors: List[list[float]], company_id: str = "default_company", collection_name: Optional[str] = None) -> int:
        ...
