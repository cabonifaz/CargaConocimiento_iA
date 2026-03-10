"""Metadata model for chunks stored in Weaviate."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class WeaviateChunkMetadata:
    """
    Metadata model for a single chunk stored in Weaviate.

    All fields map directly to Weaviate properties. The vector itself is
    passed separately to the upsert call and is not part of this model.
    """

    # ── Text content ───────────────────────────────────────────────────────
    text: str           # Full chunk text for semantic search
    bm25_text: str      # Keyword-optimised text for BM25 (guaranteed non-empty)
    doc_title: str
    section_title: str

    # ── Document / organisation identifiers ───────────────────────────────
    doc_id: str         # "CONOC-{id_documento}" — used as UUID namespace
    company_id: str     # str(id_empresa)
    company: str        # same as company_id
    area_id: str        # str(id_area)
    area: str           # same as area_id
    section_path: List[str]

    # ── Position in the source document ───────────────────────────────────
    page_start: int
    page_end: int

    # ── Embedding / traceability ───────────────────────────────────────────
    embedding_model: str
    embedding_dim: int

    # ── Technical metadata ─────────────────────────────────────────────────
    chunk_id: str
    token_count: int
    char_start: int
    char_end: int
    ingested_at: str    # ISO-8601 timestamp

    # ── Process traceability ───────────────────────────────────────────────
    id_documento: int   # RAG_INGESTA_DOCUMENTOS.ID_DOCUMENTO
    id_proceso: int     # RAG_INGESTA_PROCESOS.ID_PROCESO

    def to_weaviate_properties(self) -> dict:
        """Return a plain dict for use in Weaviate DataObject.properties."""
        return {
            "text": self.text,
            "bm25_text": self.bm25_text,
            "doc_title": self.doc_title,
            "section_title": self.section_title,
            "doc_id": self.doc_id,
            "company_id": self.company_id,
            "company": self.company,
            "area_id": self.area_id,
            "area": self.area,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "chunk_id": self.chunk_id,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "ingested_at": self.ingested_at,
            "id_documento": self.id_documento,
            "id_proceso": self.id_proceso,
        }
