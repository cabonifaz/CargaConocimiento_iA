"""Metadata model for chunks stored in Weaviate."""

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class WeaviateChunkMetadata:
    """Metadata model for chunks stored in Weaviate following the schema structure."""

    # Texto para búsqueda semántica/híbrida
    text: str
    bm25_text: str
    doc_title: str
    section_title: str

    # Identificadores y organización
    doc_id: str
    company_id: str
    company: str
    area_id: str
    area: str
    section_path: List[str]

    # Posición en el documento
    page_start: int
    page_end: int

    # Información de embedding / trazabilidad
    embedding_model: str
    embedding_dim: int

    # Opcionales útiles
    chunk_id: str
    token_count: int
    char_start: int
    char_end: int
    ingested_at: str

    def to_weaviate_properties(self) -> dict:
        """Convert to Weaviate properties dictionary."""
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
        }
