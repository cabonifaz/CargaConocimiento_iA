from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class WeaviateChunkMetadata:
    """Metadata model for chunks stored in Weaviate following the new schema structure."""

    # ==== Texto para búsqueda semántica/híbrida ====
    text: str                           # Contenido completo del chunk con jerarquía
    bm25_text: str                      # Contenido limpio sin headers para BM25
    doc_title: str                      # Título del documento
    section_title: str                  # Título de la sección más profunda

    # ==== Identificadores y organización ====
    doc_id: str                         # ID del documento
    company_id: str                     # ID string de la empresa (sin espacios)
    company: str                        # Nombre textual de la empresa
    area_id: str                        # ID string del área (sin espacios)
    area: str                           # Nombre textual del área
    section_path: List[str]             # Ruta jerárquica de secciones

    # ==== Posición en el documento ====
    page_start: int                     # Página inicial (1-based)
    page_end: int                       # Página final (1-based)

    # ==== Información de embedding / trazabilidad ====
    embedding_model: str                # Modelo usado para generar embeddings
    embedding_dim: int                  # Dimensión del vector de embedding

    # ==== Opcionales útiles ====
    chunk_id: str                       # Hash único del chunk
    token_count: int                    # Número de tokens
    char_start: int                     # Posición inicial en caracteres
    char_end: int                       # Posición final en caracteres
    ingested_at: str                    # Timestamp de ingesta (ISO format)

    def to_weaviate_properties(self) -> dict:
        """Convert to Weaviate properties dictionary."""
        return {
            # ==== Texto para búsqueda semántica/híbrida ====
            "text": self.text,
            "bm25_text": self.bm25_text,
            "doc_title": self.doc_title,
            "section_title": self.section_title,

            # ==== Identificadores y organización ====
            "doc_id": self.doc_id,
            "company_id": self.company_id,
            "company": self.company,
            "area_id": self.area_id,
            "area": self.area,
            "section_path": self.section_path,

            # ==== Posición en el documento ====
            "page_start": self.page_start,
            "page_end": self.page_end,

            # ==== Información de embedding / trazabilidad ====
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,

            # ==== Opcionales útiles ====
            "chunk_id": self.chunk_id,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "ingested_at": self.ingested_at,
        }


@dataclass(frozen=True)
class CompanyInfo:
    """Information about a company for metadata."""
    id: str
    name: str


@dataclass(frozen=True)
class AreaInfo:
    """Information about an area for metadata."""
    id: str
    name: str


@dataclass(frozen=True)
class DocumentInfo:
    """Information about a document for metadata."""
    id: str
    title: str
    company: CompanyInfo
    area: AreaInfo