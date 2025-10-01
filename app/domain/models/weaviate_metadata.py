from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


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
    company_id: int                     # ID entero de la empresa
    company: str                        # Nombre textual de la empresa
    area_id: int                        # ID entero del área
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

    @classmethod
    def from_chunk(cls,
                   chunk,  # ChunkType
                   vector: List[float],
                   company_id: int,
                   company: str,
                   area_id: int,
                   area: str,
                   doc_id: str,
                   doc_title: str = "",
                   embedding_model: str = "cohere.embed-multilingual-v3") -> WeaviateChunkMetadata:
        """Create metadata from a chunk and additional context."""

        # Extract section information from chunk text
        section_title, section_path, bm25_text = cls._extract_section_info(chunk.text)

        return cls(
            # Texto para búsqueda
            text=chunk.text,
            bm25_text=bm25_text,
            doc_title=doc_title,
            section_title=section_title,

            # Identificadores
            doc_id=doc_id,
            company_id=company_id,
            company=company,
            area_id=area_id,
            area=area,
            section_path=section_path,

            # Posición
            page_start=chunk.page_start,
            page_end=chunk.page_end,

            # Embedding info
            embedding_model=embedding_model,
            embedding_dim=len(vector),

            # Opcionales
            chunk_id=chunk.chunk_id,
            token_count=chunk.token_count,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            ingested_at=datetime.now().isoformat(),
        )

    @staticmethod
    def _extract_section_info(text: str) -> tuple[str, List[str], str]:
        """Extract section title, section path, and BM25 text from chunk text."""
        lines = text.split('\n')
        section_title = ""
        section_path = []
        content_lines = []

        # Extract hierarchy headers and build section path
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                # Count heading level
                level = len(stripped) - len(stripped.lstrip('#'))
                heading_text = stripped.lstrip('#').strip()

                # Adjust section_path to current level
                while len(section_path) >= level:
                    section_path.pop()
                section_path.append(heading_text)

                # Use the last (deepest) heading as section_title
                section_title = heading_text
            else:
                # Collect content for BM25 (skip headers)
                if stripped and not stripped.startswith('#'):
                    content_lines.append(stripped)

        # Create BM25 text without headers (cleaner for keyword search)
        bm25_text = ' '.join(content_lines)

        return section_title, section_path, bm25_text

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
    id: int
    name: str


@dataclass(frozen=True)
class AreaInfo:
    """Information about an area for metadata."""
    id: int
    name: str


@dataclass(frozen=True)
class DocumentInfo:
    """Information about a document for metadata."""
    id: str
    title: str
    company: CompanyInfo
    area: AreaInfo