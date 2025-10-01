# src/app/application/use_cases/embed_and_upsert_pdf.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.ports.outbound.embedder import EmbedderPort
from app.ports.outbound.vector_store import VectorStorePort

from app.domain.services.md_text_normalizer import MdTextNormalizer
from app.ports.outbound.chunker import ChunkerPort, ChunkerConfig
from app.domain.services.chunk_quality import QualityConfig

from app.application.use_cases.embed_chunks_from_pdf import (
    EmbedChunksFromPdf, EmbedChunksFromPdfInput, EmbedChunksFromPdfOutput
)

@dataclass
class EmbedAndUpsertPdfInput:
    relative_path: Path
    max_pages: Optional[int] = None
    chunker_cfg: Optional[ChunkerConfig] = None
    quality_cfg: Optional[QualityConfig] = None
    doc_id: Optional[str] = None  # si no viene, se usará el nombre de archivo (stem)

    # Company information
    company_id: str = "1"
    company: str = "default_company"

    # Area information
    area_id: str = "1"
    area: str = "GENERAL"

    # Document metadata
    doc_title: str = ""
    embedding_model: str = "cohere.embed-multilingual-v3"

    generate_report: Optional[bool] = False

@dataclass
class EmbedAndUpsertPdfOutput:
    source_path: Path
    page_count: int
    used_text_count: int
    written: int
    doc_id: str

class EmbedAndUpsertPdf:
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: MdTextNormalizer,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
    ) -> None:
        self.embedder_uc = EmbedChunksFromPdf(blob, extractor, normalizer, chunker, embedder)
        self.vs = vector_store

    def execute(self, params: EmbedAndUpsertPdfInput) -> EmbedAndUpsertPdfOutput:
        out: EmbedChunksFromPdfOutput = self.embedder_uc.execute(EmbedChunksFromPdfInput(
            relative_path=params.relative_path,
            max_pages=params.max_pages,
            chunker_cfg=params.chunker_cfg,
            quality_cfg=params.quality_cfg,
            generate_report=params.generate_report
        ))

        # doc_id por defecto = nombre de archivo (sin extensión)
        doc_id = params.doc_id or Path(out.source_path).stem

        # Use doc_title if provided, otherwise use filename
        doc_title = params.doc_title or Path(out.source_path).stem

        written = self.vs.upsert_chunks(
            doc_id=doc_id,
            chunks=out.chunks,
            vectors=out.vectors,
            company_id=params.company_id,
            company=params.company,
            area_id=params.area_id,
            area=params.area,
            doc_title=doc_title,
            embedding_model=params.embedding_model
        )

        return EmbedAndUpsertPdfOutput(
            source_path=out.source_path,
            page_count=out.page_count,
            used_text_count=out.used_text_count,
            written=written,
            doc_id=doc_id,
        )
