from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.domain.services.text_normalizer import TextNormalizer, NormalizerConfig
from app.domain.services.chunker_global import GlobalTokenChunker, GlobalChunkerConfig, GlobalChunk
from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput
)

@dataclass
class ExtractNormalizeChunkGlobalInput:
    relative_path: Path
    max_pages: Optional[int] = None
    normalizer_cfg: Optional[NormalizerConfig] = None
    chunker_cfg: Optional[GlobalChunkerConfig] = None

@dataclass
class ExtractNormalizeChunkGlobalOutput:
    source_path: Path
    page_count: int
    chunks: List[GlobalChunk]
    full_text: str  # texto normalizado unido (por si quieres guardarlo / debug)

class ExtractNormalizeChunkGlobalPdf:
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: TextNormalizer,
        chunker: GlobalTokenChunker,
    ) -> None:
        self.blob = blob
        self.extract_uc = ExtractTextFromPdf(blob, extractor)
        self.normalizer = normalizer
        self.chunker = chunker

    def execute(self, params: ExtractNormalizeChunkGlobalInput) -> ExtractNormalizeChunkGlobalOutput:
        # 1) extrae
        ext = self.extract_uc.execute(ExtractTextInput(
            relative_path=params.relative_path,
            max_pages=params.max_pages
        ))
        pages = ext.result.pages

        # 2) normaliza
        if params.normalizer_cfg:
            self.normalizer = TextNormalizer(params.normalizer_cfg)
        norm_pages = self.normalizer.normalize_pages(pages)

        # 3) chunking global
        if params.chunker_cfg:
            self.chunker = GlobalTokenChunker(self.chunker.tok, params.chunker_cfg)
        chunks, full_text = self.chunker.chunk_document(norm_pages)

        return ExtractNormalizeChunkGlobalOutput(
            source_path=ext.source_path,
            page_count=ext.result.page_count,
            chunks=chunks,
            full_text=full_text,
        )
