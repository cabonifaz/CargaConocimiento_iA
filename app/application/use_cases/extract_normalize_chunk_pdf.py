from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.domain.services.text_normalizer import TextNormalizer, NormalizerConfig
from app.domain.services.chunker import TokenChunker, ChunkerConfig, ChunkResult
from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput
)

@dataclass
class ExtractNormalizeChunkInput:
    relative_path: Path
    max_pages: Optional[int] = None
    normalizer_cfg: Optional[NormalizerConfig] = None
    chunker_cfg: Optional[ChunkerConfig] = None

@dataclass
class ExtractNormalizeChunkOutput:
    source_path: Path
    page_count: int
    chunks: List[ChunkResult]

class ExtractNormalizeChunkPdf:
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: TextNormalizer,
        chunker: TokenChunker,
    ) -> None:
        self.blob = blob
        self.extract_uc = ExtractTextFromPdf(blob, extractor)
        self.normalizer = normalizer
        self.chunker = chunker

    def execute(self, params: ExtractNormalizeChunkInput) -> ExtractNormalizeChunkOutput:
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

        # 3) chunking
        if params.chunker_cfg:
            self.chunker = TokenChunker(self.chunker.tok, params.chunker_cfg)
        chunks = self.chunker.chunk_pages(norm_pages)

        return ExtractNormalizeChunkOutput(
            source_path=ext.source_path,
            page_count=ext.result.page_count,
            chunks=chunks
        )
