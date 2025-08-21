from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput, ExtractTextOutput
)
from app.domain.services.text_normalizer import TextNormalizer, NormalizerConfig

@dataclass
class ExtractAndNormalizeInput:
    relative_path: Path
    max_pages: Optional[int] = None
    # (opcional)
    normalizer_cfg: Optional[NormalizerConfig] = None
    join_pages: bool = False  # True -> devuelve texto único; False -> por páginas

@dataclass
class ExtractAndNormalizeOutput:
    source_path: Path
    page_count: int
    normalized_pages: Optional[List[str]] = None
    normalized_text: Optional[str] = None

class ExtractAndNormalizePdf:
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: TextNormalizer | None = None,
    ) -> None:
        self.blob = blob
        self.extract_uc = ExtractTextFromPdf(blob, extractor)
        self.normalizer = normalizer or TextNormalizer()

    def execute(self, params: ExtractAndNormalizeInput) -> ExtractAndNormalizeOutput:
        # 1) extrae
        ext = self.extract_uc.execute(
            ExtractTextInput(relative_path=params.relative_path, max_pages=params.max_pages)
        )
        pages = ext.result.pages

        # 2) normaliza
        if params.normalizer_cfg:
            self.normalizer = TextNormalizer(params.normalizer_cfg)

        if params.join_pages:
            text = self.normalizer.normalize_and_join(pages)
            return ExtractAndNormalizeOutput(
                source_path=ext.source_path,
                page_count=ext.result.page_count,
                normalized_text=text,
                normalized_pages=None,
            )
        else:
            norm_pages = self.normalizer.normalize_pages(pages)
            return ExtractAndNormalizeOutput(
                source_path=ext.source_path,
                page_count=ext.result.page_count,
                normalized_pages=norm_pages,
                normalized_text=None,
            )
