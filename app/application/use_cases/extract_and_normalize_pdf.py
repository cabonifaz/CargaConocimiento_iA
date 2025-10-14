from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import pathlib

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput, ExtractTextOutput
)
from app.domain.services.text_normalizer import TextNormalizer, NormalizerConfig
from app.domain.services.md_text_normalizer import MdTextNormalizer, MdNormalizerConfig
from app.domain.services.ocr_md_text_normalizer import OcrMdTextNormalizer

@dataclass
class ExtractAndNormalizeInput:
    relative_path: Path
    max_pages: Optional[int] = None
    # (opcional)
    normalizer_cfg: Optional[NormalizerConfig | MdNormalizerConfig] = None
    join_pages: bool = False  # True -> devuelve texto único; False -> por páginas
    generate_report: Optional[bool] = False 

@dataclass
class ExtractAndNormalizeOutput:
    source_path: Path
    page_count: int
    normalized_pages: Optional[List[str]] = None
    normalized_text: Optional[str] = None
    # Add original extracted pages for detailed reporting
    extracted_pages: Optional[List[str]] = None

class ExtractAndNormalizePdf:
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: TextNormalizer | MdTextNormalizer | None = None,
    ) -> None:
        self.blob = blob
        self.extract_uc = ExtractTextFromPdf(blob, extractor)
        self.normalizer = normalizer or TextNormalizer()

    def execute(self, params: ExtractAndNormalizeInput) -> ExtractAndNormalizeOutput:
        if isinstance(self.normalizer, OcrMdTextNormalizer):
            print("[NORMALIZACION] Usando OcrMdTextNormalizer para Mistral OCR markdown")
        elif isinstance(self.normalizer, MdTextNormalizer):
            print("[NORMALIZACION] Usando MdTextNormalizer para formato Markdown")
        # 1) extrae
        ext = self.extract_uc.execute(
            ExtractTextInput(relative_path=params.relative_path, max_pages=params.max_pages, generate_report=params.generate_report)
        )
        pages = ext.result.pages

        # 2) normaliza
        if params.normalizer_cfg:
            if isinstance(params.normalizer_cfg, NormalizerConfig):
                self.normalizer = TextNormalizer(params.normalizer_cfg)
            elif isinstance(params.normalizer_cfg, MdNormalizerConfig):
                self.normalizer = MdTextNormalizer(params.normalizer_cfg)

        if params.join_pages:
            text = self.normalizer.normalize_and_join(pages)

            return ExtractAndNormalizeOutput(
                source_path=ext.source_path,
                page_count=ext.result.page_count,
                normalized_text=text,
                normalized_pages=None,
                extracted_pages=pages,
            )
        else:
            norm_pages = self.normalizer.normalize_pages(pages)

            if params.generate_report:
                target = pathlib.Path(params.relative_path)
                report_dir = Path("report/normalization")
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / f"{target.stem}_normalization_report.txt"
                with report_path.open("w", encoding="utf-8") as report_file:
                    report_file.write(f"Reporte de normalización para: {target}\n")
                    report_file.write(f"Páginas normalizadas: {len(norm_pages)}\n\n")
                    for i, page in enumerate(norm_pages):
                        report_file.write(f"--- Página {i+1} ---\n")
                        report_file.write(page + "\n\n")
                print(f"Reporte de normalización guardado en: {report_path}")

            return ExtractAndNormalizeOutput(
                source_path=ext.source_path,
                page_count=ext.result.page_count,
                normalized_pages=norm_pages,
                normalized_text=None,
                extracted_pages=pages,
            )
