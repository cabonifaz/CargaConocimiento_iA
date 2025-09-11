from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.ports.outbound.chunker import ChunkerPort, ChunkerConfig, ChunkType
from app.domain.services.md_text_normalizer import MdTextNormalizer, MdNormalizerConfig
from app.application.use_cases.extract_text_from_pdf import ExtractTextFromPdf
from app.application.use_cases.extract_and_normalize_pdf import (
    ExtractAndNormalizePdf, ExtractAndNormalizeInput
)

@dataclass
class ExtractNormalizeChunkGlobalInput:
    relative_path: Path
    max_pages: Optional[int] = None
    normalizer_cfg: Optional[MdNormalizerConfig] = None
    chunker_cfg: Optional[ChunkerConfig] = None
    generate_report: Optional[bool] = False

@dataclass
class ExtractNormalizeChunkGlobalOutput:
    source_path: Path
    page_count: int
    chunks: List[ChunkType]
    full_text: str  # texto normalizado unido (por si quieres guardarlo / debug)

class ExtractNormalizeChunkGlobalPdf:
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: MdTextNormalizer,
        chunker: ChunkerPort,
    ) -> None:
        self.blob = blob
        self.normalizer = normalizer
        self.chunker = chunker
        self.extract_uc = ExtractTextFromPdf(blob, extractor)
        self.extract_normalize_uc = ExtractAndNormalizePdf(blob, extractor, normalizer)

    def execute(self, params: ExtractNormalizeChunkGlobalInput) -> ExtractNormalizeChunkGlobalOutput:
        # 1) y 2) extrae y normaliza

        norm_result = self.extract_normalize_uc.execute(ExtractAndNormalizeInput(
            relative_path=params.relative_path,
            max_pages=params.max_pages,
            normalizer_cfg=params.normalizer_cfg,
            join_pages=False,
            generate_report=params.generate_report
        ))

        norm_pages = norm_result.normalized_pages or []

        # 3) chunking global
        # Note: Configuration updates should be handled by the chunker adapter implementation
        # if needed. For now, we'll use the chunker as-is.
        chunks, full_text = self.chunker.chunk_document(norm_pages)

        if params.generate_report:
            target = Path(params.relative_path)
            report_dir = Path("report/chunking")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{target.stem}_chunking_report.txt"
            with report_path.open("w", encoding="utf-8") as report_file:
                report_file.write(f"Archivo: {norm_result.source_path}\n")
                report_file.write(f"Páginas: {norm_result.page_count}\n")
                report_file.write(f"Chunks generados: {len(chunks)}\n")
                report_file.write(f"Chunker Config: {self.chunker.get_config()}\n")
                report_file.write("\n--- Chunks ---\n\n")
                for i, chunk in enumerate(chunks):
                    report_file.write(f"--- Chunk {i+1} (tokens: {chunk.token_count}) ---\n")
                    report_file.write(chunk.text)
                    report_file.write("\n\n")

        return ExtractNormalizeChunkGlobalOutput(
            source_path=norm_result.source_path,
            page_count=norm_result.page_count,
            chunks=chunks,
            full_text=full_text,
        )
