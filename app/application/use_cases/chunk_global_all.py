from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import json

from app.ports.outbound.blob_storage import BlobStoragePort, FileInfo
from app.ports.outbound.text_extractor import TextExtractorPort
from app.domain.services.md_text_normalizer import MdTextNormalizer
from app.domain.services.chunker_global import GlobalTokenChunker, GlobalChunkerConfig, GlobalChunk
from app.domain.services.chunker_global_md import GlobalTokenChunkerMd, GlobalChunkerConfigMd, GlobalChunkMd
from app.domain.services.chunk_quality import ChunkQuality, QualityConfig
from app.application.use_cases.extract_normalize_chunk_global_pdf import (
    ExtractNormalizeChunkGlobalPdf, ExtractNormalizeChunkGlobalInput
)
from app.config.settings import settings

@dataclass
class ChunkGlobalAllInput:
    max_pages: Optional[int] = None
    recursive: bool = True
    chunker_cfg: Optional[GlobalChunkerConfig] = None
    quality_cfg: Optional[QualityConfig] = None
    dry_run: bool = True  # por ahora solo pre-embedding
    report_jsonl: bool = True

@dataclass
class FileReport:
    file_path: Path
    pages: int
    chunks_total: int
    chunks_accepted: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    warnings: List[str]

@dataclass
class ChunkGlobalAllOutput:
    reports: List[FileReport]
    report_path: Optional[Path]

class ChunkGlobalAll:
    def __init__(self, blob: BlobStoragePort, extractor: TextExtractorPort,
                 normalizer: MdTextNormalizer, chunker: GlobalTokenChunker) -> None:
        self.blob = blob
        self.extract_normalize_chunk = ExtractNormalizeChunkGlobalPdf(blob, extractor, normalizer, chunker)

    def execute(self, params: ChunkGlobalAllInput) -> ChunkGlobalAllOutput:
        files: List[FileInfo] = list(self.blob.list_pdfs(
            base_dir=getattr(self.blob, "base_dir"),
            recursive=params.recursive
        ))

        quality = ChunkQuality(params.quality_cfg) if params.quality_cfg else ChunkQuality()
        reports: List[FileReport] = []
        lines: List[str] = []

        for f in files:
            try:
                out = self.extract_normalize_chunk.execute(
                    ExtractNormalizeChunkGlobalInput(
                        relative_path=f.path,
                        max_pages=params.max_pages,
                        chunker_cfg=params.chunker_cfg
                    )
                )
                chunks = out.chunks
                good: List[GlobalChunk | GlobalChunkMd] = [c for c in chunks if quality.good(c)]  # filtro simple

                toks = [c.token_count for c in good] or [0]
                report = FileReport(
                    file_path=out.source_path,
                    pages=out.page_count,
                    chunks_total=len(chunks),
                    chunks_accepted=len(good),
                    avg_tokens=(sum(toks)/len(toks)) if toks else 0.0,
                    min_tokens=min(toks),
                    max_tokens=max(toks),
                    warnings=self._warnings(out, chunks, good),
                )
                reports.append(report)

                # línea JSONL
                line = {
                    "file": str(report.file_path),
                    "pages": report.pages,
                    "chunks_total": report.chunks_total,
                    "chunks_accepted": report.chunks_accepted,
                    "avg_tokens": report.avg_tokens,
                    "min_tokens": report.min_tokens,
                    "max_tokens": report.max_tokens,
                    "warnings": report.warnings,
                }
                lines.append(json.dumps(line, ensure_ascii=False))

                # Aquí podrías guardar temporalmente los chunks filtrados `good`
                # para el siguiente paso (embedding), si no es dry-run. Por ahora, dry-run.
            except Exception as e:
                # reporta error y continua
                line = {"file": str(f.path), "error": str(e)}
                lines.append(json.dumps(line, ensure_ascii=False))

        report_path: Optional[Path] = None
        if params.report_jsonl:
            settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            report_path = settings.OUTPUT_DIR / f"chunk-global-report-{ts}.jsonl"
            report_path.write_text("\n".join(lines), encoding="utf-8")

        return ChunkGlobalAllOutput(reports=reports, report_path=report_path)

    from typing import List, Any

    def _warnings(self, out, chunks: List[Any], good: List[Any]) -> List[str]:
        warns = []
        if out.page_count == 0:
            warns.append("extracción vacía")
        if len(good) == 0 and len(chunks) > 0:
            warns.append("todos los chunks filtrados por calidad")
        return warns
