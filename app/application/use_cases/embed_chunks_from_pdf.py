from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import json

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.ports.outbound.embedder import EmbedderPort

from app.domain.services.md_text_normalizer import MdTextNormalizer
from app.domain.services.text_normalizer import TextNormalizer
from app.domain.services.chunker_global import GlobalTokenChunker, GlobalChunkerConfig, GlobalChunk
from app.domain.services.chunker_global_md import GlobalTokenChunkerMd, GlobalChunkerConfigMd, GlobalChunkMd
from app.domain.services.chunk_quality import ChunkQuality, QualityConfig

from app.application.use_cases.extract_normalize_chunk_global_pdf import (
    ExtractNormalizeChunkGlobalPdf, ExtractNormalizeChunkGlobalInput
)

@dataclass
class EmbedChunksFromPdfInput:
    relative_path: Path
    max_pages: Optional[int] = None
    chunker_cfg: Optional[GlobalChunkerConfig | GlobalChunkerConfigMd] = None
    quality_cfg: Optional[QualityConfig] = None
    batch_size: Optional[int] = None  # si quieres sobreescribir el batch local
    generate_report: Optional[bool] = False

@dataclass
class EmbedChunksFromPdfOutput:
    source_path: Path
    page_count: int
    used_text_count: int
    vectors: List[list[float]]
    chunks: Sequence[GlobalChunk] | Sequence[GlobalChunkMd] # alineados con 'vectors'

class EmbedChunksFromPdf:
    """Orquesta del pipeline hasta obtener embeddings (sin upsert)."""
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: MdTextNormalizer,
        chunker: GlobalTokenChunker | GlobalTokenChunkerMd,
        embedder: EmbedderPort,
    ) -> None:
        self.extract_norm_chunk = ExtractNormalizeChunkGlobalPdf(blob, extractor, normalizer, chunker)
        self.embedder = embedder

    def execute(self, params: EmbedChunksFromPdfInput) -> EmbedChunksFromPdfOutput:
        # 1) pipeline hasta chunks globales
        out = self.extract_norm_chunk.execute(ExtractNormalizeChunkGlobalInput(
            relative_path=params.relative_path,
            max_pages=params.max_pages,
            chunker_cfg=params.chunker_cfg,
            generate_report=params.generate_report
        ))
        chunks = out.chunks

        # 2) quality gate
        quality = ChunkQuality(params.quality_cfg) if params.quality_cfg else ChunkQuality()

        print("### Chunk Quality - good -----------")
        good = [c for c in chunks if quality.good(c)]
        print("\n---")

        # Filtrar solo instancias de GlobalChunk
        good_global_chunks = [c for c in good if isinstance(c, GlobalChunkMd)]

        # 3) embeddings (solo texto)
        texts = [c.text for c in good_global_chunks]
        print(f"--------- Texts: {len(texts)} ---------")
        vectors = self.embedder.embed_texts(texts)
        
        print("### Embedding -----------")
        print(f"🟢 Embeddings generados: {len(vectors)} / {len(chunks)} chunks (de {out.page_count} páginas, {out.source_path})")

        if params.generate_report:
            path = Path(f"report/embedding/embeddings.json")
            path.parent.mkdir(parents=True, exist_ok=True)  # crea report/embedding si no existen
            path.write_text(json.dumps(vectors), encoding="utf-8")

        return EmbedChunksFromPdfOutput(
            source_path=out.source_path,
            page_count=out.page_count,
            used_text_count=len(texts),
            vectors=vectors,
            chunks=good_global_chunks,
        )
