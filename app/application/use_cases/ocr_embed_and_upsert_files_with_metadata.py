# app/application/use_cases/ocr_embed_and_upsert_files_with_metadata.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.ports.outbound.embedder import EmbedderPort
from app.ports.outbound.vector_store import VectorStorePort

from app.domain.services.ocr_md_text_normalizer import OcrMdTextNormalizer
from app.ports.outbound.chunker import ChunkerPort, ChunkerConfig
from app.domain.services.chunk_quality import QualityConfig

from app.application.use_cases.embed_chunks_from_pdf import (
    EmbedChunksFromPdf, EmbedChunksFromPdfInput
)
from app.config.settings import settings

@dataclass
class OcrFileWithMetadataReport:
    file_path: Path
    success: bool
    chunks_written: int
    doc_id: str
    error_message: Optional[str] = None

@dataclass
class OcrEmbedAndUpsertFilesWithMetadataInput:
    company_id: str  # String ID without whitespace
    area_id: str     # String ID without whitespace
    max_pages: Optional[int] = None
    chunker_cfg: Optional[ChunkerConfig] = None
    quality_cfg: Optional[QualityConfig] = None
    embedding_model: str = "cohere.embed-multilingual-v3"

@dataclass
class OcrEmbedAndUpsertFilesWithMetadataOutput:
    company_id: str
    area_id: str
    total_files: int
    successful_files: int
    total_chunks_written: int
    reports: List[OcrFileWithMetadataReport]

class OcrEmbedAndUpsertFilesWithMetadata:
    """
    Use case para procesar archivos PDF usando Mistral OCR, el normalizer especializado
    para tablas con listas, chunker semántico optimizado, y embeddings con Bedrock Cohere.
    """
    def __init__(
        self,
        blob: BlobStoragePort,
        mistral_extractor: TextExtractorPort,  # MistralOCRTextExtractor
        ocr_normalizer: OcrMdTextNormalizer,
        chunker: ChunkerPort,  # ChunkSemanticOptimizedAdapter
        embedder: EmbedderPort,  # BedrockCohereEmbedMultilingual
        vector_store: VectorStorePort,  # WeaviateVectorStore
    ) -> None:
        self.embed_uc = EmbedChunksFromPdf(blob, mistral_extractor, ocr_normalizer, chunker, embedder)
        self.vector_store = vector_store
        self.blob = blob

    def execute(self, params: OcrEmbedAndUpsertFilesWithMetadataInput) -> OcrEmbedAndUpsertFilesWithMetadataOutput:
        # Get all PDF files from company_files/files/
        files_folder = Path("company_files") / "files"
        pdf_files = self._get_pdf_files(files_folder)

        reports: List[OcrFileWithMetadataReport] = []
        total_chunks = 0
        successful_count = 0

        print(f"\n{'=' * 80}")
        print(f"🏢 PROCESANDO ARCHIVOS CON MISTRAL OCR")
        print(f"   Empresa ID: {params.company_id} | Área ID: {params.area_id}")
        print(f"📂 Archivos encontrados: {len(pdf_files)}")
        print(f"{'=' * 80}")

        for pdf_file in pdf_files:
            try:
                print(f"\n📁 [PROCESANDO CON OCR] {pdf_file.name} (empresa_id: {params.company_id}, área_id: {params.area_id})")
                # Use relative path from company_files
                relative_path = Path("files") / pdf_file.name

                # First get embeddings (using Mistral OCR extraction)
                embed_result = self.embed_uc.execute(EmbedChunksFromPdfInput(
                    relative_path=relative_path,
                    max_pages=params.max_pages,
                    chunker_cfg=params.chunker_cfg,
                    quality_cfg=params.quality_cfg,
                ))

                # Then upsert to collection using company_id directly as collection name
                doc_id = pdf_file.stem
                collection_name = params.company_id  # Use company_id directly as collection name

                written = self.vector_store.upsert_chunks(
                    doc_id=doc_id,
                    chunks=embed_result.chunks,  # type: ignore[arg-type]
                    vectors=embed_result.vectors,
                    company_id=params.company_id,
                    company="company_name",  # Default value
                    area_id=params.area_id,
                    area=params.area_id,     # Use area_id value
                    doc_title=pdf_file.stem,  # Use filename as doc_title
                    embedding_model=settings.BEDROCK_MODEL_ID,  # Get from environment
                    collection_name=collection_name
                )

                reports.append(OcrFileWithMetadataReport(
                    file_path=pdf_file,
                    success=True,
                    chunks_written=written,
                    doc_id=doc_id,
                ))

                total_chunks += written
                successful_count += 1

                print(f"✅ [COMPLETADO] {pdf_file.name} → {written} chunks procesados con OCR y almacenados")

            except Exception as e:
                reports.append(OcrFileWithMetadataReport(
                    file_path=pdf_file,
                    success=False,
                    chunks_written=0,
                    doc_id="",
                    error_message=str(e),
                ))
                print(f"❌ [ERROR] {pdf_file.name} → {str(e)}")

        return OcrEmbedAndUpsertFilesWithMetadataOutput(
            company_id=params.company_id,
            area_id=params.area_id,
            total_files=len(pdf_files),
            successful_files=successful_count,
            total_chunks_written=total_chunks,
            reports=reports,
        )

    def _get_pdf_files(self, files_folder: Path) -> List[Path]:
        """Get all PDF files from the files folder."""
        if not files_folder.exists():
            return []

        pdf_files: List[Path] = []
        for file_path in files_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                pdf_files.append(file_path)

        return sorted(pdf_files)
