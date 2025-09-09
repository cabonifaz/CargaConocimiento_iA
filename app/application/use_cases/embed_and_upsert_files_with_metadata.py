# app/application/use_cases/embed_and_upsert_files_with_metadata.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.ports.outbound.embedder import EmbedderPort
from app.ports.outbound.vector_store import VectorStorePort

from app.domain.services.md_text_normalizer import MdTextNormalizer
from app.ports.outbound.chunker import ChunkerPort, ChunkerConfig
from app.domain.services.chunk_quality import QualityConfig

from app.application.use_cases.embed_chunks_from_pdf import (
    EmbedChunksFromPdf, EmbedChunksFromPdfInput
)

@dataclass
class FileWithMetadataReport:
    file_path: Path
    success: bool
    chunks_written: int
    doc_id: str
    error_message: Optional[str] = None

@dataclass
class EmbedAndUpsertFilesWithMetadataInput:
    company_name: str
    area_name: str
    max_pages: Optional[int] = None
    chunker_cfg: Optional[ChunkerConfig] = None
    quality_cfg: Optional[QualityConfig] = None

@dataclass
class EmbedAndUpsertFilesWithMetadataOutput:
    company_name: str
    area_name: str
    total_files: int
    successful_files: int
    total_chunks_written: int
    reports: List[FileWithMetadataReport]

class EmbedAndUpsertFilesWithMetadata:
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: MdTextNormalizer,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
    ) -> None:
        self.embed_uc = EmbedChunksFromPdf(blob, extractor, normalizer, chunker, embedder)
        self.vector_store = vector_store
        self.blob = blob

    def execute(self, params: EmbedAndUpsertFilesWithMetadataInput) -> EmbedAndUpsertFilesWithMetadataOutput:
        # Get all PDF files from company_files/files/
        files_folder = Path("company_files") / "files"
        pdf_files = self._get_pdf_files(files_folder)
        
        reports: List[FileWithMetadataReport] = []
        total_chunks = 0
        successful_count = 0

        for pdf_file in pdf_files:
            try:
                print(f"🕑 Procesando archivo {pdf_file.name} para company_id={params.company_name} area={params.area_name}...")
                # Use relative path from company_files
                relative_path = Path("files") / pdf_file.name
                
                # First get embeddings
                embed_result = self.embed_uc.execute(EmbedChunksFromPdfInput(
                    relative_path=relative_path,
                    max_pages=params.max_pages,
                    chunker_cfg=params.chunker_cfg,
                    quality_cfg=params.quality_cfg,
                ))

                # Then upsert to collection with company name as collection and specified metadata
                doc_id = pdf_file.stem
                written = self.vector_store.upsert_chunks(
                    doc_id=doc_id,
                    chunks=embed_result.chunks,  # type: ignore[arg-type]
                    vectors=embed_result.vectors,
                    company_id=params.company_name,
                    area=params.area_name,
                    collection_name=params.company_name  # Use company_name as collection name
                )

                reports.append(FileWithMetadataReport(
                    file_path=pdf_file,
                    success=True,
                    chunks_written=written,
                    doc_id=doc_id,
                ))
                
                total_chunks += written
                successful_count += 1

                print(f"✅ Procesamiento exitoso del archivo {pdf_file.name} -> {written} chunks escritos")
                
            except Exception as e:
                reports.append(FileWithMetadataReport(
                    file_path=pdf_file,
                    success=False,
                    chunks_written=0,
                    doc_id="",
                    error_message=str(e),
                ))
                print(f"❌ Error procesando {pdf_file.name}: {str(e)}")

        return EmbedAndUpsertFilesWithMetadataOutput(
            company_name=params.company_name,
            area_name=params.area_name,
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