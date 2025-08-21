# src/app/application/use_cases/embed_and_upsert_company_pdfs.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.ports.outbound.embedder import EmbedderPort
from app.ports.outbound.vector_store import VectorStorePort

from app.domain.services.text_normalizer import TextNormalizer
from app.domain.services.chunker_global import GlobalTokenChunker, GlobalChunkerConfig
from app.domain.services.chunk_quality import QualityConfig

from app.application.use_cases.embed_chunks_from_pdf import (
    EmbedChunksFromPdf, EmbedChunksFromPdfInput
)

@dataclass
class CompanyFileReport:
    file_path: Path
    success: bool
    chunks_written: int
    doc_id: str
    error_message: Optional[str] = None

@dataclass
class EmbedAndUpsertCompanyPdfsInput:
    company_id: str
    max_pages: Optional[int] = None
    chunker_cfg: Optional[GlobalChunkerConfig] = None
    quality_cfg: Optional[QualityConfig] = None

@dataclass
class EmbedAndUpsertCompanyPdfsOutput:
    company_id: str
    total_files: int
    successful_files: int
    total_chunks_written: int
    reports: List[CompanyFileReport]

class EmbedAndUpsertCompanyPdfs:
    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: TextNormalizer,
        chunker: GlobalTokenChunker,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
    ) -> None:
        self.embed_uc = EmbedChunksFromPdf(blob, extractor, normalizer, chunker, embedder)
        self.vector_store = vector_store
        self.blob = blob

    def execute(self, params: EmbedAndUpsertCompanyPdfsInput) -> EmbedAndUpsertCompanyPdfsOutput:
        # Get all PDFs from company subfolder
        company_folder = Path("company_files") / params.company_id
        pdf_files = self._get_pdf_files(company_folder)
        
        reports: List[CompanyFileReport] = []
        total_chunks = 0
        successful_count = 0

        for pdf_file in pdf_files:
            try:
                # Use relative path from company_files
                relative_path = Path(params.company_id) / pdf_file.name
                
                # First get embeddings
                embed_result = self.embed_uc.execute(EmbedChunksFromPdfInput(
                    relative_path=relative_path,
                    max_pages=params.max_pages,
                    chunker_cfg=params.chunker_cfg,
                    quality_cfg=params.quality_cfg,
                ))
                
                # Then upsert to collection with company name
                doc_id = pdf_file.stem
                written = self.vector_store.upsert_chunks(
                    doc_id=doc_id,
                    chunks=embed_result.chunks,
                    vectors=embed_result.vectors,
                    company_id=params.company_id,
                    collection_name=params.company_id  # Use company_id as collection name
                )
                
                reports.append(CompanyFileReport(
                    file_path=pdf_file,
                    success=True,
                    chunks_written=written,
                    doc_id=doc_id,
                ))
                
                total_chunks += written
                successful_count += 1
                
            except Exception as e:
                reports.append(CompanyFileReport(
                    file_path=pdf_file,
                    success=False,
                    chunks_written=0,
                    doc_id="",
                    error_message=str(e),
                ))

        return EmbedAndUpsertCompanyPdfsOutput(
            company_id=params.company_id,
            total_files=len(pdf_files),
            successful_files=successful_count,
            total_chunks_written=total_chunks,
            reports=reports,
        )

    def _get_pdf_files(self, company_folder: Path) -> List[Path]:
        """Get all PDF files from the company folder."""
        if not company_folder.exists():
            return []
        
        pdf_files = []
        for file_path in company_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                pdf_files.append(file_path)
        
        return sorted(pdf_files)