# src/app/application/use_cases/embed_and_upsert_company_pdfs.py
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
class CompanyFileReport:
    file_path: Path
    area: str
    success: bool
    chunks_written: int
    doc_id: str
    error_message: Optional[str] = None

@dataclass
class EmbedAndUpsertCompanyPdfsInput:
    company_id: str
    max_pages: Optional[int] = None
    chunker_cfg: Optional[ChunkerConfig] = None
    quality_cfg: Optional[QualityConfig] = None

    # Company metadata
    company_id_int: int = 1  # Numeric ID for Weaviate
    embedding_model: str = "cohere.embed-multilingual-v3"

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
        normalizer: MdTextNormalizer,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
    ) -> None:
        self.embed_uc = EmbedChunksFromPdf(blob, extractor, normalizer, chunker, embedder)
        self.vector_store = vector_store
        self.blob = blob

    def execute(self, params: EmbedAndUpsertCompanyPdfsInput) -> EmbedAndUpsertCompanyPdfsOutput:
        # Get all area subdirectories and their PDFs
        company_folder = Path("company_files") / params.company_id
        area_pdf_files = self._get_area_pdf_files(company_folder)
        
        reports: List[CompanyFileReport] = []
        total_chunks = 0
        successful_count = 0

        for area, pdf_file in area_pdf_files:
            try:
                print(f"🕑 Procesando archivo {pdf_file.name} del área {area}...")
                # Use relative path from company_files including area
                relative_path = Path(params.company_id) / area / pdf_file.name
                
                # First get embeddings
                embed_result = self.embed_uc.execute(EmbedChunksFromPdfInput(
                    relative_path=relative_path,
                    max_pages=params.max_pages,
                    chunker_cfg=params.chunker_cfg,
                    quality_cfg=params.quality_cfg,
                ))

                # Then upsert to collection with company name
                doc_id = pdf_file.stem
                area_id = self._get_area_id(area)  # Map area name to ID

                written = self.vector_store.upsert_chunks(
                    doc_id=doc_id,
                    chunks=embed_result.chunks,  # type: ignore[arg-type]
                    vectors=embed_result.vectors,
                    company_id=params.company_id_int,
                    company=params.company_id,
                    area_id=area_id,
                    area=area,
                    doc_title=pdf_file.stem,
                    embedding_model=params.embedding_model,
                    collection_name=params.company_id  # Use company_id as collection name
                )

                reports.append(CompanyFileReport(
                    file_path=pdf_file,
                    area=area,
                    success=True,
                    chunks_written=written,
                    doc_id=doc_id,
                ))
                
                total_chunks += written
                successful_count += 1

                print(f"\n✅ Procesamiento exitoso del archivo {pdf_file.name} del área {area}.\n\n---")
                
            except Exception as e:
                reports.append(CompanyFileReport(
                    file_path=pdf_file,
                    area=area,
                    success=False,
                    chunks_written=0,
                    doc_id="",
                    error_message=str(e),
                ))

        return EmbedAndUpsertCompanyPdfsOutput(
            company_id=params.company_id,
            total_files=len(area_pdf_files),
            successful_files=successful_count,
            total_chunks_written=total_chunks,
            reports=reports,
        )

    def _get_pdf_files(self, company_folder: Path) -> List[Path]:
        """Get all PDF files from the company folder."""
        if not company_folder.exists():
            return []
        
        pdf_files: List[Path] = []
        for file_path in company_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                pdf_files.append(file_path)
        
        return sorted(pdf_files)

    def _get_area_pdf_files(self, company_folder: Path) -> List[tuple[str, Path]]:
        """Get all PDF files from area subdirectories within the company folder.
        
        Returns:
            List of tuples (area_name, pdf_file_path)
        """
        if not company_folder.exists():
            return []
        
        area_pdf_files: List[tuple[str, Path]] = []
        
        # Iterate through subdirectories (areas)
        for area_path in company_folder.iterdir():
            if area_path.is_dir():
                area_name = area_path.name
                # Get all PDF files in this area directory
                for pdf_file in area_path.iterdir():
                    if pdf_file.is_file() and pdf_file.suffix.lower() == '.pdf':
                        area_pdf_files.append((area_name, pdf_file))
        
        return sorted(area_pdf_files, key=lambda x: (x[0], x[1].name))

    def _get_area_id(self, area_name: str) -> int:
        """Map area name to numeric ID. This could be enhanced with a database lookup."""
        area_mapping = {
            "VENTAS": 1,
            "MARKETING": 2,
            "FINANZAS": 3,
            "RECURSOS_HUMANOS": 4,
            "OPERACIONES": 5,
            "TECNOLOGIA": 6,
            "LEGAL": 7,
            "ADMINISTRACION": 8,
            "GENERAL": 9,
        }
        return area_mapping.get(area_name.upper(), 99)  # Default to 99 for unknown areas