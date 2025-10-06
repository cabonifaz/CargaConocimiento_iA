# app/application/use_cases/test_metadata_generation.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort
from app.ports.outbound.chunker import ChunkerPort, ChunkerConfig
from app.domain.services.chunk_quality import QualityConfig
from app.domain.services.md_text_normalizer import MdTextNormalizer
from app.domain.services.bm25_text_processor import BM25TextProcessor
from app.domain.models.weaviate_metadata import WeaviateChunkMetadata
from app.config.settings import settings

from app.application.use_cases.embed_chunks_from_pdf import (
    EmbedChunksFromPdf, EmbedChunksFromPdfInput
)


@dataclass
class TestMetadataReport:
    file_path: Path
    success: bool
    metadata_count: int
    doc_id: str
    output_file: str
    error_message: Optional[str] = None


@dataclass
class TestMetadataGenerationInput:
    company_id: str
    area_id: str
    max_pages: Optional[int] = None
    chunker_cfg: Optional[ChunkerConfig] = None
    quality_cfg: Optional[QualityConfig] = None


@dataclass
class TestMetadataGenerationOutput:
    company_id: str
    area_id: str
    total_files: int
    successful_files: int
    total_metadata_generated: int
    reports: List[TestMetadataReport]


class TestMetadataGeneration:
    """Use case for testing metadata generation without external services."""

    def __init__(
        self,
        blob: BlobStoragePort,
        extractor: TextExtractorPort,
        normalizer: MdTextNormalizer,
        chunker: ChunkerPort,
    ) -> None:
        # Store all components for real processing
        self.blob = blob
        self.extractor = extractor
        self.normalizer = normalizer
        self.chunker = chunker

    def execute(self, params: TestMetadataGenerationInput) -> TestMetadataGenerationOutput:
        # Get all PDF files from company_files/files/
        files_folder = Path("company_files") / "files"
        pdf_files = self._get_pdf_files(files_folder)

        reports: List[TestMetadataReport] = []
        total_metadata = 0
        successful_count = 0

        print(f"\n{'=' * 80}")
        print(f"TESTING METADATA GENERATION")
        print(f"Company ID: {params.company_id} | Area ID: {params.area_id}")
        print(f"Files found: {len(pdf_files)}")
        print(f"{'=' * 80}")

        # Create reports directory
        reports_dir = Path("reports/metadata")
        reports_dir.mkdir(parents=True, exist_ok=True)

        for pdf_file in pdf_files:
            try:
                print(f"\n[PROCESSING] {pdf_file.name}")

                # Use relative path from company_files
                relative_path = Path("files") / pdf_file.name

                # Extract, normalize and chunk (but skip embedding)
                chunk_result = self._extract_and_chunk_only(
                    relative_path, params.max_pages, params.chunker_cfg, params.quality_cfg, params.company_id, params.area_id
                )

                if not chunk_result.chunks:
                    print(f"WARNING: No chunks generated for {pdf_file.name}")
                    continue

                # Generate metadata for each chunk
                metadata_objects = []
                doc_id = pdf_file.stem
                collection_name = params.company_id  # Use company_id directly as collection name

                for chunk in chunk_result.chunks:
                    # Extract section information using domain service
                    section_title, section_path, bm25_text = BM25TextProcessor.extract_section_info(chunk.text)

                    # Create metadata (simulating what WeaviateVectorStore does)
                    metadata = WeaviateChunkMetadata(
                        # Text for search
                        text=chunk.text,
                        bm25_text=bm25_text,
                        doc_title=pdf_file.stem,
                        section_title=section_title,

                        # Identifiers
                        doc_id=doc_id,
                        company_id=params.company_id,
                        company="company_name",
                        area_id=params.area_id,
                        area=params.area_id,
                        section_path=section_path,

                        # Position
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,

                        # Embedding info (simulated)
                        embedding_model=settings.BEDROCK_MODEL_ID,
                        embedding_dim=1024,  # Simulated dimension for Cohere

                        # Optional
                        chunk_id=chunk.chunk_id,
                        token_count=chunk.token_count,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        ingested_at=datetime.now().isoformat(),
                    )

                    # Convert to JSON-serializable format
                    metadata_dict = metadata.to_weaviate_properties()
                    metadata_dict["collection_name"] = collection_name  # Add collection info
                    metadata_objects.append(metadata_dict)

                # Generate output filename
                clean_filename = pdf_file.stem.replace(" ", "-").replace("_", "-")
                output_filename = f"metadata-{clean_filename}-{params.company_id}-{params.area_id}.json"
                output_path = reports_dir / output_filename

                # Save metadata to JSON file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_objects, f, indent=2, ensure_ascii=False)

                reports.append(TestMetadataReport(
                    file_path=pdf_file,
                    success=True,
                    metadata_count=len(metadata_objects),
                    doc_id=doc_id,
                    output_file=str(output_path),
                ))

                total_metadata += len(metadata_objects)
                successful_count += 1

                print(f"COMPLETED: {pdf_file.name} -> {len(metadata_objects)} metadata objects")
                print(f"Saved to: {output_path}")

            except Exception as e:
                reports.append(TestMetadataReport(
                    file_path=pdf_file,
                    success=False,
                    metadata_count=0,
                    doc_id="",
                    output_file="",
                    error_message=str(e),
                ))
                print(f"ERROR: {pdf_file.name} -> {str(e)}")

        return TestMetadataGenerationOutput(
            company_id=params.company_id,
            area_id=params.area_id,
            total_files=len(pdf_files),
            successful_files=successful_count,
            total_metadata_generated=total_metadata,
            reports=reports,
        )

    def _extract_and_chunk_only(self, relative_path: Path, max_pages, chunker_cfg, quality_cfg, company_id: str, area_id: str):
        """Extract and chunk without embedding, generating reports for test mode."""
        from app.application.use_cases.extract_text_from_pdf import (
            ExtractTextFromPdf, ExtractTextInput
        )
        from app.application.use_cases.extract_and_normalize_pdf import (
            ExtractAndNormalizePdf, ExtractAndNormalizeInput
        )
        from app.application.use_cases.extract_normalize_chunk_global_pdf import (
            ExtractNormalizeChunkGlobalPdf, ExtractNormalizeChunkGlobalInput
        )
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = relative_path.stem

        # 1. Extract text and save report
        extract_uc = ExtractTextFromPdf(self.blob, self.extractor)
        extract_result = extract_uc.execute(ExtractTextInput(
            relative_path=relative_path,
            max_pages=max_pages,
            generate_report=False  # We'll create our own custom report
        ))

        # Save extraction report
        extraction_dir = Path("reports/extraction")
        extraction_dir.mkdir(parents=True, exist_ok=True)
        extraction_report_path = extraction_dir / f"extraction-{filename}-{timestamp}.txt"

        with extraction_report_path.open("w", encoding="utf-8") as f:
            f.write(f"=== REPORTE DE EXTRACCIÓN ===\n")
            f.write(f"Archivo: {extract_result.source_path}\n")
            f.write(f"Company ID: {company_id}\n")
            f.write(f"Area ID: {area_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Páginas extraídas: {len(extract_result.result.pages)}\n")
            f.write(f"Total de caracteres: {sum(len(page) for page in extract_result.result.pages)}\n\n")

            for i, page in enumerate(extract_result.result.pages, 1):
                f.write(f"--- Página {i} ---\n")
                f.write(page)
                f.write("\n\n")

        # 2. Normalize text and save report
        normalize_uc = ExtractAndNormalizePdf(self.blob, self.extractor, self.normalizer)
        normalize_result = normalize_uc.execute(ExtractAndNormalizeInput(
            relative_path=relative_path,
            max_pages=max_pages,
            join_pages=False,
            generate_report=False  # We'll create our own custom report
        ))

        # Save normalization report
        normalization_dir = Path("reports/normalization")
        normalization_dir.mkdir(parents=True, exist_ok=True)
        normalization_report_path = normalization_dir / f"normalization-{filename}-{timestamp}.txt"

        with normalization_report_path.open("w", encoding="utf-8") as f:
            f.write(f"=== REPORTE DE NORMALIZACIÓN ===\n")
            f.write(f"Archivo: {normalize_result.source_path}\n")
            f.write(f"Company ID: {company_id}\n")
            f.write(f"Area ID: {area_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Páginas normalizadas: {len(normalize_result.normalized_pages or [])}\n")
            f.write(f"Total de caracteres: {sum(len(page) for page in (normalize_result.normalized_pages or []))}\n\n")

            for i, page in enumerate(normalize_result.normalized_pages or [], 1):
                f.write(f"--- Página Normalizada {i} ---\n")
                f.write(page)
                f.write("\n\n")

        # 3. Chunk text and save report
        chunk_uc = ExtractNormalizeChunkGlobalPdf(
            self.blob,
            self.extractor,
            self.normalizer,
            self.chunker
        )

        chunk_result = chunk_uc.execute(ExtractNormalizeChunkGlobalInput(
            relative_path=relative_path,
            max_pages=max_pages,
            generate_report=False  # We'll create our own custom report
        ))

        # Save chunking report
        chunking_dir = Path("reports/chunking")
        chunking_dir.mkdir(parents=True, exist_ok=True)
        chunking_report_path = chunking_dir / f"chunking-{filename}-{timestamp}.txt"

        with chunking_report_path.open("w", encoding="utf-8") as f:
            f.write(f"=== REPORTE DE CHUNKING ===\n")
            f.write(f"Archivo: {chunk_result.source_path}\n")
            f.write(f"Company ID: {company_id}\n")
            f.write(f"Area ID: {area_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Páginas procesadas: {chunk_result.page_count}\n")
            f.write(f"Chunks generados: {len(chunk_result.chunks)}\n")
            f.write(f"Configuración del chunker: {self.chunker.get_config() if hasattr(self.chunker, 'get_config') else 'N/A'}\n\n")

            for i, chunk in enumerate(chunk_result.chunks, 1):
                f.write(f"--- Chunk {i} ---\n")
                f.write(f"ID: {chunk.chunk_id}\n")
                f.write(f"Páginas: {chunk.page_start}-{chunk.page_end}\n")
                f.write(f"Tokens: {chunk.token_count}\n")
                f.write(f"Caracteres: {chunk.char_start}-{chunk.char_end}\n")
                f.write(f"Contenido:\n{chunk.text}\n\n")

        print(f"Reportes guardados:")
        print(f"  - Extracción: {extraction_report_path}")
        print(f"  - Normalización: {normalization_report_path}")
        print(f"  - Chunking: {chunking_report_path}")

        return chunk_result

    def _get_pdf_files(self, files_folder: Path) -> List[Path]:
        """Get all PDF files from the files folder."""
        if not files_folder.exists():
            return []

        pdf_files: List[Path] = []
        for file_path in files_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                pdf_files.append(file_path)

        return sorted(pdf_files)