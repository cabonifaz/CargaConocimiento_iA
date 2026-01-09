from dataclasses import dataclass
from typing import List, Optional

from app.ports.outbound.blob_storage import BlobStoragePort, FileInfo
from app.ports.outbound.text_extractor import TextExtractorPort
from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf,
    ExtractTextInput,
    ExtractTextOutput,
)

@dataclass
class ExtractAllPdfsInput:
    max_pages: Optional[int] = None
    recursive: bool = True

@dataclass
class ExtractAllPdfsOutput:
    results: List[ExtractTextOutput]

class ExtractAllPdfs:
    def __init__(self, blob: BlobStoragePort, extractor: TextExtractorPort) -> None:
        self.blob = blob
        self.extractor = extractor

    def execute(self, params: ExtractAllPdfsInput) -> ExtractAllPdfsOutput:
        # Obtener lista de PDFs válidos
        files: List[FileInfo] = list(self.blob.list_pdfs(
            base_dir=self.blob.base_dir,
            recursive=params.recursive
        ))

        results: List[ExtractTextOutput] = []
        single_uc = ExtractTextFromPdf(self.blob, self.extractor)

        for f in files:
            try:
                out = single_uc.execute(
                    ExtractTextInput(relative_path=f.path, max_pages=params.max_pages)
                )
                results.append(out)
            except Exception as e:
                print(f"[WARN] No se pudo extraer {f.path}: {e}")
        return ExtractAllPdfsOutput(results=results)
