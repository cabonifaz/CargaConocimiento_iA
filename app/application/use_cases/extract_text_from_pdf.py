from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.ports.outbound.blob_storage import BlobStoragePort
from app.ports.outbound.text_extractor import TextExtractorPort, TextExtractionResult
from app.config.settings import settings

@dataclass
class ExtractTextInput:
    relative_path: Path
    max_pages: Optional[int] = None

@dataclass
class ExtractTextOutput:
    result: TextExtractionResult
    source_path: Path

class ExtractTextFromPdf:
    def __init__(self, blob: BlobStoragePort, extractor: TextExtractorPort) -> None:
        self.blob = blob
        self.extractor = extractor

    def execute(self, params: ExtractTextInput) -> ExtractTextOutput:
        base = settings.COMPANY_FILES_DIR.resolve()
        target = (base / params.relative_path).resolve()

        # Asegurar que target está dentro de company_files
        if base not in target.parents and target != base:
            raise PermissionError("El archivo solicitado no está bajo company_files/")

        # Leer bytes desde el blob
        data = self.blob.read_bytes(target)
        res = self.extractor.extract_from_bytes(data, max_pages=params.max_pages)
        return ExtractTextOutput(result=res, source_path=target)
