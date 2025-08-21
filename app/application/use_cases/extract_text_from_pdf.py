from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

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
        
        # Normalizar la ruta
        raw = Path(params.relative_path)

        if raw.is_absolute():
            target = raw
        else:
            parts = list(raw.parts)
            if parts and (parts[0].lower() == base.name.lower() or parts[0].lower() == "company_files"):
                parts = parts[1:]
            target = base.joinpath(*parts)

        target = target.resolve()

        # Validar si el archivo está dentro de company_files
        if os.path.commonpath([str(base)]) != os.path.commonpath([str(base), str(target)]):
            raise PermissionError(f"El archivo solicitado no está bajo {base}")

        # Leer bytes y extraer
        data = self.blob.read_bytes(target)
        res = self.extractor.extract_from_bytes(data, max_pages=params.max_pages)
        return ExtractTextOutput(result=res, source_path=target)
