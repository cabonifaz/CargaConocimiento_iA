from dataclasses import dataclass
from typing import List, Optional

from app.ports.outbound.blob_storage import BlobStoragePort, FileInfo
from app.ports.outbound.text_extractor import TextExtractorPort
from app.application.use_cases.extract_and_normalize_pdf import (
    ExtractAndNormalizePdf,
    ExtractAndNormalizeInput,
    ExtractAndNormalizeOutput,
)
from app.domain.services.text_normalizer import NormalizerConfig

@dataclass
class ExtractAndNormalizeAllInput:
    max_pages: Optional[int] = None
    recursive: bool = True
    normalizer_cfg: Optional[NormalizerConfig] = None
    join_pages: bool = False

@dataclass
class ExtractAndNormalizeAllOutput:
    results: List[ExtractAndNormalizeOutput]

class ExtractAndNormalizeAllPdfs:
    def __init__(self, blob: BlobStoragePort, extractor: TextExtractorPort):
        self.blob = blob
        self.extractor = extractor

    def execute(self, params: ExtractAndNormalizeAllInput) -> ExtractAndNormalizeAllOutput:
        files: List[FileInfo] = list(self.blob.list_pdfs(
            base_dir=self.blob.base_dir,
            recursive=params.recursive,
        ))

        results: List[ExtractAndNormalizeOutput] = []
        single_uc = ExtractAndNormalizePdf(self.blob, self.extractor)

        for f in files:
            try:
                out = single_uc.execute(
                    ExtractAndNormalizeInput(
                        relative_path=f.path,
                        max_pages=params.max_pages,
                        normalizer_cfg=params.normalizer_cfg,
                        join_pages=params.join_pages,
                    )
                )
                results.append(out)
            except Exception as e:
                print(f"[WARN] {f.path}: {e}")
        return ExtractAndNormalizeAllOutput(results=results)
