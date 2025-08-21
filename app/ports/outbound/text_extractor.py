from __future__ import annotations
from typing import Protocol, List, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class TextExtractionResult:
    pages: List[str]
    page_count: int
    producer: Optional[str] = None

class TextExtractorPort(Protocol):
    def extract_from_bytes(self, data: bytes, max_pages: Optional[int] = None) -> TextExtractionResult:
        ...
