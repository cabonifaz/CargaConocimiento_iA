from __future__ import annotations
from typing import Protocol, List, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class TextExtractionResult:
    pages: List[str]           # texto por página (sin normalizar)
    page_count: int            # redundante, pero útil
    producer: Optional[str] = None  # metadata opcional del PDF

class TextExtractorPort(Protocol):
    def extract_from_bytes(self, data: bytes, max_pages: Optional[int] = None) -> TextExtractionResult:
        ...
