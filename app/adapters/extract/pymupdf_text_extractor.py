from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List

import pymupdf
from app.ports.outbound.text_extractor import TextExtractorPort, TextExtractionResult


@dataclass
class PyMuPDFConfig:
    max_pages: Optional[int] = None
    sort_text: bool = True
    codec: str = "utf-8"


class PyMuPDFTextExtractor(TextExtractorPort):
    """Extractor de texto usando PyMuPDF. Trabaja SOBRE BYTES."""

    def __init__(self, config: Optional[PyMuPDFConfig] = None) -> None:
        self.cfg = config or PyMuPDFConfig()

    def extract_from_bytes(self, data: bytes, max_pages: Optional[int] = None) -> TextExtractionResult:
        print("### PyMuPDF Text Extractor - extract_from_bytes -> pages: List[str], page_count: int, producer: Optional[str]:")
        hard_cap = max_pages if max_pages is not None else self.cfg.max_pages

        doc = pymupdf.open(stream=data)

        try:
            if getattr(doc, "needs_pass", False):
                return TextExtractionResult(pages=[], page_count=0, producer=None)

            pages: List[str] = []
            page_total = doc.page_count  # número de páginas
            limit = min(page_total, hard_cap) if hard_cap is not None else page_total

            for i in range(limit):
                page = doc.load_page(i)
                txt = page.get_textpage().extractTEXT(sort=self.cfg.sort_text)
                # Normaliza saltos finales de PyMuPDF
                pages.append(txt.rstrip("\n"))

            meta = doc.metadata or {}
            producer = None
            # Clave típica en metadata es 'producer'
            if isinstance(meta, dict):
                producer = meta.get("producer") or meta.get("Producer")

            print(f"- {len(pages)} páginas extraídas (de {page_total})")

            for i, p in enumerate(pages):
                print(f"- Página {i+1} ({len(p)} chars):\n  {p[:50]!r}...")

            print("---")

            return TextExtractionResult(
                pages=pages,
                page_count=len(pages),
                producer=producer if isinstance(producer, str) else None,
            )
        finally:
            doc.close()
