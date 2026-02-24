"""Mistral OCR client for text extraction from PDF documents."""

import logging
from typing import Tuple, List

from mistralai import Mistral

logger = logging.getLogger()

# Mistral OCR pricing: $1 per 1,000 pages = $0.001 per page
COST_PER_PAGE_USD = 0.001


class MistralOCRClient:
    """
    Wrapper around the Mistral OCR API.

    Uses 'mistral-ocr-latest' model to extract markdown text from a PDF
    provided as a presigned S3 URL.
    """

    MODEL = "mistral-ocr-latest"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required")
        self._client = Mistral(api_key=api_key)

    def extract_text(self, presigned_url: str) -> Tuple[List[str], int]:
        """
        Run OCR on a PDF accessible via a presigned URL.

        Pages are returned in document order (sorted by page.index).
        Images are excluded from the response (text extraction only).

        Args:
            presigned_url: Presigned HTTPS GET URL pointing to the source PDF.

        Returns:
            (pages_markdown, total_pages):
                pages_markdown — list of markdown strings, one per page.
                total_pages    — number of pages processed (used for cost calc).

        Raises:
            RuntimeError: If the OCR response contains no pages.
        """
        logger.info(f"Calling Mistral OCR on document URL (model={self.MODEL})")

        response = self._client.ocr.process(
            model=self.MODEL,
            document={"type": "document_url", "document_url": presigned_url},
            include_image_base64=False,
        )

        if not response.pages:
            raise RuntimeError("Mistral OCR returned an empty pages list")

        # Sort pages by index to guarantee document order
        pages = sorted(response.pages, key=lambda p: p.index)
        markdowns = [p.markdown for p in pages]
        total_pages = len(pages)

        logger.info(f"OCR complete — {total_pages} page(s) extracted")
        return markdowns, total_pages

    @staticmethod
    def calculate_cost(total_pages: int) -> float:
        """Return the USD cost for processing `total_pages` pages."""
        return total_pages * COST_PER_PAGE_USD
