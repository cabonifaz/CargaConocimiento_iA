from __future__ import annotations
from typing import Protocol, List

class EmbedderPort(Protocol):
    """Devuelve un vector por cada texto de entrada, en el mismo orden."""
    def embed_texts(self, texts: List[str]) -> List[list[float]]:
        ...
    
    # Optional tracking methods for implementations that support them
    def start_pdf_tracking(self, pdf_name: str, pdf_size_mb: float) -> None:
        """Start tracking metrics for a new PDF. Optional method."""
        ...
        
    def end_pdf_tracking(self, pdf_name: str, pdf_size_mb: float) -> None:
        """End tracking and log PDF summary. Optional method."""
        ...
