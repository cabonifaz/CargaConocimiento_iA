from __future__ import annotations
from typing import Protocol, List

class EmbedderPort(Protocol):
    """Devuelve un vector por cada texto de entrada, en el mismo orden."""
    def embed_texts(self, texts: List[str]) -> List[list[float]]:
        ...
