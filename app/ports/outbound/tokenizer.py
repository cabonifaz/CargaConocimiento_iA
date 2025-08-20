from __future__ import annotations
from typing import Protocol, List, Tuple

class TokenCounterPort(Protocol):
    """interface para segmentación por tokens."""

    def tokenize(self, text: str) -> List[Tuple[int, int]]:
        ...

    def count_tokens(self, text: str) -> int:
        ...
