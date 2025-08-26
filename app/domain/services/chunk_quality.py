from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

# Para no acoplar al tipo concreto del chunk global:
class HasTextTok(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def token_count(self) -> int: ...

@dataclass(frozen=True)
class QualityConfig:
    min_tokens: int = 50
    min_chars: int = 200
    min_alpha_ratio: float = 0.30
    min_unique_ratio: float = 0.10  # unique_words / total_words

class ChunkQuality:
    def __init__(self, cfg: QualityConfig | None = None) -> None:
        self.cfg = cfg or QualityConfig()

    def good(self, c: HasTextTok) -> bool:
        text = c.text or ""
        if c.token_count < self.cfg.min_tokens: return False
        if len(text) < self.cfg.min_chars: return False
        letters = sum(ch.isalpha() for ch in text)
        if (letters / max(len(text), 1)) < self.cfg.min_alpha_ratio: return False
        words = [w for w in text.split() if w.isalpha()]
        if not words: return False
        uniq = len(set(w.lower() for w in words))
        if (uniq / len(words)) < self.cfg.min_unique_ratio: return False
        return True
