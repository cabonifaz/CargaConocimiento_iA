from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for chunk quality validation."""
    min_tokens: int = 50
    min_chars: int = 200
    min_alpha_ratio: float = 0.30
    min_unique_ratio: float = 0.10


class ChunkQuality:
    """Chunk quality validator."""

    def __init__(self, cfg: Optional[QualityConfig] = None):
        self.cfg = cfg or QualityConfig()

    def is_good_quality(self, chunk: Dict[str, Any]) -> bool:
        """Verifies if chunk meets quality standards."""
        text = chunk.get('text', '')
        token_count = chunk.get('token_count', 0)

        if token_count < self.cfg.min_tokens:
            return False
        if len(text) < self.cfg.min_chars:
            return False

        letters = sum(ch.isalpha() for ch in text)
        if (letters / max(len(text), 1)) < self.cfg.min_alpha_ratio:
            return False

        words = [w for w in text.split() if w.isalpha()]
        if not words:
            return False

        unique_words = len(set(w.lower() for w in words))
        if (unique_words / len(words)) < self.cfg.min_unique_ratio:
            return False

        return True
