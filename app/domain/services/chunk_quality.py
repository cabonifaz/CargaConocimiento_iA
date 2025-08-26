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
        if c.token_count < self.cfg.min_tokens: return False # Cantidad mínima de tokens
        print(f"\r\033[2K1/5 Validaciones exitosas", end='', flush=True)  # \033[2K limpia la línea
        
        if len(text) < self.cfg.min_chars: return False # Cantidad mínima de caracteres
        print(f"\r\033[2K2/5 Validaciones exitosas", end='', flush=True)

        letters = sum(ch.isalpha() for ch in text)
        if (letters / max(len(text), 1)) < self.cfg.min_alpha_ratio: return False # Ratio de letras a chars
        print(f"\r\033[2K3/5 Validaciones exitosas", end='', flush=True)

        words = [w for w in text.split() if w.isalpha()]
        if not words: return False # Validar palabras alfabéticas
        print(f"\r\033[2K4/5 Validaciones exitosas", end='', flush=True)

        uniq = len(set(w.lower() for w in words))
        if (uniq / len(words)) < self.cfg.min_unique_ratio: return False # Ratio de palabras únicas
        print(f"\r\033[2K5/5 Validaciones exitosas", end='', flush=True)


        return True
