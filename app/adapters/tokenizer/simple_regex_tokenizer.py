from __future__ import annotations
import re
from typing import List, Tuple
from app.ports.outbound.tokenizer import TokenCounterPort

# Tokens = palabras (\w+) o signos/puntuación no-espacio (\S)
_TOKEN_RE = re.compile(r"\w+|\S", flags=re.UNICODE)

class SimpleRegexTokenizer(TokenCounterPort):
    """Tokenizador simple y seguro por regex (sin libs externas).
    Aproxima "wordpieces": palabras + signos/puntuación.
    """

    def tokenize(self, text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in _TOKEN_RE.finditer(text or "")]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))
