from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import hashlib

from app.ports.outbound.tokenizer import TokenCounterPort

@dataclass(frozen=True)
class ChunkerConfig:
    target_tokens: int = 512        # tamaño objetivo por chunk
    overlap_tokens: int = 64        # solapamiento entre chunks
    min_tokens: int = 50            # no emitir chunks demasiado pequeños (salvo último)
    include_empty_pages: bool = False  # típicamente False

@dataclass(frozen=True)
class ChunkResult:
    text: str
    page: int               # página 1-based
    token_count: int
    chunk_id: str           # hash del contenido

class TokenChunker:
    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None) -> None:
        self.tok = tokenizer
        self.cfg = cfg or ChunkerConfig()
        assert self.cfg.target_tokens > self.cfg.overlap_tokens >= 0, "Parámetros inválidos"

    def chunk_pages(self, pages: List[str]) -> List[ChunkResult]:
        """Chunking por páginas ya normalizadas."""
        results: List[ChunkResult] = []
        for i, page_text in enumerate(pages, start=1):
            page_chunks = self._chunk_single(page_text, page=i)
            if page_chunks:
                results.extend(page_chunks)
            elif self.cfg.include_empty_pages:
                results.append(self._mk_chunk("", i, 0))
        return results

    # ---- internos ----

    def _chunk_single(self, text: str, page: int) -> List[ChunkResult]:
        tokens = self.tok.tokenize(text)
        n = len(tokens)
        if n == 0:
            return []

        T = self.cfg.target_tokens
        O = self.cfg.overlap_tokens
        stride = max(T - O, 1)

        chunks: List[ChunkResult] = []
        start_idx = 0
        while start_idx < n:
            end_idx = min(start_idx + T, n)
            # spans a índices de caracteres:
            char_start = tokens[start_idx][0]
            char_end   = tokens[end_idx - 1][1]
            raw = text[char_start:char_end]

            tok_count = end_idx - start_idx
            # evitar trozos demasiado pequeños (si queda una “cola” minúscula y ya hay al menos un chunk)
            if tok_count < self.cfg.min_tokens and chunks:
                # añade esa cola al último chunk si no excede mucho
                last = chunks[-1]
                merged = (last.text + " " + raw).strip()
                merged_count = self.tok.count_tokens(merged)
                if merged_count <= (T + O):  # límite flexible
                    chunks[-1] = self._mk_chunk(merged, page, merged_count)
                else:
                    chunks.append(self._mk_chunk(raw, page, tok_count))
                break
            else:
                chunks.append(self._mk_chunk(raw, page, tok_count))

            if end_idx == n:
                break
            start_idx += stride

        return chunks

    def _mk_chunk(self, text: str, page: int, tok_count: int) -> ChunkResult:
        cid = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return ChunkResult(text=text, page=page, token_count=tok_count, chunk_id=cid)
