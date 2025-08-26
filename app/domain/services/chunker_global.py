from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import hashlib
import bisect

from app.ports.outbound.tokenizer import TokenCounterPort

@dataclass(frozen=True)
class GlobalChunkerConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 50
    # Separador entre páginas en el texto unido (no es visible al usuario final).
    # Usa algo poco probable en el contenido para que el mapeo sea estable.
    page_separator: str = "\n\n\f\n\n"  # \f = form feed, útil como marcador

@dataclass(frozen=True)
class GlobalChunk:
    text: str
    token_count: int
    chunk_id: str
    # offsets en el texto UNIDO
    char_start: int
    char_end: int
    # páginas 1-based que cubre este chunk (derivadas del mapa)
    page_start: int
    page_end: int

class GlobalTokenChunker:
    """Chunking con ventana deslizante sobre TODO el documento unido.
    Devuelve offsets de caracteres y páginas cubiertas por cada chunk.
    """
    def __init__(self, tokenizer: TokenCounterPort, cfg: GlobalChunkerConfig | None = None) -> None:
        self.tok = tokenizer
        self.cfg = cfg or GlobalChunkerConfig()
        assert self.cfg.target_tokens > self.cfg.overlap_tokens >= 0, "Parámetros inválidos"

    # ---------- API principal ----------

    def chunk_document(self, pages: List[str]) -> Tuple[List[GlobalChunk], str]:
        """Une las páginas, realiza chunking global y devuelve (chunks, full_text)."""
        print("### Global Token Chunker - Chunk document")
        full_text, page_offsets = self._join_pages_and_offsets(pages)
        print(f"- Documento unido tiene {len(full_text)} chars y {len(page_offsets)} páginas.")
        chunks = self._chunk_over_text(full_text, page_offsets)
        print(f"- Chunking global produjo {len(chunks)} chunks.\n---")
        return chunks, full_text

    # ---------- Internos ----------

    def _join_pages_and_offsets(self, pages: List[str]) -> Tuple[str, List[int]]:
        """Une páginas con separador y devuelve:
        - full_text
        - page_offsets: lista de índices char de INICIO de cada página en full_text (len = n_pages)
        """
        sep = self.cfg.page_separator
        offsets: List[int] = []
        parts: List[str] = []
        pos = 0
        for i, p in enumerate(pages):
            offsets.append(pos)
            parts.append(p)
            pos += len(p)
            if i != len(pages) - 1:
                parts.append(sep)
                pos += len(sep)
        full_text = "".join(parts)
        return full_text, offsets

    def _chunk_over_text(self, text: str, page_offsets: List[int]) -> List[GlobalChunk]:
        tokens = self.tok.tokenize(text)
        n = len(tokens)
        if n == 0:
            return []

        T = self.cfg.target_tokens
        O = self.cfg.overlap_tokens
        stride = max(T - O, 1)

        chunks: List[GlobalChunk] = []
        start_idx = 0

        while start_idx < n:
            end_idx = min(start_idx + T, n)
            char_start = tokens[start_idx][0]
            char_end   = tokens[end_idx - 1][1]
            raw = text[char_start:char_end]
            tok_count = end_idx - start_idx

            # evitar cola diminuta
            if tok_count < self.cfg.min_tokens and chunks:
                last = chunks[-1]
                merged_text = (last.text + " " + raw).strip()
                merged_tok_count = self.tok.count_tokens(merged_text)
                if merged_tok_count <= (T + O):
                    # reconstruir chunk fusionado con mismos límites de páginas recalculados
                    # char_start se mantiene del último; char_end del actual
                    new_page_start, new_page_end = self._pages_for_span(
                        page_offsets, start_char=last.char_start, end_char=char_end
                    )
                    chunks[-1] = GlobalChunk(
                        text=merged_text,
                        token_count=merged_tok_count,
                        chunk_id=self._hash(merged_text),
                        char_start=last.char_start,
                        char_end=char_end,
                        page_start=new_page_start,
                        page_end=new_page_end,
                    )
                else:
                    page_start, page_end = self._pages_for_span(page_offsets, char_start, char_end)
                    chunks.append(self._mk_global_chunk(raw, tok_count, char_start, char_end, page_start, page_end))
                break
            else:
                page_start, page_end = self._pages_for_span(page_offsets, char_start, char_end)
                chunks.append(self._mk_global_chunk(raw, tok_count, char_start, char_end, page_start, page_end))

            if end_idx == n:
                break
            start_idx += stride

        return chunks

    def _pages_for_span(self, page_offsets: List[int], start_char: int, end_char: int) -> Tuple[int, int]:
        """Dado un rango [start_char, end_char) en full_text,
        devuelve páginas 1-based que cubre el span.
        page_offsets[k] = índice char donde inicia la página (k+1).
        """
        # page_start = last offset <= start_char
        ps = bisect.bisect_right(page_offsets, start_char) - 1
        ps = max(ps, 0)
        # para end, usamos end_char - 1 (último char real del span)
        ec = max(end_char - 1, 0)
        pe = bisect.bisect_right(page_offsets, ec) - 1
        pe = max(pe, 0)
        # 1-based
        return ps + 1, pe + 1

    def _mk_global_chunk(self, text: str, tok_count: int, cs: int, ce: int, pstart: int, pend: int) -> GlobalChunk:
        return GlobalChunk(
            text=text,
            token_count=tok_count,
            chunk_id=self._hash(text),
            char_start=cs,
            char_end=ce,
            page_start=pstart,
            page_end=pend,
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
