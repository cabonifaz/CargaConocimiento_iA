from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import hashlib
import bisect
import re

from app.ports.outbound.tokenizer import TokenCounterPort

@dataclass(frozen=True)
class GlobalChunkerConfigMd:
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 50
    # Separador entre páginas en el texto unido (no es visible al usuario final).
    # Usa algo poco probable en el contenido para que el mapeo sea estable.
    page_separator: str = "\n\n\f\n\n"  # \f = form feed, útil como marcador

@dataclass(frozen=True)
class GlobalChunkMd:
    text: str
    token_count: int
    chunk_id: str
    # offsets en el texto UNIDO
    char_start: int
    char_end: int
    # páginas 1-based que cubre este chunk (derivadas del mapa)
    page_start: int
    page_end: int

class GlobalTokenChunkerMd:
    """Chunking con ventana deslizante sobre TODO el documento unido.
    Respeta límites semánticos Markdown (tablas, listas, headings, fences).
    Devuelve offsets de caracteres y páginas cubiertas por cada chunk.
    """
    def __init__(self, tokenizer: TokenCounterPort, cfg: GlobalChunkerConfigMd | None = None) -> None:
        self.tok = tokenizer
        self.cfg = cfg or GlobalChunkerConfigMd()
        assert self.cfg.target_tokens > self.cfg.overlap_tokens >= 0, "Parámetros inválidos"

    # ---------- API principal ----------

    def chunk_document(self, pages: List[str]) -> Tuple[List[GlobalChunkMd], str]:
        """Une las páginas, realiza chunking global y devuelve (chunks, full_text)."""
        print("### Global Token Chunker - Chunk document (Markdown-aware) --------------------------")
        full_text, page_offsets = self._join_pages_and_offsets(pages)
        print(f"- Documento unido tiene {len(full_text)} chars y {len(page_offsets)} páginas.")
        chunks = self._chunk_over_text(full_text, page_offsets)
        print(f"- Chunking global produjo {len(chunks)} chunks.\n---")
        return chunks, full_text

    # ---------- Internos (join y offsets) ----------

    def _join_pages_and_offsets(self, pages: List[str]) -> Tuple[str, List[int]]:
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

    # ---------- Markdown helpers ----------

    @staticmethod
    def _line_starts(text: str) -> List[int]:
        # Índices de inicio de cada línea (0-based)
        starts = [0]
        for m in re.finditer(r"\n", text):
            starts.append(m.end())
        return starts

    @staticmethod
    def _is_hr(line: str) -> bool:
        return bool(re.match(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", line))

    @staticmethod
    def _is_heading(line: str) -> bool:
        return bool(re.match(r"^\s*#{1,6}\s+\S", line))

    @staticmethod
    def _is_list(line: str) -> bool:
        return bool(re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+)", line))

    @staticmethod
    def _is_table_sep(line: str) -> bool:
        # línea de separadores de tabla: | --- | :---: | ---: |
        return bool(re.match(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$", line))

    @staticmethod
    def _looks_table_row(line: str) -> bool:
        # fila con pipes (tolerante)
        return "|" in line and not line.strip().startswith("!")  # evita ![img|...], simple heurística

    @staticmethod
    def _trim_span_to_non_ws(text: str, start_char: int, end_char: int) -> Tuple[int, int]:
        # recorta espacios en extremos
        s = start_char
        e = end_char
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        return s, e

    def _build_md_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Devuelve una lista de bloques Markdown:
        [{'type': 'table'|'code'|'heading'|'list'|'hr'|'para'|'other', 'start': int, 'end': int}]
        Los spans son [start, end).
        """
        blocks: List[Dict[str, Any]] = []
        line_starts = self._line_starts(text)
        n_lines = len(line_starts)

        def line_at(i: int) -> str:
            start = line_starts[i]
            end = line_starts[i + 1] - 1 if i + 1 < n_lines else len(text)
            return text[start:end]

        i = 0
        # fence triple ```
        fence_re = re.compile(r"^\s*```")
        while i < n_lines:
            start_i = i
            ln = line_at(i)

            # code-fence block
            if fence_re.match(ln):
                fence_start = line_starts[i]
                i += 1
                while i < n_lines and not fence_re.match(line_at(i)):
                    i += 1
                # incluir línea de cierre si existe
                if i < n_lines:
                    i_end = line_starts[i + 1] - 1 if (i + 1) < n_lines else len(text)
                    blocks.append({"type": "code", "start": fence_start, "end": i_end})
                    i += 1
                else:
                    # fence abierto sin cierre: hasta final
                    blocks.append({"type": "code", "start": fence_start, "end": len(text)})
                continue

            # tabla: secuencia de filas con '|' y, opcionalmente, segunda línea como separador
            if self._looks_table_row(ln):
                # detecta si la segunda línea es separadora
                j = i + 1
                is_table = False
                if j < n_lines and self._is_table_sep(line_at(j)):
                    is_table = True
                    j += 1
                    # extender mientras sigan filas con '|'
                    while j < n_lines and self._looks_table_row(line_at(j)) and line_at(j).strip() != "":
                        j += 1
                else:
                    # tolerante: grupo continuo de líneas con '|' (para tablas “simples” sin sep)
                    k = i
                    count_pipe = 0
                    while k < n_lines and self._looks_table_row(line_at(k)) and line_at(k).strip() != "":
                        count_pipe += 1
                        k += 1
                    if count_pipe >= 2:  # al menos 2 líneas con | para considerarlo tabla
                        is_table = True
                        j = k

                if is_table:
                    start_char = line_starts[i]
                    end_char = line_starts[j] - 1 if j < n_lines else len(text)
                    blocks.append({"type": "table", "start": start_char, "end": end_char})
                    i = j
                    continue

            # horizontal rule
            if self._is_hr(ln):
                start_char = line_starts[i]
                end_char = line_starts[i + 1] - 1 if (i + 1) < n_lines else len(text)
                blocks.append({"type": "hr", "start": start_char, "end": end_char})
                i += 1
                continue

            # heading
            if self._is_heading(ln):
                start_char = line_starts[i]
                end_char = line_starts[i + 1] - 1 if (i + 1) < n_lines else len(text)
                blocks.append({"type": "heading", "start": start_char, "end": end_char})
                i += 1
                continue

            # lista (bloque de ítems contiguos)
            if self._is_list(ln):
                j = i + 1
                fence_re = re.compile(r"^\s*```")
                # Permitimos: siguiente línea sea otro ítem, o línea en blanco,
                # o una línea de continuación (no vacía y no inicia otro bloque fuerte)
                def is_continuation(L: str) -> bool:
                    return (
                        L.strip() != "" and
                        not self._is_heading(L) and
                        not fence_re.match(L) and
                        not self._is_hr(L) and
                        not self._looks_table_row(L)
                    )
                while j < n_lines and (
                    self._is_list(line_at(j)) or
                    line_at(j).strip() == "" or
                    is_continuation(line_at(j))
                ):
                    j += 1
                start_char = line_starts[i]
                end_char = line_starts[j] - 1 if j < n_lines else len(text)
                blocks.append({"type": "list", "start": start_char, "end": end_char})
                i = j
                continue


            # párrafo: líneas no vacías hasta un separador en blanco
            if ln.strip() != "":
                j = i + 1
                while j < n_lines and line_at(j).strip() != "":
                    # si encontramos inicio claro de bloque, cortamos
                    if self._is_heading(line_at(j)) or fence_re.match(line_at(j)) or self._is_hr(line_at(j)):
                        break
                    j += 1
                start_char = line_starts[i]
                end_char = line_starts[j] - 1 if j < n_lines else len(text)
                blocks.append({"type": "para", "start": start_char, "end": end_char})
                i = j
                continue

            # línea en blanco -> avanza
            i += 1

        # Orden por start
        blocks.sort(key=lambda b: b["start"])
        # gaps como 'other'
        result: List[Dict[str, Any]] = []
        last_end = 0
        for b in blocks:
            if b["start"] > last_end:
                result.append({"type": "other", "start": last_end, "end": b["start"]})
            result.append(b)
            last_end = b["end"]
        if last_end < len(text):
            result.append({"type": "other", "start": last_end, "end": len(text)})
        return result

    @staticmethod
    def _find_block_covering(blocks: List[Dict[str, Any]], idx: int, types: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        # busca bloque que cubra idx, opcionalmente filtrando por tipos
        lo, hi = 0, len(blocks) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            b = blocks[mid]
            if idx < b["start"]:
                hi = mid - 1
            elif idx >= b["end"]:
                lo = mid + 1
            else:
                if not types or b["type"] in types:
                    return b
                # si tipo no coincide, no paramos, intentamos vecinos
                # (lineal alrededor, porque lista corta en práctica)
                # pero para simplicidad retornamos bloque actual y que el llamador evalúe tipo
                return b
        return None

    def _nearest_blankline_left(self, text: str, idx: int) -> int:
        # devuelve índice del inicio de la línea anterior en blanco, o 0 si no hay
        # Busca "\n\n" hacia la izquierda
        m = list(re.finditer(r"\n\s*\n", text[:idx]))
        if not m:
            return 0
        return m[-1].end()

    def _nearest_blankline_right(self, text: str, idx: int) -> int:
        # devuelve índice del final de la línea en blanco siguiente, o len(text) si no hay
        m = re.search(r"\n\s*\n", text[idx:])
        if not m:
            return len(text)
        return idx + m.start()

    def _snap_to_md_boundaries(
        self,
        text: str,
        start_char: int,
        end_char: int,
        tokens_window_limit: int,
        blocks: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Ajusta [start_char, end_char) a límites seguros de Markdown:
        - No cortar tablas ni fences; si interseca, intenta incluir el bloque completo
        si cabe en tokens_window_limit.
        - Para tablas demasiado grandes, se permite dividir por filas (en lugar de descartarlas).
        - Prefiere cortar en líneas en blanco (entre párrafos).
        """
        s, e = start_char, end_char
        s, e = self._trim_span_to_non_ws(text, s, e)

        # Si cae dentro de bloque crítico
        critical_types = ["code", "table"]
        for t in critical_types:
            b_s = self._find_block_covering(blocks, s, [t])
            b_e = self._find_block_covering(blocks, e - 1, [t]) if e > s else None

            blk = b_s or b_e
            if blk and blk["type"] == t:
                candidate_s = min(s, blk["start"])
                candidate_e = max(e, blk["end"])
                candidate_text = text[candidate_s:candidate_e]
                token_count = self.tok.count_tokens(candidate_text)

                if token_count <= tokens_window_limit:
                    # ✅ El bloque completo cabe → usarlo entero
                    s, e = candidate_s, candidate_e
                else:
                    if blk["type"] == "table":
                        # 🔧 En lugar de descartar → cortar tabla por filas
                        table_text = text[blk["start"]:blk["end"]]
                        rows = table_text.splitlines()

                        # buscar punto de corte dentro de la tabla
                        running_tokens = 0
                        cut_index = None
                        for i, row in enumerate(rows):
                            running_tokens += self.tok.count_tokens(row + "\n")
                            if running_tokens > tokens_window_limit:
                                cut_index = i
                                break

                        if cut_index is not None:
                            # cortar tabla en dos partes
                            part1 = "\n".join(rows[:cut_index])
                            s = blk["start"]
                            e = s + len(part1)
                        else:
                            # si aún así no encontramos corte, fallback: usar lo que quepa
                            e = blk["start"] + len(table_text[:tokens_window_limit])
                    else:
                        # para code blocks muy grandes → fallback original
                        if blk["start"] >= s:
                            e = min(e, blk["start"])
                        else:
                            s = max(s, blk["end"])

                s, e = self._trim_span_to_non_ws(text, s, e)

        # Preferencias suaves: headings y listas al inicio de chunk si están cerca
        head_block = self._find_block_covering(blocks, s)
        if head_block and head_block["type"] in ("para", "other"):
            m = re.search(r"(^|\n)\s*(#{1,6}\s+\S|[-*+]\s+\S|\d+\.\s+\S)", text[s:e])
            if m and m.start() <= 120:
                new_s = s + m.start()
                candidate_text = text[new_s:e]
                if self.tok.count_tokens(candidate_text) <= tokens_window_limit:
                    s = new_s

        # Cortes en líneas en blanco como fallback
        bl_right = self._nearest_blankline_right(text, s)
        if bl_right < e and (bl_right - s) <= 120:
            candidate_text = text[bl_right:e]
            if self.tok.count_tokens(candidate_text) >= max(10, self.cfg.min_tokens // 2):
                s = bl_right

        bl_left = self._nearest_blankline_left(text, e)
        if bl_left > s and (e - bl_left) <= 120:
            candidate_text = text[s:bl_left]
            if self.tok.count_tokens(candidate_text) >= max(10, self.cfg.min_tokens // 2):
                e = bl_left

        s, e = self._trim_span_to_non_ws(text, s, e)
        if e <= s:
            e = min(len(text), s + 1)
        return s, e

    # ---------- Chunking principal con awareness de Markdown ----------

    def _chunk_over_text(self, text: str, page_offsets: List[int]) -> List[GlobalChunkMd]:
        tokens = self.tok.tokenize(text)  # List[Tuple[char_start, char_end]]
        n = len(tokens)
        if n == 0:
            return []

        T = self.cfg.target_tokens
        O = self.cfg.overlap_tokens
        stride = max(T - O, 1)

        # Analiza estructura Markdown una sola vez
        md_blocks = self._build_md_blocks(text)

        chunks: List[GlobalChunkMd] = []
        start_idx = 0

        while start_idx < n:
            end_idx = min(start_idx + T, n)
            char_start = tokens[start_idx][0]
            char_end   = tokens[end_idx - 1][1]

            # Ajusta a límites MD (no cortar tablas/fences; preferir saltos entre párrafos)
            adj_start, adj_end = self._snap_to_md_boundaries(
                text,
                start_char=char_start,
                end_char=char_end,
                tokens_window_limit=T + O,  # permitimos pequeña expansión si cabe
                blocks=md_blocks,
            )

            # token count sobre texto ajustado
            raw = text[adj_start:adj_end]
            tok_count = self.tok.count_tokens(raw)

            # Evitar colas diminutas fusionando con el último chunk si cabe
            if tok_count < self.cfg.min_tokens and chunks:
                last = chunks[-1]
                candidate_text = (last.text + "\n" + raw).strip()
                candidate_tok_count = self.tok.count_tokens(candidate_text)
                if candidate_tok_count <= (T + O):
                    new_page_start, new_page_end = self._pages_for_span(
                        page_offsets, start_char=last.char_start, end_char=adj_end
                    )
                    chunks[-1] = GlobalChunkMd(
                        text=candidate_text,
                        token_count=candidate_tok_count,
                        chunk_id=self._hash(candidate_text),
                        char_start=last.char_start,
                        char_end=adj_end,
                        page_start=new_page_start,
                        page_end=new_page_end,
                    )
                else:
                    page_start, page_end = self._pages_for_span(page_offsets, adj_start, adj_end)
                    chunks.append(self._mk_global_chunk(raw, tok_count, adj_start, adj_end, page_start, page_end))
                # mover ventana al siguiente stride desde end_idx original
                if end_idx == n:
                    break
                start_idx += stride
                continue
            else:
                page_start, page_end = self._pages_for_span(page_offsets, adj_start, adj_end)
                chunks.append(self._mk_global_chunk(raw, tok_count, adj_start, adj_end, page_start, page_end))

            if end_idx == n:
                break
            start_idx += stride

        return chunks

    # ---------- utilidades de páginas / construcción de chunk ----------

    def _pages_for_span(self, page_offsets: List[int], start_char: int, end_char: int) -> Tuple[int, int]:
        """Dado un rango [start_char, end_char) en full_text,
        devuelve páginas 1-based que cubre el span.
        page_offsets[k] = índice char donde inicia la página (k+1).
        """
        ps = bisect.bisect_right(page_offsets, start_char) - 1
        ps = max(ps, 0)
        ec = max(end_char - 1, 0)
        pe = bisect.bisect_right(page_offsets, ec) - 1
        pe = max(pe, 0)
        return ps + 1, pe + 1

    def _mk_global_chunk(self, text: str, tok_count: int, cs: int, ce: int, pstart: int, pend: int) -> GlobalChunkMd:
        return GlobalChunkMd(
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
