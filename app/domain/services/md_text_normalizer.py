from __future__ import annotations
import pathlib
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Iterable

# ===== Zero-width chars que conviene remover =====
_ZW_CHARS = [
    "\u200B",  # zero width space
    "\u200C",  # zero width non-joiner
    "\u200D",  # zero width joiner
    "\uFEFF",  # zero width no-break space (BOM)
]

# ===== Detectores de bloques Markdown =====
_HEADER_RE      = re.compile(r'^(#{1,6})\s+.*$', re.M)
_CODE_FENCE_RE  = re.compile(r'^```.*$', re.M)
_TABLE_ROW_RE   = re.compile(r'^\s*\|.*\|\s*$', re.M)
_LIST_RE        = re.compile(r'^\s*(?:[-*+]\s+|\d+\.\s+|✓\s+).+', re.M)
_QUOTE_RE       = re.compile(r'^\s*>\s+.+', re.M)

def _iter_markdown_blocks(md: str) -> List[Tuple[str, str, int, int]]:
    """
    Escáner robusto línea-a-línea (sin find()).
    Devuelve (block_text, block_type, start_idx, end_idx)
    block_type in {'heading','code','table','list','quote','paragraph'}
    """
    lines = md.splitlines(keepends=True)  # preserva saltos para offsets correctos
    nlines = len(lines)
    blocks: List[Tuple[str, str, int, int]] = []

    # Estado del autómata
    cur_type: str | None = None
    buf: List[str] = []
    block_start_offset: int | None = None
    offset = 0  # offset acumulado en el string original

    def flush():
        nonlocal cur_type, buf, block_start_offset, offset
        if cur_type is None or not buf:
            cur_type, buf, block_start_offset = None, [], None
            return
        text = "".join(buf)
        if text.strip():  # evita bloques vacíos
            start = block_start_offset if block_start_offset is not None else offset - len(text)
            end = start + len(text)
            blocks.append((text, cur_type, start, end))
        # reset
        cur_type, buf, block_start_offset = None, [], None

    def start_block(btype: str, line: str):
        nonlocal cur_type, buf, block_start_offset, offset
        flush()
        cur_type = btype
        buf = [line]
        block_start_offset = offset

    def continue_block(line: str):
        nonlocal buf
        buf.append(line)

    # helpers de clasificación por línea (sin el \n final)
    def is_code_fence(s: str) -> bool:
        return bool(_CODE_FENCE_RE.match(s))
    def is_heading(s: str) -> bool:
        return bool(_HEADER_RE.match(s))
    def is_table_row(s: str) -> bool:
        return bool(_TABLE_ROW_RE.match(s))
    def is_list_line(s: str) -> bool:
        return bool(_LIST_RE.match(s))
    def is_quote_line(s: str) -> bool:
        return bool(_QUOTE_RE.match(s))
    def is_blank(s: str) -> bool:
        return s.strip() == ""

    in_code = False  # estamos dentro de ``` … ```

    for idx in range(nlines):
        line = lines[idx]
        line_noeol = line[:-1] if line.endswith("\n") else line  # para probar regex

        # 1) Manejo de code fences (tiene prioridad absoluta)
        if is_code_fence(line_noeol):
            if not in_code:
                # abrir bloque de código
                start_block('code', line)
                in_code = True
            else:
                # cerrar bloque de código
                if cur_type == 'code':
                    continue_block(line)
                    flush()
                else:
                    # si por alguna razón estábamos en otro bloque, lo cerramos y abrimos/cerramos code
                    flush()
                    start_block('code', line)
                    flush()
                in_code = False
            offset += len(line)
            continue

        if in_code:
            # todo va dentro del bloque de código hasta encontrar otra fence
            if cur_type != 'code':
                start_block('code', line)
            else:
                continue_block(line)
            offset += len(line)
            continue

        # 2) Clasificación fuera de code
        if is_heading(line_noeol):
            start_block('heading', line)
            flush()
            offset += len(line)
            continue

        if is_table_row(line_noeol):
            if cur_type == 'table':
                continue_block(line)
            else:
                start_block('table', line)
            offset += len(line)
            continue

        if is_list_line(line_noeol):
            if cur_type == 'list':
                continue_block(line)
            else:
                start_block('list', line)
            offset += len(line)
            continue

        if is_quote_line(line_noeol):
            if cur_type == 'quote':
                continue_block(line)
            else:
                start_block('quote', line)
            offset += len(line)
            continue

        if is_blank(line_noeol):
            # Línea en blanco: cierra bloques estructurados;
            # en párrafos, la preservamos como separador.
            if cur_type in ('table', 'list', 'quote', 'heading'):
                flush()
            elif cur_type == 'paragraph':
                continue_block(line)  # párrafo puede contener blanks internos
            else:
                # sin bloque activo: simplemente avanzamos (blank fuera de bloque)
                pass
            offset += len(line)
            continue

        # 3) Default: párrafo
        if cur_type == 'paragraph':
            continue_block(line)
        else:
            start_block('paragraph', line)

        offset += len(line)

    # Fin: flush de lo pendiente
    flush()

    return blocks


# ===== Config compatible con tu puerto =====
@dataclass(frozen=True)
class MdNormalizerConfig:
    unicode_form: unicodedata._NormalizationForm = "NFC"  # "NFC" / "NFKC"
    collapse_whitespace: bool = True          # Solo en párrafos/citas (mid-line)
    strip_control_chars: bool = True
    normalize_nbsp: bool = True
    remove_zero_width: bool = True
    fix_broken_hyphens: bool = True           # Solo en párrafos/citas
    join_soft_linebreaks: bool = True         # Solo en párrafos/citas
    keep_double_newlines: bool = True         # Preserva párrafos
    html_br_to_newline_in_paragraphs: bool = True  # <br> -> '\n' en párrafos/citas

    # Extras útiles:
    max_consecutive_blank_lines: int = 2      # colapsar \n\n\n… a 2
    list_left_pad_max: int = 2                # máximo de espacios a la izquierda en items (evita indentación “falsa”)
    normalize_checkmark_bullets: bool = True  # "✓ " -> "- "
    strip_bold_punctuation: bool = True       # convierte **,** -> ,
    strip_heading_bold: bool = True           # "# **Titulo**" -> "# Titulo"
    simplify_mailto_links: bool = True        # [email](mailto:email) -> email

class MdTextNormalizer:
    """
    Implementa la MISMA interfaz/puerto que tu TextNormalizer:
      - normalize_pages(pages: List[str]) -> List[str]
      - normalize_and_join(pages: List[str]) -> str
    y el interno:
      - _normalize_page(text: str) -> str

    Además, aplica heurísticas MD-aware y limpia encabezados/pies repetidos entre páginas.
    """
    def __init__(self, cfg: MdNormalizerConfig | None = None) -> None:
        self.cfg = cfg or MdNormalizerConfig()

    # === API pública (MISMA FIRMA) ===
    def normalize_pages(self, pages: List[str]) -> List[str]:
        print("### MdTextNormalizer - normalize_pages -> List[str]: ------------------------------------------")
        # 1) normaliza cada página
        norm_per_page = []
        for i, p in enumerate(pages):
            print(f"Normalización de página {i+1}")
            ntext = self._normalize_page(p or "")
            norm_per_page.append(ntext)
            print(f"-> Página {i+1} normalizada ({len(p)} -> {len(ntext)} chars).")

        # 2) detecta encabezados/pies repetidos (heurística simple top/bottom 5 líneas)
        headers_to_drop, footers_to_drop = self._detect_common_headers_footers(norm_per_page)
        if headers_to_drop or footers_to_drop:
            print(f"- Detectados headers comunes: {len(headers_to_drop)} | footers comunes: {len(footers_to_drop)}")
            norm_per_page = [self._drop_headers_footers(t, headers_to_drop, footers_to_drop) for t in norm_per_page]

        # 3) colapsa saltos extra por página
        norm_per_page = [self._collapse_blank_lines(t) for t in norm_per_page]

        print("---")

        return norm_per_page

    def normalize_and_join(self, pages: List[str]) -> str:
        normed_pages = self.normalize_pages(pages)
        joined = "\n\n".join(normed_pages).strip()
        # último pase global de colapso de saltos para el documento unido
        return self._collapse_blank_lines(joined)

    # === Internos ===
    def _normalize_page(self, text: str) -> str:
        t = text or ""
        print("Subprocesos:", end=" ")

        # 1) Sanitizaciones base (seguras en todo el string)
        t = self._base_sanitizations(t)
        print(f"\r\033[2K- 1/3 base sanitizations", end='', flush=True)

        # 2) Split en bloques MD + normalización por tipo
        blocks = _iter_markdown_blocks(t)
        print(f"\r\033[2K- 2/3 split en {len(blocks)} bloques", end='', flush=True)

        normed_blocks: List[str] = []
        for bt, btype, s, e in blocks:
            if btype == 'code':
                normed_blocks.append(self._normalize_code_block(bt))
            elif btype == 'table':
                normed_blocks.append(self._normalize_table_block(bt))
            elif btype == 'list':
                normed_blocks.append(self._normalize_list_block(bt))
            elif btype == 'quote':
                normed_blocks.append(self._normalize_quote_block(bt))
            elif btype == 'heading':
                normed_blocks.append(self._normalize_heading(bt))
            else:  # 'paragraph'
                normed_blocks.append(self._normalize_paragraph(bt))

        # 3) Reconstrucción y colapso de saltos
        result = "\n".join(normed_blocks).strip()
        result = self._collapse_blank_lines(result)
        print(f"\r\033[2K- 3/3 reconstrucción completa", end=' ', flush=True)
        return result

    # ---- Helpers de normalización ----
    def _base_sanitizations(self, t: str) -> str:
        # Unicode
        if self.cfg.unicode_form:
            t = unicodedata.normalize(self.cfg.unicode_form, t)
        # NBSP -> espacio
        if self.cfg.normalize_nbsp:
            t = t.replace("\u00A0", " ")
        # Zero-width
        if self.cfg.remove_zero_width:
            for zw in _ZW_CHARS:
                t = t.replace(zw, "")
        # Control chars (excepto \n, \t)
        if self.cfg.strip_control_chars:
            t = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", t)
        # Puntuación en negrita aislada (**,** -> ,)
        if self.cfg.strip_bold_punctuation:
            t = re.sub(r"\*\*([.,;:!?])\*\*", r"\1", t)
        # mailto rotos -> email simple
        if self.cfg.simplify_mailto_links:
            # [lo-que-se-ve](mailto:algo@dominio) -> algo@dominio (o lo-que-se-ve si parece email)
            t = re.sub(r"\[([^\]]+)\]\(mailto:([^)]+)\)", lambda m: m.group(2) if "@" in m.group(2) else m.group(1), t)
        return t

    def _normalize_paragraph(self, text: str) -> str:
        t = text

        # <br> -> salto de línea (mejor para embeddings)
        if self.cfg.html_br_to_newline_in_paragraphs:
            t = t.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")

        placeholder = "<<<PARA_BREAK>>>"
        if self.cfg.keep_double_newlines:
            t = t.replace("\r\n", "\n")
            # preserva párrafos como bloques
            t = re.sub(r"\n{2,}", placeholder, t)

        # Une saltos "suaves": letra/num + \n + minúscula/dígito
        if self.cfg.join_soft_linebreaks:
            t = re.sub(r"(?<=\w)\n(?=[a-záéíóúñ0-9])", " ", t)

        # Repara guiones de corte palabra-\ncontinuación -> palabracontinuación
        if self.cfg.fix_broken_hyphens:
            t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

        # Colapsa espacios en medio de línea (NO al inicio)
        if self.cfg.collapse_whitespace:
            # múltiplos espacios/tabs en medio -> 1 espacio
            t = re.sub(r"(?<!^)[ \t]+(?!$)", " ", t, flags=re.MULTILINE)
            # quita espacios antes de salto
            t = re.sub(r"[ \t]+\n", "\n", t)
            # quita trailing spaces por línea (conserva leading)
            t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)

        if self.cfg.keep_double_newlines:
            t = t.replace(placeholder, "\n\n")

        return t.strip()

    def _normalize_quote_block(self, text: str) -> str:
        # Mantén el prefijo '>' por línea; limpieza suave
        lines = text.splitlines()
        out = [line.rstrip() for line in lines]  # trailing
        t = "\n".join(out)

        if self.cfg.fix_broken_hyphens:
            t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

        if self.cfg.collapse_whitespace:
            t = re.sub(r"(?<!^)[ \t]+(?!$)", " ", t, flags=re.MULTILINE)
            t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)

        return t.strip()

    def _normalize_list_block(self, text: str) -> str:
        """
        - Normaliza bullets raros (✓ -> -)
        - Colapsa indentación inicial excesiva a como mucho `list_left_pad_max` espacios.
        - NO une ítems; respeta sangrías (no tocar leading salvo colapso).
        """
        lines = text.splitlines()
        norm_lines: List[str] = []
        for line in lines:
            orig = line

            # Mantener líneas en blanco
            if not line.strip():
                norm_lines.append("")
                continue

            # Normaliza bullets raros a '- '
            if self.cfg.normalize_checkmark_bullets:
                line = re.sub(r'^(\s*)✓\s+', r'\1- ', line)

            # Colapsa sangrías excesivas de items (evita que "        - " rompa el MD)
            m = re.match(r'^(\s+)([-*+]\s+|\d+\.\s+)', line)
            if m:
                pad = m.group(1)
                bullet = m.group(2)
                if len(pad) > self.cfg.list_left_pad_max:
                    line = " " * self.cfg.list_left_pad_max + bullet + line[m.end():]

            # Limpia espacios antes de salto / trailing
            line = re.sub(r"[ \t]+$", "", line)

            norm_lines.append(line)

        t = "\n".join(norm_lines)

        # Quita espacios antes de salto globales
        t = re.sub(r"[ \t]+\n", "\n", t)
        return t.strip()

    def _normalize_table_block(self, text: str) -> str:
        # Respeta pipes y <br> en celdas; limpia trailing por línea
        t = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        return t

    def _normalize_code_block(self, text: str) -> str:
        # No tocar contenido de código
        return text

    def _normalize_heading(self, text: str) -> str:
        # Encabezados: quita **negritas** internas comunes y trailing
        line = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        if self.cfg.strip_heading_bold:
            # "# **Titulo** **(algo)**" -> "# Titulo (algo)"
            def _strip_inner_bold(s: str) -> str:
                return re.sub(r"\*\*(.+?)\*\*", r"\1", s)
            # solo procesa después del '#...'
            m = re.match(r'^(#{1,6}\s+)(.*)$', line)
            if m:
                line = m.group(1) + _strip_inner_bold(m.group(2))
        return line

    # ---- Utilidades de documento (por páginas) ----
    def _collapse_blank_lines(self, t: str) -> str:
        # Colapsa más de N saltos en bruto (preserva dobles para párrafos)
        max_n = max(1, self.cfg.max_consecutive_blank_lines)
        pattern = r"\n{%d,}" % (max_n + 1)
        t = re.sub(pattern, "\n" * max_n, t)
        # Limpia espacios en líneas vacías
        t = re.sub(r"^[ \t]+$", "", t, flags=re.MULTILINE)
        return t.strip()

    def _detect_common_headers_footers(self, pages: List[str]) -> tuple[set[str], set[str]]:
        """
        Heurística simple: toma top/bottom 5 líneas de cada página;
        si una línea (tras strip) aparece en >= 50% de páginas y es corta
        (<= 120 chars), la marca para eliminar.
        """
        def head_tail_lines(s: str, k: int = 5) -> tuple[List[str], List[str]]:
            lines = [ln.strip() for ln in s.splitlines() if ln.strip() != ""]
            head = lines[:k]
            tail = lines[-k:] if lines else []
            return head, tail

        heads: List[str] = []
        tails: List[str] = []
        for p in pages:
            h, t = head_tail_lines(p, 5)
            heads.extend(h)
            tails.extend(t)

        def frequent(lines: Iterable[str], n_pages: int) -> set[str]:
            from collections import Counter
            c = Counter([ln for ln in lines if 0 < len(ln) <= 120])
            threshold = max(2, int(0.5 * n_pages))  # >=50%
            return {ln for ln, cnt in c.items() if cnt >= threshold}

        return frequent(heads, len(pages)), frequent(tails, len(pages))

    def _drop_headers_footers(self, t: str, headers: set[str], footers: set[str]) -> str:
        lines = t.splitlines()
        # drop en top
        i = 0
        while i < len(lines) and lines[i].strip() in headers:
            i += 1
        # drop en bottom
        j = len(lines) - 1
        while j >= i and lines[j].strip() in footers:
            j -= 1
        kept = lines[i:j+1] if i <= j else []
        return "\n".join(kept).strip()
