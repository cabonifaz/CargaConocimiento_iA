from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from typing import List

_ZW_CHARS = [
    "\u200B",  # zero width space
    "\u200C",  # zero width non-joiner
    "\u200D",  # zero width joiner
    "\uFEFF",  # zero width no-break space (BOM)
]

@dataclass(frozen=True)
class NormalizerConfig:
    unicode_form: unicodedata._NormalizationForm = "NFC"          # "NFC" / "NFKC"
    collapse_whitespace: bool = True   # colapsar espacios en blanco consecutivos
    strip_control_chars: bool = True   # remover \x00-\x1F excepto \n\t
    normalize_nbsp: bool = True        # reemplaza NBSP por espacio normal
    remove_zero_width: bool = True     # elimina \u200B, etc.
    fix_broken_hyphens: bool = True    # repara guiones de corte de línea
    join_soft_linebreaks: bool = True  # une líneas separadas artificialmente
    keep_double_newlines: bool = True  # conserva párrafos (doble salto)

class TextNormalizer:
    def __init__(self, cfg: NormalizerConfig | None = None) -> None:
        self.cfg = cfg or NormalizerConfig()

    def normalize_pages(self, pages: List[str]) -> List[str]:
        """Normaliza cada página independientemente (mismo largo que 'pages')."""
        return [self._normalize_page(p) for p in pages]

    def normalize_and_join(self, pages: List[str]) -> str:
        """Normaliza y devuelve un único string (todas las páginas unidas)."""
        normed_pages = self.normalize_pages(pages)
        # Unir páginas con salto de página lógico
        return "\n\n".join(normed_pages).strip()

    # Helpers

    def _normalize_page(self, text: str) -> str:
        t = text or ""

        # 1) Unicode canonical / compatibility normalization
        if self.cfg.unicode_form:
            t = unicodedata.normalize(self.cfg.unicode_form, t)

        # 2) Normaliza NBSP (no-break space) a espacio regular
        if self.cfg.normalize_nbsp:
            t = t.replace("\u00A0", " ")

        # 3) Remueve zero-width chars
        if self.cfg.remove_zero_width:
            for zw in _ZW_CHARS:
                t = t.replace(zw, "")

        # 4) Remueve caracteres de control (excepto \n, \t)
        if self.cfg.strip_control_chars:
            t = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", t)

        # 5) Repara guiones de final de línea (hyphenation) si parecen cortes
        #    Regla heurística: 'palabra-\ncontinuacion' -> 'palabracontinuacion'
        if self.cfg.fix_broken_hyphens:
            t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

        # 6) Unir saltos de línea “suaves” en el mismo párrafo
        #    Heurística: si una línea termina con letra/número y la siguiente inicia en minúscula,
        #    probablemente era un wrap visual, lo convertimos a espacio.
        if self.cfg.join_soft_linebreaks:
            # Primero, preserva doble salto de línea con placeholder
            placeholder = "<<<PARA_BREAK>>>"
            if self.cfg.keep_double_newlines:
                t = t.replace("\r\n", "\n")
                t = re.sub(r"\n{2,}", lambda m: placeholder, t)
            # Une líneas suaves
            t = re.sub(r"(?<=\w)\n(?=[a-záéíóúñ0-9])", " ", t)
            # Restaura párrafos
            if self.cfg.keep_double_newlines:
                t = t.replace(placeholder, "\n\n")

        # 7) Colapsa espacios en blanco redundantes (sin tocar dobles \n\n)
        if self.cfg.collapse_whitespace:
            # Colapsa múltiples espacios/tabs
            t = re.sub(r"[ \t]+", " ", t)
            # Limpia espacios antes de saltos
            t = re.sub(r"[ \t]+\n", "\n", t)
            # Trim de cada línea
            t = "\n".join(line.strip() for line in t.splitlines())

        # Trim global final
        return t.strip()
