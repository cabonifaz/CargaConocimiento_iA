from __future__ import annotations
from pathlib import Path
import pathlib
import re
from typing import List
from app.domain.services.md_text_normalizer import MdTextNormalizer, MdNormalizerConfig


class OcrMdTextNormalizer(MdTextNormalizer):
    """
    Normalizer especializado para markdown generado por Mistral OCR.
    Extiende MdTextNormalizer con normalizacion especial para tablas con listas en celdas.

    Reemplaza saltos de linea entre elementos de lista dentro de celdas de tabla
    por <br/>, evitando agregar <br/> al inicio o final del contenido de la celda.
    """

    def __init__(self, cfg: MdNormalizerConfig | None = None) -> None:
        super().__init__(cfg)

    def _normalize_table_block(self, text: str) -> str:
        """
        Normaliza tablas markdown con manejo especial para listas dentro de celdas.

        Identifica listas dentro de celdas (ej: | - item 1\n- item 2\n- item 3 |)
        y reemplaza los saltos de linea entre elementos por <br/>.
        """
        # Primero hacemos la limpieza basica del parent
        t = text

        # Remove bold asterisks from table content: **text** -> text
        t = re.sub(r'\*\*([^*]+)\*\*', r'\1', t)

        # Clean up multiple spaces
        t = re.sub(r"[ \t]+", " ", t)

        # Procesar cada linea de la tabla
        lines = t.splitlines()
        normalized_lines = []

        for line in lines:
            # Verificar si es una linea de separacion (|---|---|---|)
            if re.match(r'^\s*\|[\s\-\|]+\|\s*$', line):
                normalized_lines.append(line)
                continue

            # Verificar si es una fila de tabla
            if re.match(r'^\s*\|.*\|\s*$', line):
                normalized_line = self._normalize_table_row(line)
                normalized_lines.append(normalized_line)
            else:
                normalized_lines.append(line)

        result = "\n".join(normalized_lines)

        # Clean trailing spaces per line
        result = re.sub(r"[ \t]+$", "", result, flags=re.MULTILINE)

        return result

    def _normalize_table_row(self, row: str) -> str:
        """
        Normaliza una fila de tabla, reemplazando saltos de linea entre elementos
        de listas dentro de celdas por <br/>.

        Ejemplo:
        Input:  | - item 1\n- item 2\n- item 3 | otra celda |
        Output: | - item 1<br/>- item 2<br/>- item 3 | otra celda |
        """
        # Extraer el contenido entre pipes
        if not row.strip().startswith('|') or not row.strip().endswith('|'):
            return row

        # Remover pipes iniciales y finales
        content = row.strip()[1:-1]

        # Separar celdas por pipes (considerando que el contenido puede tener \n)
        # Usamos una estrategia diferente: procesar toda la linea
        cells = []
        parts = content.split('|')

        for part in parts:
            # Verificar si esta parte contiene elementos de lista
            if self._has_list_items(part):
                normalized_part = self._normalize_list_in_cell(part)
                cells.append(normalized_part)
            else:
                cells.append(part.strip())

        # Reconstruir la fila
        return '| ' + ' | '.join(cells) + ' |'

    def _has_list_items(self, cell_content: str) -> bool:
        """
        Detecta si el contenido de una celda tiene elementos de lista.

        Busca patrones como:
        - item
        * item
        + item
        1. item
        """
        # Buscar patrones de lista con saltos de linea
        list_pattern = r'[-*+]\s+[^\n]+\n\s*[-*+]\s+'  # lista no numerada
        numbered_list_pattern = r'\d+\.\s+[^\n]+\n\s*\d+\.\s+'  # lista numerada

        return bool(re.search(list_pattern, cell_content)) or \
               bool(re.search(numbered_list_pattern, cell_content))

    def _normalize_list_in_cell(self, cell_content: str) -> str:
        """
        Normaliza listas dentro de una celda, reemplazando saltos de linea
        entre elementos de lista por <br/>.

        NO agrega <br/> al inicio ni al final del contenido de la celda.
        """
        # Limpiar espacios al inicio y final
        content = cell_content.strip()

        if not content:
            return content

        # Patron para identificar saltos de linea entre elementos de lista
        # Busca: final de un elemento de lista (\n) seguido de otro elemento de lista
        # Patrones de lista no numerada: - item, * item, + item
        # Patrones de lista numerada: 1. item, 2. item, etc.

        # Reemplazar \n entre elementos de lista no numerada
        content = re.sub(
            r'([-*+]\s+[^\n]+)\n\s*([-*+]\s+)',
            r'\1<br/>\2',
            content
        )

        # Reemplazar \n entre elementos de lista numerada
        content = re.sub(
            r'(\d+\.\s+[^\n]+)\n\s*(\d+\.\s+)',
            r'\1<br/>\2',
            content
        )

        # Si quedan saltos de linea sueltos dentro de la celda (pero no al inicio/final),
        # tambien los reemplazamos por espacios o <br/> segun el contexto
        # Solo si hay mas de un elemento de lista
        if self._count_list_items(content) > 1:
            # Reemplazar saltos de linea residuales que no esten al inicio/final
            lines = content.split('\n')
            if len(lines) > 1:
                # Unir lineas que no son elementos de lista
                normalized_lines = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        if i > 0 and normalized_lines and not self._is_list_item_start(line):
                            # Unir con la linea anterior
                            normalized_lines[-1] += ' ' + line
                        else:
                            normalized_lines.append(line)

                content = '<br/>'.join(normalized_lines)

        return content

    def _count_list_items(self, text: str) -> int:
        """Cuenta el numero de elementos de lista en el texto."""
        # Contar elementos de lista no numerada
        count = len(re.findall(r'[-*+]\s+', text))
        # Sumar elementos de lista numerada
        count += len(re.findall(r'\d+\.\s+', text))
        return count

    def _is_list_item_start(self, line: str) -> bool:
        """Verifica si una linea comienza con un marcador de lista."""
        return bool(re.match(r'^\s*([-*+]|\d+\.)\s+', line))

    def normalize_pages(self, pages: List[str]) -> List[str]:
        """
        Override para agregar logging especifico de OCR normalizer.
        """
        print("### OcrMdTextNormalizer - normalize_pages (OCR Mistral con tablas especiales) -> List[str]: ------------------------------------------")
        result = super().normalize_pages(pages)

        # Reporte de normalizacion especifico para OCR
        path = Path(f"report/normalization/ocr_md_text_normalizer.txt")
        pathlib.Path(path.parent).mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, (orig, norm) in enumerate(zip(pages, result)):
                f.write(f"--- Pagina {i+1} (OCR Mistral) ---\n")
                f.write(f"Normalizada ({len(orig)} -> {len(norm)} chars):\n-----<INICIO>-----\n{norm}\n-----<FIN>-----\n")
                f.write("\n\n")
        print(f"Reporte de normalizacion OCR guardado en {path.resolve()}")

        return result
