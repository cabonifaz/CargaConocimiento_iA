from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List

import re
import pathlib
import pymupdf
import pymupdf4llm
from app.ports.outbound.text_extractor import TextExtractorPort, TextExtractionResult


@dataclass
class PyMuPDFConfig:
    max_pages: Optional[int] = None
    sort_text: bool = True
    codec: str = "utf-8"


class PyMuPDFTextExtractor(TextExtractorPort):
    """Extractor de texto usando PyMuPDF. Trabaja SOBRE BYTES."""

    def __init__(self, config: Optional[PyMuPDFConfig] = None) -> None:
        self.cfg = config or PyMuPDFConfig()

    def _is_slide_application(self, metadata: dict) -> tuple[bool, str]:
        """Detecta si el PDF fue creado desde una aplicación de slides."""
        # Indicadores de aplicaciones de slides
        slide_indicators = [
            # Microsoft PowerPoint
            'microsoft office powerpoint', 'powerpoint', 'pptx', 'ppt',
            # Google Slides
            'google slides', 'docs.google.com',
            # Canva
            'canva', 'canva.com',
            # Otros
            'keynote', 'prezi', 'slides', 'presentation',
            'libre office impress', 'openoffice impress'
        ]

        # Obtener metadatos
        creator = (metadata.get('creator') or metadata.get('Creator') or '').lower()
        producer = (metadata.get('producer') or metadata.get('Producer') or '').lower()
        title = (metadata.get('title') or metadata.get('Title') or '').lower()
        subject = (metadata.get('subject') or metadata.get('Subject') or '').lower()

        # Combinar toda la metadata en un string
        all_metadata = f"{creator} {producer} {title} {subject}"

        # Buscar indicadores en metadata
        for indicator in slide_indicators:
            if indicator in all_metadata:
                return True, f"Detectado por metadata: {indicator}"

        return False, "No detectado como aplicación de slides"

    def _extract_tables_from_slides(self, doc, limit: int) -> dict:
        """Extrae tablas de un PDF de slides usando pymupdf4llm sin ignore_graphics."""
        tables_by_page = {}

        for i in range(limit):
            try:
                # Extraer con ignore_graphics=False para capturar tablas
                table_content = pymupdf4llm.to_markdown(
                    doc,
                    pages=[i],
                    detect_bg_color=False,
                    ignore_alpha=True,
                    write_images=False,
                    force_text=True,
                    ignore_images=True,
                    ignore_graphics=False,  # Importante: False para capturar tablas
                    margins=0,
                    use_glyphs=True
                )

                # Extraer solo las tablas del contenido
                tables = self._extract_tables_from_content(table_content)
                if tables:
                    tables_by_page[i] = tables
                    print(f"\rPage {i+1}: {len(tables)} tabla(s) encontrada(s)", end='', flush=True)

            except Exception as e:
                print(f"\rError extrayendo tablas página {i+1}: {e}", end='', flush=True)
                continue

        if tables_by_page:
            print(f"\nTotal: {len(tables_by_page)} páginas con tablas")
        else:
            print("\nNo se encontraron tablas en el documento")

        return tables_by_page

    def _extract_table_with_native_api(self, page, page_num: int) -> list:
        """
        Extrae tablas usando la API nativa de PyMuPDF que maneja mejor celdas combinadas.
        Devuelve una lista de tablas en formato markdown con información adicional sobre merged cells.
        """
        tables = []

        try:
            # Buscar tablas en la página
            tabs = page.find_tables()

            if not tabs or not tabs.tables:
                return []

            for table_idx, table in enumerate(tabs.tables):
                try:
                    # Extraer datos de la tabla
                    table_data = table.extract()

                    if not table_data or len(table_data) < 2:
                        continue

                    # Verificar si hay celdas combinadas
                    has_merged_cells = self._has_merged_cells(table)

                    if has_merged_cells:
                        # Para tablas con celdas combinadas, crear representación especial
                        table_md = self._build_table_markdown_with_merged_info(table_data, table)
                    else:
                        # Tabla simple sin celdas combinadas
                        table_md = self._build_simple_table_markdown(table_data)

                    if table_md:
                        tables.append(table_md)

                except Exception as e:
                    print(f"\nError procesando tabla {table_idx} en página {page_num+1}: {e}")
                    continue

        except Exception as e:
            print(f"\nError buscando tablas en página {page_num+1}: {e}")

        return tables

    def _has_merged_cells(self, table) -> bool:
        """Detecta si una tabla tiene celdas combinadas."""
        try:
            # PyMuPDF proporciona información sobre merged cells a través del objeto tabla
            for row_idx in range(table.row_count):
                for col_idx in range(table.col_count):
                    cell = table.cell(row_idx, col_idx)
                    # Si una celda tiene rowspan > 1 o colspan > 1, hay merged cells
                    if hasattr(cell, 'rowspan') and cell.rowspan > 1:
                        return True
                    if hasattr(cell, 'colspan') and cell.colspan > 1:
                        return True
        except:
            pass
        return False

    def _build_table_markdown_with_merged_info(self, table_data: list, table) -> str:
        """
        Construye markdown de tabla con información sobre celdas combinadas.
        Para celdas combinadas, repite el valor en todas las celdas afectadas.
        """
        if not table_data:
            return ""

        # Procesar filas
        processed_data = []

        # Mapear celdas combinadas
        merged_map = {}
        try:
            for row_idx in range(table.row_count):
                for col_idx in range(table.col_count):
                    cell = table.cell(row_idx, col_idx)
                    if hasattr(cell, 'rowspan') and hasattr(cell, 'colspan'):
                        rowspan = getattr(cell, 'rowspan', 1)
                        colspan = getattr(cell, 'colspan', 1)
                        if rowspan > 1 or colspan > 1:
                            merged_map[(row_idx, col_idx)] = {
                                'rowspan': rowspan,
                                'colspan': colspan,
                                'value': table_data[row_idx][col_idx] if row_idx < len(table_data) and col_idx < len(table_data[row_idx]) else ""
                            }
        except:
            # Si falla la detección, usar tabla simple
            return self._build_simple_table_markdown(table_data)

        # Construir tabla expandiendo celdas combinadas
        for row_idx, row in enumerate(table_data):
            processed_row = []
            for col_idx, cell in enumerate(row):
                # Verificar si esta celda es parte de una celda combinada
                source_cell = None
                for (m_row, m_col), info in merged_map.items():
                    if (m_row <= row_idx < m_row + info['rowspan'] and
                        m_col <= col_idx < m_col + info['colspan']):
                        source_cell = info['value']
                        break

                if source_cell is not None:
                    processed_row.append(source_cell)
                else:
                    processed_row.append(cell or "")

            processed_data.append(processed_row)

        return self._build_simple_table_markdown(processed_data)

    def _build_simple_table_markdown(self, table_data: list) -> str:
        """Construye markdown simple de tabla sin celdas combinadas."""
        if not table_data or len(table_data) < 1:
            return ""

        md_lines = []

        # Header (primera fila)
        header = table_data[0]
        md_lines.append("| " + " | ".join(str(cell or "").strip() for cell in header) + " |")

        # Separador
        md_lines.append("| " + " | ".join("---" for _ in header) + " |")

        # Filas de datos
        for row in table_data[1:]:
            # Asegurar que la fila tenga el mismo número de columnas
            row_cells = list(row) + [""] * (len(header) - len(row))
            md_lines.append("| " + " | ".join(str(cell or "").strip() for cell in row_cells[:len(header)]) + " |")

        return "\n".join(md_lines)

    def _extract_tables_from_content(self, content: str) -> list:
        """Extrae tablas markdown del contenido."""
        tables = []
        lines = content.split('\n')

        current_table = []
        in_table = False

        for line in lines:
            # Detectar inicio de tabla (línea con |)
            if '|' in line and line.strip():
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            else:
                # Fin de tabla
                if in_table and current_table:
                    # Validar que es una tabla real (al menos 2 filas)
                    if len(current_table) >= 2:
                        table_md = '\n'.join(current_table)
                        tables.append(table_md)
                    current_table = []
                    in_table = False

        # Capturar tabla al final del contenido
        if in_table and len(current_table) >= 2:
            table_md = '\n'.join(current_table)
            tables.append(table_md)

        return tables
    
    def _convert_to_markdown(self, text: str) -> str:
        """Convert plain text to basic markdown format."""
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue
            
            # Simple heuristics for markdown conversion
            # Detect potential headings (short lines, all caps, etc.)
            if len(line) < 60 and (line.isupper() or line.istitle()) and not line.endswith('.'):
                # Convert to heading
                markdown_lines.append(f"## {line}")
            # Detect list items (lines starting with numbers, letters, or bullets)
            elif re.match(r'^\s*[\d\w]{1,3}[.)]\s+', line):
                markdown_lines.append(f"- {line}")
            # Regular paragraph text
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)

    def extract_from_bytes(self, data: bytes, max_pages: Optional[int] = None) -> TextExtractionResult:
        print("### PyMuPDF Text Extractor - extract_from_bytes -> pages: List[str], page_count: int, producer: Optional[str]:")
        hard_cap = max_pages if max_pages is not None else self.cfg.max_pages

        doc = pymupdf.open(stream=data)

        try:
            if getattr(doc, "needs_pass", False):
                return TextExtractionResult(pages=[], page_count=0, producer=None)

            pages: List[str] = []
            page_total = doc.page_count  # número de páginas
            limit = min(page_total, hard_cap) if hard_cap is not None else page_total

            # Obtener metadatos para validar si es PDF de slides
            meta = doc.metadata or {}
            is_slide_pdf, detection_reason = self._is_slide_application(meta)
            ignore_graphics = is_slide_pdf

            print(f"PDF de slides detectado: {is_slide_pdf}")
            print(f"Razón: {detection_reason}")
            print(f"ignore_graphics: {ignore_graphics}")

            # Para PDFs de slides, extraer tablas por separado
            tables_by_page = {}
            if is_slide_pdf:
                print("Extrayendo tablas adicionales para PDF de slides...")
                tables_by_page = self._extract_tables_from_slides(doc, limit)

            for i in range(limit):
                print(f"\rExtracting page {i+1}/{limit}...", end='', flush=True)

                page = doc[i]

                # Try pymupdf4llm first for markdown formatting
                try:
                    md_page_text = pymupdf4llm.to_markdown(
                        doc,
                        pages=[i],
                        detect_bg_color=False,
                        ignore_alpha=True,
                        write_images=False,
                        force_text=True,
                        ignore_images=True,
                        ignore_graphics=ignore_graphics,  # Dinámico según detección
                        margins=0,
                        use_glyphs=True
                    )
                except Exception as e:
                    print(f"\rERROR Page {i+1}/{limit} - pymupdf4llm error: {e}")
                    md_page_text = ""

                # If pymupdf4llm fails, convert regular text to basic markdown
                if len(md_page_text.strip()) == 0:
                    regular_text = page.get_text()
                    if len(regular_text.strip()) > 0:
                        # Convert plain text to basic markdown format
                        md_page_text = self._convert_to_markdown(regular_text)
                        print(f"\rPage {i+1}/{limit} - Converted to markdown ({len(md_page_text)} chars)")
                    else:
                        print(f"\rWARNING Page {i+1}/{limit} - Truly empty page (0 chars)")
                else:
                    print(f"\rExtracted page {i+1}/{limit} ({len(md_page_text)} chars)")

                # Extraer tablas con API nativa (maneja mejor celdas combinadas)
                native_tables = self._extract_table_with_native_api(page, i)

                # Para PDFs de slides, agregar tablas si existen para esta página
                if is_slide_pdf and i in tables_by_page:
                    tables_for_page = tables_by_page[i]
                    tables_section = "\n\n## Tablas de la página\n\n" + "\n\n".join(tables_for_page)
                    md_page_text += tables_section
                    print(f" + {len(tables_for_page)} tabla(s) agregada(s)")
                elif native_tables:
                    # Para PDFs digitables, usar tablas extraídas con API nativa si existen
                    tables_section = "\n\n## Tablas de la página\n\n" + "\n\n".join(native_tables)
                    md_page_text += tables_section
                    print(f" + {len(native_tables)} tabla(s) nativa(s) agregada(s)")

                pages.append(md_page_text)

            # Obtener producer para compatibilidad con resultado
            producer_final = meta.get("producer") or meta.get("Producer")
            producer_final = producer_final if isinstance(producer_final, str) else None

            print(f"- {len(pages)} páginas extraídas (de {page_total})")

            for i, p in enumerate(pages):
                print(f"- Página {i+1} ({len(p)} chars):\n  {p[:50]!r}...")


            return TextExtractionResult(
                pages=pages,
                page_count=len(pages),
                producer=producer_final,
            )
        finally:
            doc.close()
