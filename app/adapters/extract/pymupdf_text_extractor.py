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
                    page = doc[i]
                    regular_text = page.get_text()
                    if len(regular_text.strip()) > 0:
                        # Convert plain text to basic markdown format
                        md_page_text = self._convert_to_markdown(regular_text)
                        print(f"\rPage {i+1}/{limit} - Converted to markdown ({len(md_page_text)} chars)")
                    else:
                        print(f"\rWARNING Page {i+1}/{limit} - Truly empty page (0 chars)")
                else:
                    print(f"\rExtracted page {i+1}/{limit} ({len(md_page_text)} chars)")

                # Para PDFs de slides, agregar tablas si existen para esta página
                if is_slide_pdf and i in tables_by_page:
                    tables_for_page = tables_by_page[i]
                    tables_section = "\n\n## Tablas de la página\n\n" + "\n\n".join(tables_for_page)
                    md_page_text += tables_section
                    print(f" + {len(tables_for_page)} tabla(s) agregada(s)")

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
