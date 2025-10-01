from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re
import bisect
import hashlib
import json
from pathlib import Path

from app.ports.outbound.chunker import ChunkerPort, Chunk, ChunkerConfig
from app.ports.outbound.tokenizer import TokenCounterPort

class SemanticChunker:
    """Semantic chunking with 25% overlap that respects Markdown structure boundaries."""

    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None) -> None:
        self.tok = tokenizer
        self.cfg = cfg or ChunkerConfig()
        assert self.cfg.target_tokens > 0, "target_tokens must be > 0"

        # Metadata context for chunks
        self.current_filename: str = "documento.pdf"
        self.current_document_title: str = "Documento"
        self.section_hierarchy: List[str] = []
        self.current_table_headers: List[str] = []
        self.current_list_context: str = ""
    
    def chunk_document(self, pages: List[str], filename: str = "documento.pdf", document_title: str = "Documento") -> Tuple[List[Chunk], str]:
        """Perform semantic chunking with 25% overlap on joined document."""
        print("[CHUNKING] Método: Semántico con 25% overlap")

        # Set document context
        self.current_filename = filename
        self.current_document_title = document_title

        full_text, page_offsets = self._join_pages_and_offsets(pages)
        # Parse Markdown blocks for semantic boundaries and extract context
        md_blocks = self._parse_markdown_blocks(full_text)

        chunks = self._semantic_chunk_with_overlap(full_text, page_offsets, md_blocks)
        print(f"[CHUNKING] -> {len(chunks)} chunks semánticos generados")
        return chunks, full_text
    
    def _join_pages_and_offsets(self, pages: List[str]) -> Tuple[str, List[int]]:
        """Join pages with separator and return full text and page offsets."""
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
    
    def _generate_hierarchical_header(self, block_context: Dict[str, Any], page_num: int) -> str:
        """Generate hierarchical titles header for chunk."""
        # Use the hierarchy from the block context, not the global one
        hierarchy = block_context.get('hierarchy', [])
        if not hierarchy:
            return ""

        # Generate hierarchical titles
        headers = []
        for i, section_title in enumerate(hierarchy):
            level = i + 1  # Start from H1
            headers.append(f"{'#' * level} {section_title}")

        return "\n".join(headers)

    def _parse_markdown_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Parse Markdown blocks for semantic boundaries and extract context."""
        lines = text.splitlines(keepends=True)
        blocks: List[Dict[str, Any]] = []

        # Reset section hierarchy
        self.section_hierarchy = []

        # Track current position in text
        offset = 0
        i = 0
        
        while i < len(lines):
            line = lines[i]
            line_content = line.rstrip('\r\n')
            
            # Detect block type and boundaries
            block_start = offset
            block_lines = [line]
            block_type = self._classify_line(line_content)

            # Update section hierarchy for headings
            if block_type == 'heading':
                heading_level = self._get_heading_level(line_content)
                heading_text = re.sub(r'^#{1,6}\s+', '', line_content).strip()

                # Update hierarchy based on level
                if heading_level <= len(self.section_hierarchy):
                    # Replace or truncate hierarchy
                    self.section_hierarchy = self.section_hierarchy[:heading_level-1]
                    self.section_hierarchy.append(heading_text)
                else:
                    # Add to hierarchy
                    self.section_hierarchy.append(heading_text)

            # For structured blocks, consume until boundary
            if block_type == 'heading':
                # Headings are single-line
                pass
            elif block_type in ['code', 'table_json']:
                # Code fence - consume until closing fence
                if line_content.strip().startswith('```'):
                    json_content_lines = []
                    i += 1
                    offset += len(line)
                    while i < len(lines):
                        line = lines[i]
                        block_lines.append(line)
                        if not line.rstrip('\r\n').strip().startswith('```'):
                            json_content_lines.append(line.rstrip('\r\n'))
                        offset += len(line)
                        if line.rstrip('\r\n').strip().startswith('```'):
                            break
                        i += 1

                    # If it's a JSON table, extract headers
                    if block_type == 'table_json' and json_content_lines:
                        json_content = '\n'.join(json_content_lines)
                        self._extract_json_table_headers(json_content)
            elif block_type in ['table', 'list', 'quote']:
                # Extract context for tables and lists
                if block_type == 'table':
                    self._extract_table_headers(line_content)
                elif block_type == 'list':
                    self._extract_list_context(line_content)

                # Structured blocks - consume while same type
                i += 1
                offset += len(line)
                while i < len(lines):
                    line = lines[i]
                    line_content = line.rstrip('\r\n')
                    if not line_content.strip():  # Blank line ends structured block
                        break
                    if self._classify_line(line_content) != block_type:
                        i -= 1  # Back up one line
                        offset -= len(line)
                        break
                    block_lines.append(line)

                    # Update context for additional table/list items
                    if block_type == 'table' and '|' in line_content:
                        self._extract_table_headers(line_content)

                    offset += len(line)
                    i += 1
            else:  # paragraph
                # Consume until blank line or different block type
                i += 1
                offset += len(line)
                while i < len(lines):
                    line = lines[i]
                    line_content = line.rstrip('\r\n')
                    if not line_content.strip():  # Blank line
                        # Include blank line in paragraph but stop
                        block_lines.append(line)
                        offset += len(line)
                        break
                    new_type = self._classify_line(line_content)
                    if new_type != 'paragraph':
                        i -= 1  # Back up one line
                        offset -= len(line)
                        break
                    block_lines.append(line)
                    offset += len(line)
                    i += 1
            
            block_text = ''.join(block_lines)
            block_end = block_start + len(block_text)
            
            blocks.append({
                'type': block_type,
                'text': block_text,
                'start': block_start,
                'end': block_end,
                'level': self._get_heading_level(line_content) if block_type == 'heading' else 0,
                'hierarchy': self.section_hierarchy.copy()  # Captura la jerarquía actual
            })
            
            offset += len(line)
            i += 1
        
        return blocks

    def _extract_table_headers(self, line: str) -> None:
        """Extract table headers from first table row."""
        if not self.current_table_headers and '|' in line:
            # Clean up the line and extract headers
            headers = [h.strip() for h in line.split('|')]
            # Remove empty first/last elements from splitting
            headers = [h for h in headers if h and not h.replace('-', '').strip() == '']
            if headers:
                self.current_table_headers = headers[:5]  # Limit to first 5 headers

    def _extract_json_table_headers(self, json_content: str) -> None:
        """Extract table headers from JSON table content."""
        try:
            # Parse the JSON content
            data = json.loads(json_content)

            # Navigate to the table structure
            if 'table' in data and 'headers' in data['table']:
                headers = data['table']['headers']
                if isinstance(headers, list) and headers:
                    self.current_table_headers = headers[:5]  # Limit to first 5 headers
        except (json.JSONDecodeError, KeyError, TypeError):
            # If JSON parsing fails, leave headers empty
            pass

    def _extract_list_context(self, line: str) -> None:
        """Extract context from list item to understand what the list is about."""
        if not self.current_list_context:
            # Try to infer list context from first item
            clean_line = re.sub(r'^\s*[-*+]\s+', '', line.strip())
            clean_line = re.sub(r'^\s*\d+\.\s+', '', clean_line)

            # Extract key terms that might indicate list purpose
            if len(clean_line) > 10:
                # Take first meaningful words
                words = clean_line.split()
                if len(words) > 2:
                    self.current_list_context = ' '.join(words[:3]) + "..."

    def _classify_line(self, line: str) -> str:
        """Classify a line into a Markdown block type."""
        line = line.strip()

        if not line:
            return 'paragraph'

        # Heading
        if re.match(r'^#{1,6}\s+', line):
            return 'heading'

        # Code fence - check if it's a JSON table
        if line.startswith('```'):
            # Check if it's specifically a JSON table
            if line == '```json':
                return 'table_json'
            return 'code'

        # Traditional table row
        if line.startswith('|') and line.endswith('|'):
            return 'table'
        if '|' in line and line.count('|') >= 2:
            return 'table'

        # List item
        if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            return 'list'

        # Quote
        if line.startswith('>'):
            return 'quote'

        return 'paragraph'
    
    def _get_heading_level(self, line: str) -> int:
        """Get heading level (1-6) from heading line."""
        match = re.match(r'^(#{1,6})\s+', line.strip())
        return len(match.group(1)) if match else 0

    
    def _semantic_chunk_with_overlap(self, text: str, page_offsets: List[int], blocks: List[Dict[str, Any]]) -> List[Chunk]:
        """Create semantic chunks with 25% overlap."""
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        target_tokens = self.cfg.target_tokens
        min_tokens = self.cfg.min_tokens

        # First, extract standalone JSON table chunks (adicionales)
        json_table_chunks = self._create_json_table_chunks(text, page_offsets, blocks)
        chunks.extend(json_table_chunks)

        # No excluir posiciones de tablas markdown de chunks normales
        # Las tablas markdown seguirán siendo parte del texto normal

        # Start from beginning
        current_pos = 0
        
        while current_pos < len(text):
            # Find optimal chunk boundary starting from current_pos
            chunk_start = current_pos
            chunk_end, is_semantic_boundary = self._find_optimal_boundary(
                text, blocks, chunk_start, target_tokens
            )
            
            # Extract chunk text
            chunk_text = text[chunk_start:chunk_end].strip()
            
            if not chunk_text:
                break
            
            # Count tokens
            token_count = self.tok.count_tokens(chunk_text)
            
            # Handle tiny chunks by merging with previous if possible
            if token_count < min_tokens and chunks:
                last_chunk = chunks[-1]
                merged_text = last_chunk.text + "\n\n" + chunk_text
                merged_tokens = self.tok.count_tokens(merged_text)
                
                # If merged chunk is reasonable size, update the last chunk
                if merged_tokens <= target_tokens * 1.5:  # Allow some flexibility
                    page_start, page_end = self._pages_for_span(page_offsets, last_chunk.char_start, chunk_end)
                    # Extract raw text from previous chunk (remove metadata)
                    prev_raw_text = self._extract_content_from_chunk(last_chunk.text)
                    merged_raw_text = prev_raw_text + "\n\n" + chunk_text

                    # Find best block context for merged chunk
                    merged_context = self._find_block_context_for_span(blocks, last_chunk.char_start, chunk_end)
                    enhanced_merged = self._create_chunk(merged_raw_text, last_chunk.char_start, chunk_end, page_start, page_end, merged_context)
                    chunks[-1] = enhanced_merged
                else:
                    # Create separate chunk even if small
                    page_start, page_end = self._pages_for_span(page_offsets, chunk_start, chunk_end)
                    block_context = self._find_block_context_for_span(blocks, chunk_start, chunk_end)
                    chunks.append(self._create_chunk(chunk_text, chunk_start, chunk_end, page_start, page_end, block_context))
            else:
                # Create normal chunk
                page_start, page_end = self._pages_for_span(page_offsets, chunk_start, chunk_end)
                block_context = self._find_block_context_for_span(blocks, chunk_start, chunk_end)
                chunks.append(self._create_chunk(chunk_text, chunk_start, chunk_end, page_start, page_end, block_context))
            
            # Calculate 25% overlap for next chunk
            if chunk_end >= len(text):
                break
            
            # Find overlap start position (25% back from end)
            overlap_tokens = max(1, int(token_count * 0.25))
            overlap_start = self._find_overlap_start(text, chunk_start, chunk_end, overlap_tokens)
            
            # Protection against infinite loops
            if overlap_start <= current_pos:
                overlap_start = current_pos + 1
            
            current_pos = overlap_start
        
        return chunks

    def _create_json_table_chunks(self, text: str, page_offsets: List[int], blocks: List[Dict[str, Any]]) -> List[Chunk]:
        """Create dedicated JSON chunks for all tables (both markdown and table_json)."""
        table_chunks = []

        for block in blocks:
            if block['type'] == 'table':
                # Reset headers for each table
                self.current_table_headers = []

                # Extract markdown table and convert to JSON
                block_text = block['text']
                self._extract_table_headers(block_text.split('\n')[0] if '\n' in block_text else block_text)

                try:
                    # Parse markdown table
                    table_data = self._parse_markdown_table(block_text)
                    json_content = self._table_to_json_string(table_data)

                    # Create JSON chunk for this markdown table
                    page_start, page_end = self._pages_for_span(page_offsets, block['start'], block['end'])
                    block_context = {'type': 'table_json', 'hierarchy': block.get('hierarchy', [])}

                    chunk = self._create_chunk(
                        json_content,
                        block['start'],
                        block['end'],
                        page_start,
                        page_end,
                        block_context
                    )
                    table_chunks.append(chunk)
                except Exception as e:
                    print(f"Error converting markdown table to JSON: {e}")

            elif block['type'] == 'table_json':
                # Reset headers for each table
                self.current_table_headers = []

                # Extract the JSON content from the block
                block_text = block['text']

                # Extract JSON content from the code block
                lines = block_text.strip().split('\n')
                json_lines = []
                in_json = False

                for line in lines:
                    if line.strip() == '```json':
                        in_json = True
                        continue
                    elif line.strip() == '```':
                        break
                    elif in_json:
                        json_lines.append(line)

                if json_lines:
                    json_content = '\n'.join(json_lines)
                    self._extract_json_table_headers(json_content)

                # Create chunk for this JSON table
                page_start, page_end = self._pages_for_span(page_offsets, block['start'], block['end'])
                block_context = {'type': 'table_json', 'hierarchy': block.get('hierarchy', [])}

                chunk = self._create_chunk(
                    block_text.strip(),
                    block['start'],
                    block['end'],
                    page_start,
                    page_end,
                    block_context
                )
                table_chunks.append(chunk)

        return table_chunks

    def _is_position_covered(self, pos: int, covered_ranges: List[Tuple[int, int]]) -> bool:
        """Check if a position is already covered by existing chunks."""
        for start, end in covered_ranges:
            if start <= pos < end:
                return True
        return False

    def _adjust_end_for_tables(self, start: int, end: int, covered_ranges: List[Tuple[int, int]]) -> int:
        """Adjust chunk end to avoid overlapping with table chunks."""
        for table_start, table_end in covered_ranges:
            # If chunk would overlap with a table, cut it short
            if start < table_start < end:
                return table_start
        return end

    def _find_optimal_boundary(self, text: str, blocks: List[Dict[str, Any]], start_pos: int, target_tokens: int) -> Tuple[int, bool]:
        """Find optimal chunk boundary respecting semantic structure."""
        # Start with token-based boundary
        tokens = self.tok.tokenize(text[start_pos:start_pos + target_tokens * 8])  # Rough estimate
        if not tokens:
            return len(text), False
        
        # Find blocks that intersect our target range
        target_end = start_pos + sum(token[1] - token[0] for token in tokens[:target_tokens])
        target_end = min(target_end, len(text))
        
        # Look for semantic boundaries near target end
        best_boundary = target_end
        is_semantic = False
        
        # Check for heading boundaries (highest priority)
        for block in blocks:
            if block['type'] == 'heading' and block['start'] <= target_end + target_tokens * 2:
                if block['start'] >= start_pos + min(target_tokens // 2, 50):  # Not too close to start
                    boundary_tokens = self.tok.count_tokens(text[start_pos:block['start']])
                    if boundary_tokens >= target_tokens * 0.7:  # At least 70% of target
                        best_boundary = block['start']
                        is_semantic = True
                        break
        
        # If no heading found, look for end of structured blocks
        if not is_semantic:
            for block in blocks:
                if (block['type'] in ['table', 'table_json', 'list', 'code', 'quote'] and
                    block['end'] <= target_end + target_tokens and block['end'] >= start_pos):
                    boundary_tokens = self.tok.count_tokens(text[start_pos:block['end']])
                    if boundary_tokens >= target_tokens * 0.6:  # At least 60% of target
                        best_boundary = block['end']
                        is_semantic = True
                        break
        
        # If no semantic boundary found, use paragraph breaks
        if not is_semantic:
            paragraph_breaks = [m.end() for m in re.finditer(r'\n\s*\n', text[start_pos:target_end + 200])]
            for break_pos in reversed(paragraph_breaks):
                abs_break_pos = start_pos + break_pos
                boundary_tokens = self.tok.count_tokens(text[start_pos:abs_break_pos])
                if boundary_tokens >= target_tokens * 0.5:  # At least 50% of target
                    best_boundary = abs_break_pos
                    is_semantic = True
                    break
        
        return min(best_boundary, len(text)), is_semantic
    
    def _find_overlap_start(self, text: str, chunk_start: int, chunk_end: int, overlap_tokens: int) -> int:
        """Find the start position for overlap (25% back from chunk end)."""
        chunk_text = text[chunk_start:chunk_end]
        tokens = self.tok.tokenize(chunk_text)
        
        if len(tokens) <= overlap_tokens:
            return chunk_start
        
        # Find position that corresponds to last overlap_tokens
        overlap_start_in_chunk = tokens[-overlap_tokens][0]
        return chunk_start + overlap_start_in_chunk
    
    def _pages_for_span(self, page_offsets: List[int], start_char: int, end_char: int) -> Tuple[int, int]:
        """Get 1-based page numbers that span covers."""
        ps = bisect.bisect_right(page_offsets, start_char) - 1
        ps = max(ps, 0)
        ec = max(end_char - 1, 0)
        pe = bisect.bisect_right(page_offsets, ec) - 1
        pe = max(pe, 0)
        return ps + 1, pe + 1
    
    def _create_chunk(self, text: str, char_start: int, char_end: int, page_start: int, page_end: int, block_context: Dict[str, Any] = None) -> Chunk:
        """Create a Chunk object with hierarchical titles."""
        # Generate hierarchical header
        context_info = block_context or {'type': 'paragraph'}
        hierarchical_header = self._generate_hierarchical_header(context_info, page_start)

        # Combine hierarchical titles with actual content
        if hierarchical_header:
            enhanced_text = f"{hierarchical_header}\n{text}"
        else:
            enhanced_text = text

        return Chunk(
            text=enhanced_text,
            token_count=self.tok.count_tokens(enhanced_text),
            chunk_id=self._hash(enhanced_text),
            char_start=char_start,
            char_end=char_end,
            page_start=page_start,
            page_end=page_end,
        )

    def _extract_content_from_chunk(self, chunk_text: str) -> str:
        """Extract the actual content from a chunk, removing hierarchical headers."""
        # Since we're now using hierarchical headers, we need a different approach
        # to extract just the content without the repeated titles
        lines = chunk_text.split('\n')

        # Skip heading lines at the beginning
        content_start = 0
        for i, line in enumerate(lines):
            if not line.strip().startswith('#'):
                content_start = i
                break

        if content_start > 0:
            return '\n'.join(lines[content_start:]).strip()
        return chunk_text

    def _find_block_context_for_span(self, blocks: List[Dict[str, Any]], start_char: int, end_char: int) -> Dict[str, Any]:
        """Find the most appropriate block context for a character span."""
        # Find blocks that intersect with the span
        intersecting_blocks = []
        for block in blocks:
            if (block['start'] < end_char and block['end'] > start_char):
                intersecting_blocks.append(block)

        if not intersecting_blocks:
            # Find the nearest heading hierarchy by looking at blocks before this span
            nearest_hierarchy = self._find_nearest_hierarchy(blocks, start_char)
            return {'type': 'paragraph', 'hierarchy': nearest_hierarchy}

        # Find the block with the most specific hierarchy (longest hierarchy list)
        best_block = intersecting_blocks[0]
        for block in intersecting_blocks:
            # Prefer blocks with longer hierarchy (more specific context)
            if len(block.get('hierarchy', [])) > len(best_block.get('hierarchy', [])):
                best_block = block
            # If same hierarchy length, prefer structured content over paragraphs
            elif len(block.get('hierarchy', [])) == len(best_block.get('hierarchy', [])):
                priority_order = ['heading', 'table', 'table_json', 'list', 'code', 'quote', 'paragraph']
                if (priority_order.index(block['type']) if block['type'] in priority_order else 999) < \
                   (priority_order.index(best_block['type']) if best_block['type'] in priority_order else 999):
                    best_block = block

        return best_block

    def _find_nearest_hierarchy(self, blocks: List[Dict[str, Any]], position: int) -> List[str]:
        """Find the nearest heading hierarchy before the given position."""
        nearest_hierarchy = []
        for block in reversed(blocks):
            if block['start'] < position and block['type'] == 'heading':
                # Found a heading before this position, use its hierarchy
                hierarchy = block.get('hierarchy', [])
                if hierarchy:
                    nearest_hierarchy = hierarchy
                    break
        return nearest_hierarchy

    def _parse_markdown_table(self, table_text: str) -> Dict[str, Any]:
        """
        Parsea una tabla markdown y la convierte a estructura JSON.

        Formato de entrada esperado:
        | Header1 | Header2 | Header3 |
        |---------|---------|---------|
        | Cell1   | Cell2   | Cell3   |
        | Cell4   | Cell5   | Cell6   |

        Formato de salida:
        {
            "headers": ["Header1", "Header2", "Header3"],
            "rows": [
                ["Cell1", "Cell2", "Cell3"],
                ["Cell4", "Cell5", "Cell6"]
            ]
        }
        """
        lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]

        if len(lines) < 2:
            return {"headers": [], "rows": []}

        headers = []
        rows = []
        separator_found = False

        for i, line in enumerate(lines):
            # Verificar si es una línea de separación (|---|---|---|)
            if re.match(r'^\s*\|[\s\-\|]+\|\s*$', line):
                separator_found = True
                continue

            # Extraer celdas de la línea
            cells = self._extract_table_cells(line)

            if not cells:
                continue

            if i == 0:
                # Primera línea = headers
                headers = cells
            elif separator_found:
                # Líneas después del separador = datos
                rows.append(cells)
            elif i > 0:
                # Si no hay separador pero hay más líneas, tratarlas como datos
                rows.append(cells)

        return {
            "headers": headers,
            "rows": rows
        }

    def _extract_table_cells(self, line: str) -> List[str]:
        """
        Extrae las celdas de una línea de tabla markdown.
        Ejemplo: "| Cell1 | Cell2 | Cell3 |" -> ["Cell1", "Cell2", "Cell3"]
        """
        # Remover pipes del inicio y final
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]

        # Dividir por pipes y limpiar espacios
        cells = [cell.strip() for cell in line.split('|')]

        # Filtrar celdas vacías al principio y final
        while cells and not cells[0]:
            cells.pop(0)
        while cells and not cells[-1]:
            cells.pop()

        return cells

    def _table_to_json_string(self, table_data: Dict[str, Any]) -> str:
        """
        Convierte la estructura de tabla a una representación JSON compacta.
        """
        if not table_data["headers"] and not table_data["rows"]:
            return "```json\n{}\n```"

        # Crear estructura más legible para el JSON
        result = {
            "table": {
                "headers": table_data["headers"],
                "rows": table_data["rows"],
                "row_count": len(table_data["rows"]),
                "column_count": len(table_data["headers"]) if table_data["headers"] else 0
            }
        }

        # Formatear JSON compacto usando stringify
        json_str = json.dumps(result, ensure_ascii=False, separators=(',', ':'))

        # Envolver en bloque de código para mejor visualización
        return f"```json\n{json_str}\n```"

    @staticmethod
    def _hash(text: str) -> str:
        """Generate hash for chunk ID."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkSemanticAdapter(ChunkerPort):
    """Adapter for semantic chunking with 25% overlap."""
    
    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None) -> None:
        """Initialize semantic chunker with tokenizer and configuration."""
        self._chunker = SemanticChunker(tokenizer, cfg)
    
    def chunk_document(self, pages: List[str], filename: str = "documento.pdf", document_title: str = "Documento") -> Tuple[List[Chunk], str]:
        """Chunk document using semantic boundaries with 25% overlap."""
        return self._chunker.chunk_document(pages, filename, document_title)
    
    def get_config(self) -> ChunkerConfig:
        """Get the current chunker configuration."""
        return self._chunker.cfg