from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re
import bisect
import hashlib
import json

from app.ports.outbound.chunker import ChunkerPort, Chunk, ChunkerConfig
from app.ports.outbound.tokenizer import TokenCounterPort


class HierarchyManager:
    """Manages document hierarchy with semantic awareness for keywords."""

    HIERARCHY_KEYWORDS = {
        'chapter': [r'cap[íi]tulo\s+\d+', r'chapter\s+\d+'],
        'title': [r't[íi]tulo\s+[ivxlcdm]+', r'title\s+[ivxlcdm]+'],
        'article': [r'art[íi]culo\s+\d+', r'article\s+\d+'],
        'section': [r'secci[óo]n\s+\d+', r'section\s+\d+'],
        'subsection': [r'\d+\.\d+\.?\s', r'\d+\.\s'],
        'point': [r'punto\s+\d+', r'point\s+\d+'],
        'item': [r'item\s+\d+', r'inciso\s+[a-z]\)'],
    }

    def __init__(self):
        self.hierarchy: List[str] = []

    def update_hierarchy(self, heading_text: str, markdown_level: int) -> None:
        """Update hierarchy with semantic awareness."""
        heading_lower = heading_text.lower()

        # Check if this is a semantic keyword
        semantic_info = self._get_semantic_info(heading_lower)

        if semantic_info:
            effective_level = self._calculate_semantic_level(semantic_info, heading_text)
        else:
            effective_level = markdown_level

        # Update hierarchy
        if effective_level <= len(self.hierarchy):
            self.hierarchy = self.hierarchy[:effective_level-1]
            self.hierarchy.append(heading_text)
        else:
            self.hierarchy.append(heading_text)

    def _get_semantic_info(self, heading_lower: str) -> Optional[Dict[str, Any]]:
        """Identify semantic type and patterns."""
        for keyword_type, patterns in self.HIERARCHY_KEYWORDS.items():
            for pattern in patterns:
                if re.match(pattern, heading_lower):
                    return {
                        'type': keyword_type,
                        'pattern': pattern,
                        'text': heading_lower
                    }
        return None

    def _calculate_semantic_level(self, semantic_info: Dict[str, Any], heading_text: str) -> int:
        """Calculate effective level based on semantic rules."""
        keyword_type = semantic_info['type']
        pattern = semantic_info['pattern']

        # Find previous occurrences of same pattern type
        for i in range(len(self.hierarchy) - 1, -1, -1):
            prev_heading = self.hierarchy[i].lower()
            if re.match(pattern, prev_heading):
                return i + 1  # Same level as previous

        # Default levels for new keyword types
        level_map = {
            'chapter': 1,
            'title': 2,
            'article': 3,
            'section': 3,
            'subsection': 4,
            'point': 4,
            'item': 5
        }

        base_level = level_map.get(keyword_type, len(self.hierarchy) + 1)
        return min(base_level, 6)  # Max H6

    def get_hierarchy_headers(self) -> str:
        """Generate markdown headers for current hierarchy."""
        if not self.hierarchy:
            return ""

        headers = []
        for i, section_title in enumerate(self.hierarchy):
            level = i + 1
            headers.append(f"{'#' * level} {section_title}")

        return "\n".join(headers)

    def copy(self) -> List[str]:
        """Return a copy of current hierarchy."""
        return self.hierarchy.copy()


class TableProcessor:
    """Handles table detection and JSON conversion."""

    @staticmethod
    def parse_markdown_table(table_text: str) -> Dict[str, Any]:
        """Parse markdown table to JSON structure."""
        lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]

        if len(lines) < 2:
            return {"headers": [], "rows": []}

        headers = []
        rows = []
        separator_found = False

        for i, line in enumerate(lines):
            if re.match(r'^\s*\|[\s\-\|]+\|\s*$', line):
                separator_found = True
                continue

            cells = TableProcessor._extract_table_cells(line)
            if not cells:
                continue

            if i == 0:
                headers = cells
            elif separator_found or i > 0:
                rows.append(cells)

        return {"headers": headers, "rows": rows}

    @staticmethod
    def _extract_table_cells(line: str) -> List[str]:
        """Extract cells from table row."""
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]

        cells = [cell.strip() for cell in line.split('|')]

        # Remove empty cells at start/end
        while cells and not cells[0]:
            cells.pop(0)
        while cells and not cells[-1]:
            cells.pop()

        return cells

    @staticmethod
    def table_to_json_string(table_data: Dict[str, Any]) -> str:
        """Convert table to compact JSON string."""
        if not table_data["headers"] and not table_data["rows"]:
            return "```json\n{}\n```"

        result = {
            "table": {
                "headers": table_data["headers"],
                "rows": table_data["rows"],
                "row_count": len(table_data["rows"]),
                "column_count": len(table_data["headers"]) if table_data["headers"] else 0
            }
        }

        json_str = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        return f"```json\n{json_str}\n```"


class DocumentParser:
    """Parses document into semantic blocks with hierarchy context."""

    def __init__(self, hierarchy_manager: HierarchyManager):
        self.hierarchy = hierarchy_manager

    def parse_to_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Parse text into semantic blocks with hierarchy context."""
        lines = text.splitlines(keepends=True)
        blocks: List[Dict[str, Any]] = []

        self.hierarchy.hierarchy = []  # Reset

        offset = 0
        i = 0

        while i < len(lines):
            line = lines[i]
            line_content = line.rstrip('\r\n')

            block_start = offset
            block_lines = [line]
            block_type = self._classify_line(line_content)

            # Update hierarchy for headings
            if block_type == 'heading':
                heading_level = self._get_heading_level(line_content)
                heading_text = re.sub(r'^#{1,6}\s+', '', line_content).strip()
                self.hierarchy.update_hierarchy(heading_text, heading_level)

            # Consume block content
            i += 1
            offset += len(line)

            if block_type in ['table', 'list', 'quote']:
                # Consume similar lines
                while i < len(lines):
                    line = lines[i]
                    line_content = line.rstrip('\r\n')
                    if not line_content.strip() or self._classify_line(line_content) != block_type:
                        break
                    block_lines.append(line)
                    offset += len(line)
                    i += 1
            elif block_type == 'paragraph':
                # Consume until blank line or different type
                while i < len(lines):
                    line = lines[i]
                    line_content = line.rstrip('\r\n')
                    if not line_content.strip():
                        block_lines.append(line)
                        offset += len(line)
                        break
                    if self._classify_line(line_content) != 'paragraph':
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
                'hierarchy': self.hierarchy.copy()
            })

        return blocks

    def _classify_line(self, line: str) -> str:
        """Classify line type."""
        line = line.strip()

        if not line:
            return 'paragraph'

        if re.match(r'^#{1,6}\s+', line):
            return 'heading'

        if line.startswith('|') and line.endswith('|'):
            return 'table'
        if '|' in line and line.count('|') >= 2:
            return 'table'

        if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            return 'list'

        if line.startswith('>'):
            return 'quote'

        return 'paragraph'

    def _get_heading_level(self, line: str) -> int:
        """Get heading level from line."""
        match = re.match(r'^(#{1,6})\s+', line.strip())
        return len(match.group(1)) if match else 0


class SemanticChunkerOptimized:
    """Optimized semantic chunker with clear separation of concerns."""

    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None):
        self.tok = tokenizer
        self.cfg = cfg or ChunkerConfig()

        # Cohere API character limit
        self.max_chars = 2048

        self.current_filename: str = "documento.pdf"
        self.current_document_title: str = "Documento"

    def chunk_document(self, pages: List[str], filename: str = "documento.pdf",
                      document_title: str = "Documento") -> Tuple[List[Chunk], str]:
        """Main chunking method."""
        print("[CHUNKING] Método: Semántico optimizado con 25% overlap")

        self.current_filename = filename
        self.current_document_title = document_title

        # 1. Join pages
        full_text, page_offsets = self._join_pages(pages)

        # 2. Parse to semantic blocks
        hierarchy_manager = HierarchyManager()
        parser = DocumentParser(hierarchy_manager)
        blocks = parser.parse_to_blocks(full_text)

        # 3. Create chunks
        chunks = self._create_chunks(full_text, page_offsets, blocks)

        print(f"[CHUNKING] -> {len(chunks)} chunks generados")
        return chunks, full_text

    def _join_pages(self, pages: List[str]) -> Tuple[str, List[int]]:
        """Join pages and track offsets."""
        sep = self.cfg.page_separator
        offsets: List[int] = []
        parts: List[str] = []
        pos = 0

        for i, page in enumerate(pages):
            offsets.append(pos)
            parts.append(page)
            pos += len(page)
            if i != len(pages) - 1:
                parts.append(sep)
                pos += len(sep)

        return "".join(parts), offsets

    def _create_chunks(self, text: str, page_offsets: List[int],
                      blocks: List[Dict[str, Any]]) -> List[Chunk]:
        """Create all chunks (regular + table JSON)."""
        chunks: List[Chunk] = []

        # 1. Create JSON table chunks
        table_chunks = self._create_table_json_chunks(text, page_offsets, blocks)
        chunks.extend(table_chunks)

        # 2. Create regular content chunks with 25% overlap
        content_chunks = self._create_content_chunks_with_overlap(text, page_offsets, blocks)
        chunks.extend(content_chunks)

        return chunks

    def _create_table_json_chunks(self, text: str, page_offsets: List[int],
                                 blocks: List[Dict[str, Any]]) -> List[Chunk]:
        """Create dedicated JSON chunks for tables."""
        table_chunks = []

        for block in blocks:
            if block['type'] == 'table':
                try:
                    table_data = TableProcessor.parse_markdown_table(block['text'])
                    json_content = TableProcessor.table_to_json_string(table_data)

                    page_start, page_end = self._get_pages_for_span(page_offsets, block['start'], block['end'])
                    hierarchy_header = self._generate_hierarchy_header(block['hierarchy'])

                    chunk_text = f"{hierarchy_header}\n{json_content}" if hierarchy_header else json_content

                    # Validate character limit
                    chunk_text = self._validate_and_fix_chunk_length(chunk_text)

                    chunk = Chunk(
                        text=chunk_text,
                        token_count=self.tok.count_tokens(chunk_text),
                        chunk_id=self._hash(chunk_text),
                        char_start=block['start'],
                        char_end=block['end'],
                        page_start=page_start,
                        page_end=page_end,
                    )
                    table_chunks.append(chunk)

                except Exception as e:
                    print(f"Error processing table: {e}")

        return table_chunks

    def _create_content_chunks_with_overlap(self, text: str, page_offsets: List[int],
                                          blocks: List[Dict[str, Any]]) -> List[Chunk]:
        """Create content chunks with 25% overlap."""
        chunks: List[Chunk] = []
        target_tokens = self.cfg.target_tokens
        min_tokens = self.cfg.min_tokens

        current_pos = 0

        while current_pos < len(text):
            # Find chunk boundary
            chunk_start = current_pos
            chunk_end = self._find_optimal_boundary(text, blocks, chunk_start, target_tokens)

            chunk_text = text[chunk_start:chunk_end].strip()
            if not chunk_text:
                break

            # Find hierarchy context for this span
            hierarchy = self._find_hierarchy_for_span(blocks, chunk_start, chunk_end)
            hierarchy_header = self._generate_hierarchy_header(hierarchy)

            # Create enhanced text with hierarchy
            enhanced_text = f"{hierarchy_header}\n{chunk_text}" if hierarchy_header else chunk_text

            # Validate character limit
            enhanced_text = self._validate_and_fix_chunk_length(enhanced_text)

            token_count = self.tok.count_tokens(enhanced_text)

            # Handle small chunks by merging with previous
            if token_count < min_tokens and chunks:
                self._try_merge_with_previous(chunks, enhanced_text, chunk_start, chunk_end, page_offsets)
            else:
                page_start, page_end = self._get_pages_for_span(page_offsets, chunk_start, chunk_end)
                chunk = Chunk(
                    text=enhanced_text,
                    token_count=token_count,
                    chunk_id=self._hash(enhanced_text),
                    char_start=chunk_start,
                    char_end=chunk_end,
                    page_start=page_start,
                    page_end=page_end,
                )
                chunks.append(chunk)

            # Calculate 25% overlap for next chunk
            if chunk_end >= len(text):
                break

            overlap_start = self._calculate_overlap_start(text, chunk_start, chunk_end, token_count)
            current_pos = max(overlap_start, current_pos + 1)  # Prevent infinite loops

        return chunks

    def _find_optimal_boundary(self, text: str, blocks: List[Dict[str, Any]],
                              start_pos: int, target_tokens: int) -> int:
        """Find optimal chunk boundary."""
        # Rough estimate for target end position
        estimated_chars = target_tokens * 4  # Rough chars per token
        target_end = min(start_pos + estimated_chars, len(text))

        # Look for semantic boundaries
        best_boundary = target_end

        # Check for heading boundaries (highest priority)
        for block in blocks:
            if (block['type'] == 'heading' and
                block['start'] >= start_pos + 50 and  # Not too close to start
                block['start'] <= target_end + 200):  # Within reasonable range

                boundary_tokens = self.tok.count_tokens(text[start_pos:block['start']])
                if boundary_tokens >= target_tokens * 0.7:  # At least 70% of target
                    best_boundary = block['start']
                    break

        return min(best_boundary, len(text))

    def _find_hierarchy_for_span(self, blocks: List[Dict[str, Any]],
                                start_char: int, end_char: int) -> List[str]:
        """Find most appropriate hierarchy for character span."""
        best_hierarchy = []
        max_hierarchy_length = 0

        for block in blocks:
            if block['start'] < end_char and block['end'] > start_char:
                hierarchy = block.get('hierarchy', [])
                if len(hierarchy) > max_hierarchy_length:
                    max_hierarchy_length = len(hierarchy)
                    best_hierarchy = hierarchy

        return best_hierarchy

    def _generate_hierarchy_header(self, hierarchy: List[str]) -> str:
        """Generate hierarchy header from list."""
        if not hierarchy:
            return ""

        headers = []
        for i, section_title in enumerate(hierarchy):
            level = i + 1
            headers.append(f"{'#' * level} {section_title}")

        return "\n".join(headers)

    def _calculate_overlap_start(self, text: str, chunk_start: int, chunk_end: int, token_count: int) -> int:
        """Calculate 25% overlap start position."""
        overlap_tokens = max(1, int(token_count * 0.25))
        chunk_text = text[chunk_start:chunk_end]
        tokens = self.tok.tokenize(chunk_text)

        if len(tokens) <= overlap_tokens:
            return chunk_start

        overlap_start_in_chunk = tokens[-overlap_tokens][0]
        return chunk_start + overlap_start_in_chunk

    def _try_merge_with_previous(self, chunks: List[Chunk], new_text: str,
                                start_char: int, end_char: int, page_offsets: List[int]) -> None:
        """Try to merge small chunk with previous chunk."""
        if not chunks:
            return

        last_chunk = chunks[-1]
        merged_text = last_chunk.text + "\n\n" + new_text

        # Validate character limit for merged text
        merged_text = self._validate_and_fix_chunk_length(merged_text)

        merged_tokens = self.tok.count_tokens(merged_text)

        if merged_tokens <= self.cfg.target_tokens * 1.5:  # Allow some flexibility
            page_start, page_end = self._get_pages_for_span(page_offsets, last_chunk.char_start, end_char)

            updated_chunk = Chunk(
                text=merged_text,
                token_count=merged_tokens,
                chunk_id=self._hash(merged_text),
                char_start=last_chunk.char_start,
                char_end=end_char,
                page_start=page_start,
                page_end=page_end,
            )
            chunks[-1] = updated_chunk

    def _get_pages_for_span(self, page_offsets: List[int], start_char: int, end_char: int) -> Tuple[int, int]:
        """Get page numbers for character span."""
        ps = bisect.bisect_right(page_offsets, start_char) - 1
        ps = max(ps, 0)
        pe = bisect.bisect_right(page_offsets, max(end_char - 1, 0)) - 1
        pe = max(pe, 0)
        return ps + 1, pe + 1

    def _validate_and_fix_chunk_length(self, chunk_text: str) -> str:
        """Validate chunk doesn't exceed character limit and truncate if needed."""
        if len(chunk_text) <= self.max_chars:
            return chunk_text

        print(f"[WARNING] Chunk excede límite de caracteres ({len(chunk_text)} > {self.max_chars}), truncando...")

        # Truncate to max_chars, trying to break at word boundary
        truncated = chunk_text[:self.max_chars]

        # Try to find last word boundary to avoid cutting words
        last_space = truncated.rfind(' ')
        last_newline = truncated.rfind('\n')

        # Use the latest word boundary found
        boundary = max(last_space, last_newline)
        if boundary > self.max_chars * 0.9:  # Only if we don't lose too much content
            truncated = truncated[:boundary]

        return truncated

    @staticmethod
    def _hash(text: str) -> str:
        """Generate hash for chunk ID."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkSemanticOptimizedAdapter(ChunkerPort):
    """Adapter for optimized semantic chunker."""

    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None):
        self._chunker = SemanticChunkerOptimized(tokenizer, cfg)

    def chunk_document(self, pages: List[str], filename: str = "documento.pdf",
                      document_title: str = "Documento") -> Tuple[List[Chunk], str]:
        return self._chunker.chunk_document(pages, filename, document_title)

    def get_config(self) -> ChunkerConfig:
        return self._chunker.cfg