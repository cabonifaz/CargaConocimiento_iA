import re
import json
import hashlib
import logging
import bisect
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Simple regex-based tokenizer."""

    def count_tokens(self, text: str) -> int:
        """Counts tokens approximately."""
        if not text:
            return 0
        tokens = re.findall(r'\b\w+\b', text)
        return len(tokens)

    def tokenize(self, text: str) -> List[Tuple[int, str]]:
        """Tokenizes and returns (offset, token)."""
        tokens = []
        for match in re.finditer(r'\b\w+\b', text):
            tokens.append((match.start(), match.group()))
        return tokens


class SemanticChunker:
    """Semantic chunker with hierarchy and table handling."""

    def __init__(self, target_tokens: int = 400, min_tokens: int = 50, max_chars: int = 2048):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_chars = max_chars
        self.tokenizer = SimpleTokenizer()
        self.page_separator = "\n\n\f\n\n"  # Form feed character, matches CLI behavior

    def chunk_document(self, pages: List[str], filename: str = "document.pdf") -> List[Dict[str, Any]]:
        """Creates semantic chunks from pages (like CLI)."""
        full_text, page_offsets = self._join_pages(pages)
        blocks = self._parse_blocks(full_text)
        chunks = []
        chunks.extend(self._create_table_chunks(blocks, page_offsets))
        chunks.extend(self._create_content_chunks(full_text, blocks, page_offsets))
        return chunks

    def _join_pages(self, pages: List[str]) -> Tuple[str, List[int]]:
        """Join pages and track offsets."""
        offsets = []
        parts = []
        pos = 0

        for i, page in enumerate(pages):
            offsets.append(pos)
            parts.append(page)
            pos += len(page)
            if i != len(pages) - 1:
                parts.append(self.page_separator)
                pos += len(self.page_separator)

        return "".join(parts), offsets

    def _get_pages_for_span(self, page_offsets: List[int], start_char: int, end_char: int) -> Tuple[int, int]:
        """Get page numbers for character span."""
        ps = bisect.bisect_right(page_offsets, start_char) - 1
        ps = max(ps, 0)
        pe = bisect.bisect_right(page_offsets, max(end_char - 1, 0)) - 1
        pe = max(pe, 0)
        return ps + 1, pe + 1

    def _parse_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Parses text into semantic blocks."""
        lines = text.splitlines(keepends=True)
        blocks = []
        offset = 0
        hierarchy = []

        i = 0
        while i < len(lines):
            line = lines[i]
            line_content = line.rstrip('\r\n')

            block_start = offset
            block_lines = [line]
            block_type = self._classify_line(line_content)

            if block_type == 'heading':
                level = self._get_heading_level(line_content)
                heading_text = re.sub(r'^#{1,6}\s+', '', line_content).strip()
                hierarchy = self._update_hierarchy(hierarchy, heading_text, level)

            i += 1
            offset += len(line)

            if block_type in ['table', 'list']:
                while i < len(lines):
                    line = lines[i]
                    line_content = line.rstrip('\r\n')
                    if not line_content.strip() or self._classify_line(line_content) != block_type:
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
                'hierarchy': hierarchy.copy()
            })

        return blocks

    def _classify_line(self, line: str) -> str:
        """Classifies line type."""
        line = line.strip()
        if not line:
            return 'paragraph'
        if re.match(r'^#{1,6}\s+', line):
            return 'heading'
        if line.startswith('|') and line.endswith('|'):
            return 'table'
        if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            return 'list'
        return 'paragraph'

    def _get_heading_level(self, line: str) -> int:
        """Gets heading level."""
        match = re.match(r'^(#{1,6})\s+', line.strip())
        return len(match.group(1)) if match else 0

    def _update_hierarchy(self, hierarchy: List[str], heading: str, level: int) -> List[str]:
        """Updates heading hierarchy."""
        if level <= len(hierarchy):
            hierarchy = hierarchy[:level-1]
        hierarchy.append(heading)
        return hierarchy[:6]

    def _create_table_chunks(self, blocks: List[Dict[str, Any]], page_offsets: List[int]) -> List[Dict[str, Any]]:
        """Creates JSON chunks for tables."""
        chunks = []
        for block in blocks:
            if block['type'] == 'table':
                try:
                    table_data = self._parse_markdown_table(block['text'])
                    json_content = self._table_to_json(table_data)
                    hierarchy_header = self._generate_hierarchy_header(block['hierarchy'])
                    chunk_text = f"{hierarchy_header}\n{json_content}" if hierarchy_header else json_content
                    fragments = self._split_oversized(chunk_text)

                    page_start, page_end = self._get_pages_for_span(page_offsets, block['start'], block['end'])

                    for i, fragment in enumerate(fragments):
                        chunks.append({
                            'text': fragment,
                            'token_count': self.tokenizer.count_tokens(fragment),
                            'chunk_id': self._hash(f"{fragment}_{i}"),
                            'char_start': block['start'],
                            'char_end': block['end'],
                            'page_start': page_start,
                            'page_end': page_end,
                            'type': 'table_json'
                        })
                except Exception as e:
                    logger.error(f"Error processing table: {e}")
        return chunks

    def _parse_markdown_table(self, table_text: str) -> Dict[str, Any]:
        """Parses markdown table to JSON structure."""
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

            cells = self._extract_table_cells(line)
            if not cells:
                continue

            if i == 0:
                headers = cells
            elif separator_found or i > 0:
                rows.append(cells)

        return {"headers": headers, "rows": rows}

    def _extract_table_cells(self, line: str) -> List[str]:
        """Extracts cells from table row."""
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]

        cells = [cell.strip() for cell in line.split('|')]
        while cells and not cells[0]:
            cells.pop(0)
        while cells and not cells[-1]:
            cells.pop()
        return cells

    def _table_to_json(self, table_data: Dict[str, Any]) -> str:
        """Converts table to JSON string."""
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

    def _create_content_chunks(self, text: str, blocks: List[Dict[str, Any]], page_offsets: List[int]) -> List[Dict[str, Any]]:
        """Creates content chunks with 15% overlap."""
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            chunk_start = current_pos
            chunk_end = self._find_boundary(text, blocks, chunk_start)

            chunk_text = text[chunk_start:chunk_end].strip()
            if not chunk_text:
                break

            hierarchy = self._find_hierarchy(blocks, chunk_start, chunk_end)
            hierarchy_header = self._generate_hierarchy_header(hierarchy)
            enhanced_text = f"{hierarchy_header}\n{chunk_text}" if hierarchy_header else chunk_text
            fragments = self._split_oversized(enhanced_text)

            page_start, page_end = self._get_pages_for_span(page_offsets, chunk_start, chunk_end)

            for i, fragment in enumerate(fragments):
                token_count = self.tokenizer.count_tokens(fragment)
                if token_count < self.min_tokens and not fragments[i+1:]:
                    continue

                chunks.append({
                    'text': fragment,
                    'token_count': token_count,
                    'chunk_id': self._hash(f"{fragment}_{i}"),
                    'char_start': chunk_start,
                    'char_end': chunk_end,
                    'page_start': page_start,
                    'page_end': page_end,
                    'type': 'content'
                })

            if chunk_end >= len(text):
                break

            overlap_pos = self._calculate_overlap(text, chunk_start, chunk_end, self.tokenizer.count_tokens(chunk_text))
            current_pos = max(overlap_pos, current_pos + 1)

        return chunks

    def _find_boundary(self, text: str, blocks: List[Dict[str, Any]], start_pos: int) -> int:
        """Finds optimal chunk boundary."""
        estimated_chars = self.target_tokens * 4
        target_end = min(start_pos + estimated_chars, len(text))

        for block in blocks:
            if (block['type'] == 'heading' and
                block['start'] >= start_pos + 50 and
                block['start'] <= target_end + 200):
                boundary_tokens = self.tokenizer.count_tokens(text[start_pos:block['start']])
                if boundary_tokens >= self.target_tokens * 0.7:
                    return block['start']

        return min(target_end, len(text))

    def _find_hierarchy(self, blocks: List[Dict[str, Any]], start_char: int, end_char: int) -> List[str]:
        """Finds hierarchy for a span."""
        best_hierarchy = []
        max_len = 0

        for block in blocks:
            if block['start'] < end_char and block['end'] > start_char:
                hierarchy = block.get('hierarchy', [])
                if len(hierarchy) > max_len:
                    max_len = len(hierarchy)
                    best_hierarchy = hierarchy

        return best_hierarchy

    def _generate_hierarchy_header(self, hierarchy: List[str]) -> str:
        """Generates hierarchy header."""
        if not hierarchy:
            return ""
        headers = []
        for i, section in enumerate(hierarchy):
            level = i + 1
            headers.append(f"{'#' * level} {section}")
        return "\n".join(headers)

    def _calculate_overlap(self, text: str, chunk_start: int, chunk_end: int, token_count: int) -> int:
        """Calculates start position with 15% overlap."""
        overlap_tokens = max(1, int(token_count * 0.15))
        chunk_text = text[chunk_start:chunk_end]
        tokens = self.tokenizer.tokenize(chunk_text)

        if len(tokens) <= overlap_tokens:
            return chunk_start

        overlap_start = tokens[-overlap_tokens][0]
        return chunk_start + overlap_start

    def _split_oversized(self, text: str) -> List[str]:
        """Splits chunks exceeding character limit."""
        if len(text) <= self.max_chars:
            return [text]

        chunks = []
        pos = 0

        while pos < len(text):
            end_pos = min(pos + self.max_chars, len(text))

            if end_pos < len(text):
                min_boundary = pos + int(self.max_chars * 0.8)
                last_space = text.rfind(' ', min_boundary, end_pos)
                last_newline = text.rfind('\n', min_boundary, end_pos)
                boundary = max(last_space, last_newline)
                if boundary > min_boundary:
                    end_pos = boundary

            chunk = text[pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            pos = end_pos
            while pos < len(text) and text[pos].isspace():
                pos += 1

        return chunks

    @staticmethod
    def _hash(text: str) -> str:
        """Generates hash for chunk ID."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
