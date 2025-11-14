import re
import hashlib
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


class MarkdownNormalizer:
    """Normaliza markdown con manejo especial para tablas con listas."""

    MAX_CHARS_PER_LINE = 2048

    def normalize(self, text: str) -> str:
        """Normaliza el texto markdown completo."""
        text = self._base_sanitizations(text)
        text = self._normalize_tables(text)
        text = self._normalize_paragraphs(text)
        text = self._collapse_blank_lines(text)
        return text.strip()

    def _base_sanitizations(self, text: str) -> str:
        """Limpieza básica de caracteres."""
        text = text.replace("\u00A0", " ")
        for zw in ["\u200B", "\u200C", "\u200D", "\uFEFF"]:
            text = text.replace(zw, "")
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)
        text = text.replace("\uFFFD", "")
        return text

    def _normalize_tables(self, text: str) -> str:
        """Normaliza tablas markdown, manejando listas en celdas."""
        lines = text.splitlines()
        result_lines = []
        in_table = False
        table_lines = []

        for line in lines:
            if re.match(r'^\s*\|.*\|\s*$', line):
                in_table = True
                table_lines.append(line)
            else:
                if in_table and table_lines:
                    normalized_table = self._normalize_table_block(table_lines)
                    result_lines.extend(normalized_table)
                    table_lines = []
                    in_table = False
                result_lines.append(line)

        if table_lines:
            normalized_table = self._normalize_table_block(table_lines)
            result_lines.extend(normalized_table)

        return "\n".join(result_lines)

    def _normalize_table_block(self, table_lines: List[str]) -> List[str]:
        """Normaliza un bloque de tabla completo."""
        normalized = []
        for line in table_lines:
            if re.match(r'^\s*\|[\s\-\|]+\|\s*$', line):
                normalized.append(line)
                continue
            normalized.append(self._normalize_table_row(line))
        return normalized

    def _normalize_table_row(self, row: str) -> str:
        """Normaliza una fila de tabla, reemplazando \n por <br/> en listas."""
        if not row.strip().startswith('|') or not row.strip().endswith('|'):
            return row

        content = row.strip()[1:-1]
        cells = content.split('|')

        normalized_cells = []
        for cell in cells:
            cell = cell.strip()
            if self._has_list_items(cell):
                cell = self._normalize_list_in_cell(cell)
            normalized_cells.append(cell)

        return '| ' + ' | '.join(normalized_cells) + ' |'

    def _has_list_items(self, cell_content: str) -> bool:
        """Detecta si una celda contiene elementos de lista."""
        list_pattern = r'[-*+]\s+[^\n]+\n\s*[-*+]\s+'
        numbered_pattern = r'\d+\.\s+[^\n]+\n\s*\d+\.\s+'
        return bool(re.search(list_pattern, cell_content)) or \
               bool(re.search(numbered_pattern, cell_content))

    def _normalize_list_in_cell(self, cell_content: str) -> str:
        """Reemplaza \n entre elementos de lista por <br/>."""
        content = cell_content.strip()
        content = re.sub(r'([-*+]\s+[^\n]+)\n\s*([-*+]\s+)', r'\1<br/>\2', content)
        content = re.sub(r'(\d+\.\s+[^\n]+)\n\s*(\d+\.\s+)', r'\1<br/>\2', content)
        return content

    def _normalize_paragraphs(self, text: str) -> str:
        """Normaliza párrafos y espacios."""
        text = text.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
        text = re.sub(r"(?<=\w)\n(?=[a-záéíóúñ0-9])", " ", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text

    def _collapse_blank_lines(self, text: str) -> str:
        """Colapsa líneas en blanco excesivas."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^[ \t]+$", "", text, flags=re.MULTILINE)
        return text


@dataclass(frozen=True)
class QualityConfig:
    """Configuración para validación de calidad de chunks."""
    min_tokens: int = 50
    min_chars: int = 200
    min_alpha_ratio: float = 0.30
    min_unique_ratio: float = 0.10


class ChunkQuality:
    """Validador de calidad de chunks."""

    def __init__(self, cfg: Optional[QualityConfig] = None):
        self.cfg = cfg or QualityConfig()

    def is_good_quality(self, chunk: Dict[str, Any]) -> bool:
        """Verifica si un chunk cumple con los estándares de calidad."""
        text = chunk.get('text', '')
        token_count = chunk.get('token_count', 0)

        if token_count < self.cfg.min_tokens:
            return False
        if len(text) < self.cfg.min_chars:
            return False

        letters = sum(ch.isalpha() for ch in text)
        if (letters / max(len(text), 1)) < self.cfg.min_alpha_ratio:
            return False

        words = [w for w in text.split() if w.isalpha()]
        if not words:
            return False

        unique_words = len(set(w.lower() for w in words))
        if (unique_words / len(words)) < self.cfg.min_unique_ratio:
            return False

        return True


class SimpleTokenizer:
    """Tokenizador simple basado en regex."""

    def count_tokens(self, text: str) -> int:
        """Cuenta tokens aproximadamente."""
        if not text:
            return 0
        tokens = re.findall(r'\b\w+\b', text)
        return len(tokens)

    def tokenize(self, text: str) -> List[Tuple[int, str]]:
        """Tokeniza y retorna (offset, token)."""
        tokens = []
        for match in re.finditer(r'\b\w+\b', text):
            tokens.append((match.start(), match.group()))
        return tokens


class SemanticChunker:
    """Chunker semántico con manejo de jerarquías y tablas."""

    def __init__(self, target_tokens: int = 400, min_tokens: int = 50, max_chars: int = 2048):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_chars = max_chars
        self.tokenizer = SimpleTokenizer()

    def chunk(self, text: str, filename: str = "document.pdf") -> List[Dict[str, Any]]:
        """Crea chunks semánticos del texto."""
        blocks = self._parse_blocks(text)
        chunks = []
        chunks.extend(self._create_table_chunks(blocks))
        chunks.extend(self._create_content_chunks(text, blocks))
        return chunks

    def _parse_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Parsea texto en bloques semánticos."""
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
        """Clasifica tipo de línea."""
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
        """Obtiene nivel de encabezado."""
        match = re.match(r'^(#{1,6})\s+', line.strip())
        return len(match.group(1)) if match else 0

    def _update_hierarchy(self, hierarchy: List[str], heading: str, level: int) -> List[str]:
        """Actualiza jerarquía de encabezados."""
        if level <= len(hierarchy):
            hierarchy = hierarchy[:level-1]
        hierarchy.append(heading)
        return hierarchy[:6]

    def _create_table_chunks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crea chunks JSON para tablas."""
        chunks = []
        for block in blocks:
            if block['type'] == 'table':
                try:
                    table_data = self._parse_markdown_table(block['text'])
                    json_content = self._table_to_json(table_data)
                    hierarchy_header = self._generate_hierarchy_header(block['hierarchy'])
                    chunk_text = f"{hierarchy_header}\n{json_content}" if hierarchy_header else json_content
                    fragments = self._split_oversized(chunk_text)

                    for i, fragment in enumerate(fragments):
                        chunks.append({
                            'text': fragment,
                            'token_count': self.tokenizer.count_tokens(fragment),
                            'chunk_id': self._hash(f"{fragment}_{i}"),
                            'char_start': block['start'],
                            'char_end': block['end'],
                            'type': 'table_json'
                        })
                except Exception as e:
                    print(f"Error processing table: {e}")
        return chunks

    def _parse_markdown_table(self, table_text: str) -> Dict[str, Any]:
        """Parsea tabla markdown a estructura JSON."""
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
        """Extrae celdas de una fila de tabla."""
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
        """Convierte tabla a JSON string."""
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

    def _create_content_chunks(self, text: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crea chunks de contenido con 15% overlap."""
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
                    'type': 'content'
                })

            if chunk_end >= len(text):
                break

            overlap_pos = self._calculate_overlap(text, chunk_start, chunk_end, self.tokenizer.count_tokens(chunk_text))
            current_pos = max(overlap_pos, current_pos + 1)

        return chunks

    def _find_boundary(self, text: str, blocks: List[Dict[str, Any]], start_pos: int) -> int:
        """Encuentra límite óptimo del chunk."""
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
        """Encuentra jerarquía para un span."""
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
        """Genera header de jerarquía."""
        if not hierarchy:
            return ""
        headers = []
        for i, section in enumerate(hierarchy):
            level = i + 1
            headers.append(f"{'#' * level} {section}")
        return "\n".join(headers)

    def _calculate_overlap(self, text: str, chunk_start: int, chunk_end: int, token_count: int) -> int:
        """Calcula posición de inicio con 15% overlap."""
        overlap_tokens = max(1, int(token_count * 0.15))
        chunk_text = text[chunk_start:chunk_end]
        tokens = self.tokenizer.tokenize(chunk_text)

        if len(tokens) <= overlap_tokens:
            return chunk_start

        overlap_start = tokens[-overlap_tokens][0]
        return chunk_start + overlap_start

    def _split_oversized(self, text: str) -> List[str]:
        """Divide chunks que exceden el límite de caracteres."""
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
        """Genera hash para chunk ID."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_markdown(
    markdown_text: str,
    filename: str = "document.pdf",
    target_tokens: int = 400,
    min_tokens: int = 50,
    enable_quality_filter: bool = True,
    quality_min_tokens: int = 50,
    quality_min_chars: int = 200,
    quality_min_alpha_ratio: float = 0.30,
    quality_min_unique_ratio: float = 0.10
) -> Dict[str, Any]:
    """Procesa texto markdown: normaliza, crea chunks y filtra por calidad."""
    try:
        normalizer = MarkdownNormalizer()
        normalized_text = normalizer.normalize(markdown_text)

        chunker = SemanticChunker(target_tokens=target_tokens, min_tokens=min_tokens)
        all_chunks = chunker.chunk(normalized_text, filename)

        if enable_quality_filter:
            quality_config = QualityConfig(
                min_tokens=quality_min_tokens,
                min_chars=quality_min_chars,
                min_alpha_ratio=quality_min_alpha_ratio,
                min_unique_ratio=quality_min_unique_ratio
            )
            quality_checker = ChunkQuality(quality_config)

            accepted_chunks = []
            rejected_count = 0

            for chunk in all_chunks:
                if quality_checker.is_good_quality(chunk):
                    accepted_chunks.append(chunk)
                else:
                    rejected_count += 1

            chunks_to_use = accepted_chunks
        else:
            chunks_to_use = all_chunks
            rejected_count = 0

        for i, chunk in enumerate(chunks_to_use):
            chunk['filename'] = filename
            chunk['chunk_index'] = i

        return {
            'success': True,
            'filename': filename,
            'total_chunks_created': len(all_chunks),
            'total_chunks_accepted': len(chunks_to_use),
            'total_chunks_rejected': rejected_count,
            'chunks': chunks_to_use
        }

    except Exception as e:
        return {
            'success': False,
            'filename': filename,
            'error': str(e),
            'total_chunks_created': 0,
            'total_chunks_accepted': 0,
            'total_chunks_rejected': 0,
            'chunks': []
        }


for item in $input.all():
    markdown_text = item.json.get('markdown_text', '')
    filename = item.json.get('filename', 'document.pdf')

    if not markdown_text:
        return [{'error': 'No markdown_text provided', 'success': False}]

    result = process_markdown(markdown_text=markdown_text, filename=filename)

    if result['success']:
        return result['chunks']
    else:
        return [result]
