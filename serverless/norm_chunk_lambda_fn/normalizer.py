import re
from typing import List


class MarkdownNormalizer:
    """Normalizes markdown with special handling for tables with lists."""

    MAX_CHARS_PER_LINE = 2048

    def normalize(self, text: str) -> str:
        """Normalizes complete markdown text."""
        text = self._base_sanitizations(text)
        text = self._normalize_tables(text)
        text = self._normalize_paragraphs(text)
        text = self._collapse_blank_lines(text)
        return text.strip()

    def _base_sanitizations(self, text: str) -> str:
        """Basic character cleanup."""
        text = text.replace("\u00A0", " ")
        for zw in ["\u200B", "\u200C", "\u200D", "\uFEFF"]:
            text = text.replace(zw, "")
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)
        text = text.replace("\uFFFD", "")
        return text

    def _normalize_tables(self, text: str) -> str:
        """Normalizes markdown tables, handling lists in cells."""
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
        """Normalizes complete table block."""
        normalized = []
        for line in table_lines:
            if re.match(r'^\s*\|[\s\-\|]+\|\s*$', line):
                normalized.append(line)
                continue
            normalized.append(self._normalize_table_row(line))
        return normalized

    def _normalize_table_row(self, row: str) -> str:
        """Normalizes table row, replacing newlines with <br/> in lists."""
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
        """Detects if cell contains list items."""
        list_pattern = r'[-*+]\s+[^\n]+\n\s*[-*+]\s+'
        numbered_pattern = r'\d+\.\s+[^\n]+\n\s*\d+\.\s+'
        return bool(re.search(list_pattern, cell_content)) or \
               bool(re.search(numbered_pattern, cell_content))

    def _normalize_list_in_cell(self, cell_content: str) -> str:
        """Replaces newlines between list items with <br/>."""
        content = cell_content.strip()
        content = re.sub(r'([-*+]\s+[^\n]+)\n\s*([-*+]\s+)', r'\1<br/>\2', content)
        content = re.sub(r'(\d+\.\s+[^\n]+)\n\s*(\d+\.\s+)', r'\1<br/>\2', content)
        return content

    def _normalize_paragraphs(self, text: str) -> str:
        """Normalizes paragraphs and spaces."""
        text = text.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
        text = re.sub(r"(?<=\w)\n(?=[a-záéíóúñ0-9])", " ", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text

    def _collapse_blank_lines(self, text: str) -> str:
        """Collapses excessive blank lines."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^[ \t]+$", "", text, flags=re.MULTILINE)
        return text
