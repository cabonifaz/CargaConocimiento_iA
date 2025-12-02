import logging
import hashlib
import re
import bisect
from typing import Dict, Any, List, Tuple

from langchain_text_splitters import MarkdownTextSplitter

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SimpleTokenizer:
    """Simple regex-based tokenizer for token counting."""

    def count_tokens(self, text: str) -> int:
        """Counts tokens approximately."""
        if not text:
            return 0
        tokens = re.findall(r'\b\w+\b', text)
        return len(tokens)


def join_pages(pages: List[str]) -> Tuple[str, List[int]]:
    """Join pages with separator and track offsets for page mapping."""
    page_separator = "\n\n\f\n\n"  # Form feed character separator
    offsets = []
    parts = []
    pos = 0

    for i, page in enumerate(pages):
        offsets.append(pos)
        parts.append(page)
        pos += len(page)
        if i != len(pages) - 1:
            parts.append(page_separator)
            pos += len(page_separator)

    return "".join(parts), offsets


def get_pages_for_span(page_offsets: List[int], start_char: int, end_char: int) -> Tuple[int, int]:
    """Calculate which pages a character span covers."""
    ps = bisect.bisect_right(page_offsets, start_char) - 1
    ps = max(ps, 0)
    pe = bisect.bisect_right(page_offsets, max(end_char - 1, 0)) - 1
    pe = max(pe, 0)
    return ps + 1, pe + 1  # Convert to 1-indexed


def extract_headings_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract all markdown headings with their positions and hierarchy.

    Returns list of dicts with: {level, text, start_pos, end_pos}
    """
    headings = []
    lines = text.split('\n')
    current_pos = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # Count heading level
            level = 0
            for char in stripped:
                if char == '#':
                    level += 1
                else:
                    break

            # Extract heading text
            heading_text = stripped.lstrip('#').strip()

            headings.append({
                'level': level,
                'text': heading_text,
                'start_pos': current_pos,
                'end_pos': current_pos + len(line)
            })

        current_pos += len(line) + 1  # +1 for newline

    return headings


def get_section_info_for_chunk(headings: List[Dict[str, Any]], chunk_start: int, chunk_end: int) -> Tuple[str, List[str]]:
    """
    Determine section title and path for a chunk based on headings.

    Returns: (section_title, section_path)
    """
    # Find all headings that appear before or at the chunk start
    relevant_headings = [h for h in headings if h['start_pos'] <= chunk_end]

    if not relevant_headings:
        return "", []

    # Build section hierarchy
    section_path = []
    current_level = 0

    for heading in relevant_headings:
        level = heading['level']
        text = heading['text']

        # If heading is within the chunk, include it
        if heading['start_pos'] >= chunk_start:
            # Reset path to this level
            section_path = section_path[:level-1]
            section_path.append(text)
            current_level = level
        # If heading is before chunk, update hierarchy
        elif heading['start_pos'] < chunk_start:
            # Adjust hierarchy based on level
            if level <= len(section_path):
                section_path = section_path[:level-1]
            section_path.append(text)
            current_level = level

    section_title = section_path[-1] if section_path else ""

    return section_title, section_path


def lambda_handler(event: Dict[str, Any], context: Any = None) -> List[Dict[str, Any]]:  # noqa: ARG001

    try:
        pages = event.get('pages', [])
        filename = event.get('filename', 'document.pdf')
        chunk_size = event.get('chunk_size', 2000)  # ~500 tokens, optimal for RAG (safe for Cohere's 512 token limit)
        chunk_overlap = event.get('chunk_overlap', 300)  # 15% overlap (NVIDIA research optimal)

        if not pages:
            logger.warning("No pages provided in event")
            raise ValueError("pages array is required")

        logger.info(f"Processing started - filename={filename}, pages={len(pages)}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

        # Join all pages and track offsets for page mapping
        full_text, page_offsets = join_pages(pages)
        logger.info(f"Pages joined - total_length={len(full_text)} chars, page_boundaries={len(page_offsets)}")

        # Extract all headings from the full document for section metadata
        headings = extract_headings_from_text(full_text)
        logger.info(f"Extracted {len(headings)} headings from document")

        # Initialize LangChain MarkdownTextSplitter
        splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Split the entire document
        chunks_text = splitter.split_text(full_text)
        logger.info(f"Document split - initial_chunks={len(chunks_text)}")

        tokenizer = SimpleTokenizer()
        all_chunks = []
        skipped_empty = 0

        # Track current position in the full text to calculate char positions
        current_pos = 0

        for idx, chunk_text in enumerate(chunks_text):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                skipped_empty += 1
                continue

            # Find the chunk in the full text to get its position
            chunk_start = full_text.find(chunk_text, current_pos)
            if chunk_start == -1:
                # If exact match not found, approximate position
                logger.warning(f"Chunk {idx}: exact position not found, using approximation")
                chunk_start = current_pos
            chunk_end = chunk_start + len(chunk_text)

            # Calculate which pages this chunk spans
            page_start, page_end = get_pages_for_span(page_offsets, chunk_start, chunk_end)

            # Extract section metadata based on document structure
            section_title, section_path = get_section_info_for_chunk(headings, chunk_start, chunk_end)

            token_count = tokenizer.count_tokens(chunk_text)
            chunk_id = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

            chunk = {
                'text': chunk_text,
                'token_count': token_count,
                'chunk_id': chunk_id,
                'char_start': chunk_start,
                'char_end': chunk_end,
                'page_start': page_start,
                'page_end': page_end,
                'type': 'content',
                'filename': filename,
                'chunk_index': len(all_chunks),
                'section_title': section_title,
                'section_path': section_path
            }

            all_chunks.append(chunk)
            current_pos = chunk_end

        if skipped_empty > 0:
            logger.info(f"Skipped {skipped_empty} empty chunks")

        # Calculate statistics
        token_counts = [c['token_count'] for c in all_chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        multi_page_chunks = sum(1 for c in all_chunks if c['page_start'] != c['page_end'])

        logger.info(f"Processing completed - total_chunks={len(all_chunks)}, "
                   f"avg_tokens={avg_tokens:.1f}, min_tokens={min_tokens}, max_tokens={max_tokens}, "
                   f"multi_page_chunks={multi_page_chunks}")

        return all_chunks

    except Exception as e:
        logger.error(f"Error processing markdown: {str(e)}", exc_info=True)
        raise
