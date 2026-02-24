"""Markdown chunking logic — refactored from norm_chunk_lambda_fn/lambda_function.py."""

import bisect
import hashlib
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import MarkdownTextSplitter

logger = logging.getLogger()

# Form-feed separator used internally to track page boundaries in the joined text.
# The caller splits the .md file on the extraction PAGE_SEPARATOR ("\n\n---\n\n")
# to recover individual pages, which are then re-joined here with this separator.
_PAGE_JOIN_SEPARATOR = "\n\n\f\n\n"


class _SimpleTokenizer:
    """Regex-based approximate token counter."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r'\b\w+\b', text))


def _join_pages(pages: List[str]) -> Tuple[str, List[int]]:
    """
    Join page strings with a form-feed separator and record each page's
    character offset in the resulting full text.

    Returns:
        (full_text, page_offsets) where page_offsets[i] is the start position
        of page i in full_text.
    """
    offsets = []
    parts = []
    pos = 0

    for i, page in enumerate(pages):
        offsets.append(pos)
        parts.append(page)
        pos += len(page)
        if i != len(pages) - 1:
            parts.append(_PAGE_JOIN_SEPARATOR)
            pos += len(_PAGE_JOIN_SEPARATOR)

    return "".join(parts), offsets


def _get_pages_for_span(
    page_offsets: List[int], start_char: int, end_char: int
) -> Tuple[int, int]:
    """
    Return the 1-indexed (page_start, page_end) that a character span covers.
    """
    ps = max(bisect.bisect_right(page_offsets, start_char) - 1, 0)
    pe = max(bisect.bisect_right(page_offsets, max(end_char - 1, 0)) - 1, 0)
    return ps + 1, pe + 1


def _extract_headings(text: str) -> List[Dict[str, Any]]:
    """
    Extract markdown headings with their character positions and hierarchy level.

    Returns list of dicts: {level, text, start_pos, end_pos}.
    """
    headings = []
    current_pos = 0

    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            level = 0
            for char in stripped:
                if char == '#':
                    level += 1
                else:
                    break
            headings.append({
                'level': level,
                'text': stripped.lstrip('#').strip(),
                'start_pos': current_pos,
                'end_pos': current_pos + len(line),
            })
        current_pos += len(line) + 1  # +1 for the newline

    return headings


def _get_section_info(
    headings: List[Dict[str, Any]], chunk_start: int, chunk_end: int
) -> Tuple[str, List[str]]:
    """
    Determine the section title and hierarchical path for a chunk.

    Returns: (section_title, section_path)
    """
    relevant = [h for h in headings if h['start_pos'] <= chunk_end]
    if not relevant:
        return "", []

    section_path: List[str] = []

    for heading in relevant:
        level = heading['level']
        text = heading['text']

        if heading['start_pos'] >= chunk_start:
            section_path = section_path[:level - 1]
            section_path.append(text)
        else:
            if level <= len(section_path):
                section_path = section_path[:level - 1]
            section_path.append(text)

    section_title = section_path[-1] if section_path else ""
    return section_title, section_path


def chunk_markdown(
    pages: List[str],
    filename: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 300,
) -> List[Dict[str, Any]]:
    """
    Split a list of markdown page strings into overlapping chunks with metadata.

    Pages are joined internally with a form-feed separator so that page
    boundaries can be tracked precisely via character offsets.

    Args:
        pages:         List of per-page markdown strings (one entry per page).
        filename:      Original document filename — stored in each chunk dict.
        chunk_size:    Target chunk size in characters (~500 tokens for Cohere).
        chunk_overlap: Overlap between consecutive chunks in characters (~15%).

    Returns:
        List of chunk dicts with fields:
            text, token_count, chunk_id, char_start, char_end,
            page_start, page_end, type, filename, chunk_index,
            section_title, section_path
    """
    if not pages:
        raise ValueError("pages list is empty — nothing to chunk")

    logger.info(
        f"Chunking '{filename}' — {len(pages)} page(s), "
        f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )

    full_text, page_offsets = _join_pages(pages)
    logger.info(f"Pages joined — total_length={len(full_text)} chars")

    headings = _extract_headings(full_text)
    logger.info(f"Extracted {len(headings)} heading(s)")

    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    raw_chunks = splitter.split_text(full_text)
    logger.info(f"MarkdownTextSplitter produced {len(raw_chunks)} raw chunk(s)")

    tokenizer = _SimpleTokenizer()
    all_chunks: List[Dict[str, Any]] = []
    skipped = 0
    current_pos = 0

    for idx, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            skipped += 1
            continue

        chunk_start = full_text.find(chunk_text, current_pos)
        if chunk_start == -1:
            logger.warning(f"Chunk {idx}: exact position not found, using approximation")
            chunk_start = current_pos
        chunk_end = chunk_start + len(chunk_text)

        page_start, page_end = _get_pages_for_span(page_offsets, chunk_start, chunk_end)
        section_title, section_path = _get_section_info(headings, chunk_start, chunk_end)
        token_count = tokenizer.count_tokens(chunk_text)
        chunk_id = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        all_chunks.append({
            "text": chunk_text,
            "token_count": token_count,
            "chunk_id": chunk_id,
            "char_start": chunk_start,
            "char_end": chunk_end,
            "page_start": page_start,
            "page_end": page_end,
            "type": "content",
            "filename": filename,
            "chunk_index": len(all_chunks),
            "section_title": section_title,
            "section_path": section_path,
        })

        current_pos = chunk_end

    if skipped:
        logger.info(f"Skipped {skipped} empty chunk(s)")

    if all_chunks:
        token_counts = [c["token_count"] for c in all_chunks]
        logger.info(
            f"Chunking complete — chunks={len(all_chunks)}, "
            f"avg_tokens={sum(token_counts)/len(token_counts):.1f}, "
            f"min={min(token_counts)}, max={max(token_counts)}, "
            f"multi_page={sum(1 for c in all_chunks if c['page_start'] != c['page_end'])}"
        )

    return all_chunks
