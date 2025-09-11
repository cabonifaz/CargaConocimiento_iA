from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re
import bisect
import hashlib

from app.ports.outbound.chunker import ChunkerPort, Chunk, ChunkerConfig
from app.ports.outbound.tokenizer import TokenCounterPort

class SemanticChunker:
    """Semantic chunking with 25% overlap that respects Markdown structure boundaries."""
    
    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None) -> None:
        self.tok = tokenizer
        self.cfg = cfg or ChunkerConfig()
        assert self.cfg.target_tokens > 0, "target_tokens must be > 0"
    
    def chunk_document(self, pages: List[str]) -> Tuple[List[Chunk], str]:
        """Perform semantic chunking with 25% overlap on joined document."""
        print("🧩 [CHUNKING] Método: Semántico con 25% overlap")
        full_text, page_offsets = self._join_pages_and_offsets(pages)
        # Parse Markdown blocks for semantic boundaries  
        md_blocks = self._parse_markdown_blocks(full_text)
        
        chunks = self._semantic_chunk_with_overlap(full_text, page_offsets, md_blocks)
        print(f"🧩 [CHUNKING] → {len(chunks)} chunks semánticos generados")
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
    
    def _parse_markdown_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Parse Markdown blocks for semantic boundaries."""
        lines = text.splitlines(keepends=True)
        blocks: List[Dict[str, Any]] = []
        
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
            
            # For structured blocks, consume until boundary
            if block_type == 'heading':
                # Headings are single-line
                pass
            elif block_type == 'code':
                # Code fence - consume until closing fence
                if line_content.strip().startswith('```'):
                    i += 1
                    offset += len(line)
                    while i < len(lines):
                        line = lines[i]
                        block_lines.append(line)
                        offset += len(line)
                        if line.rstrip('\r\n').strip().startswith('```'):
                            break
                        i += 1
            elif block_type in ['table', 'list', 'quote']:
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
                'level': self._get_heading_level(line_content) if block_type == 'heading' else 0
            })
            
            offset += len(line)
            i += 1
        
        return blocks
    
    def _classify_line(self, line: str) -> str:
        """Classify a line into a Markdown block type."""
        line = line.strip()
        
        if not line:
            return 'paragraph'
        
        # Heading
        if re.match(r'^#{1,6}\s+', line):
            return 'heading'
        
        # Code fence
        if line.startswith('```'):
            return 'code'
        
        # Table row
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
                    chunks[-1] = Chunk(
                        text=merged_text.strip(),
                        token_count=merged_tokens,
                        chunk_id=self._hash(merged_text),
                        char_start=last_chunk.char_start,
                        char_end=chunk_end,
                        page_start=page_start,
                        page_end=page_end,
                    )
                else:
                    # Create separate chunk even if small
                    page_start, page_end = self._pages_for_span(page_offsets, chunk_start, chunk_end)
                    chunks.append(self._create_chunk(chunk_text, chunk_start, chunk_end, page_start, page_end))
            else:
                # Create normal chunk
                page_start, page_end = self._pages_for_span(page_offsets, chunk_start, chunk_end)
                chunks.append(self._create_chunk(chunk_text, chunk_start, chunk_end, page_start, page_end))
            
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
                if (block['type'] in ['table', 'list', 'code', 'quote'] and 
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
    
    def _create_chunk(self, text: str, char_start: int, char_end: int, page_start: int, page_end: int) -> Chunk:
        """Create a Chunk object."""
        return Chunk(
            text=text,
            token_count=self.tok.count_tokens(text),
            chunk_id=self._hash(text),
            char_start=char_start,
            char_end=char_end,
            page_start=page_start,
            page_end=page_end,
        )
    
    @staticmethod
    def _hash(text: str) -> str:
        """Generate hash for chunk ID."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkSemanticAdapter(ChunkerPort):
    """Adapter for semantic chunking with 25% overlap."""
    
    def __init__(self, tokenizer: TokenCounterPort, cfg: ChunkerConfig | None = None) -> None:
        """Initialize semantic chunker with tokenizer and configuration."""
        self._chunker = SemanticChunker(tokenizer, cfg)
    
    def chunk_document(self, pages: List[str]) -> Tuple[List[Chunk], str]:
        """Chunk document using semantic boundaries with 25% overlap."""
        return self._chunker.chunk_document(pages)
    
    def get_config(self) -> ChunkerConfig:
        """Get the current chunker configuration."""
        return self._chunker.cfg