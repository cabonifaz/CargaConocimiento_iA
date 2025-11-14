import logging
from typing import Dict, Any

from normalizer import MarkdownNormalizer
from chunker import SemanticChunker
from quality import QualityConfig, ChunkQuality

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any) -> list:

    try:
        markdown_text = event.get('markdown_text', '')
        filename = event.get('filename', 'document.pdf')
        target_tokens = event.get('target_tokens', 400)
        min_tokens = event.get('min_tokens', 50)
        enable_quality_filter = event.get('enable_quality_filter', True)

        if not markdown_text:
            logger.warning("No markdown_text provided in event")
            raise ValueError("markdown_text is required")

        logger.info(f"Processing file: {filename}, text length: {len(markdown_text)}")

        normalizer = MarkdownNormalizer()
        normalized_text = normalizer.normalize(markdown_text)
        logger.info(f"Text normalized, length: {len(normalized_text)}")

        chunker = SemanticChunker(target_tokens=target_tokens, min_tokens=min_tokens)
        all_chunks = chunker.chunk(normalized_text, filename)
        logger.info(f"Initial chunks created: {len(all_chunks)}")

        if enable_quality_filter:
            quality_config = QualityConfig(
                min_tokens=event.get('quality_min_tokens', 50),
                min_chars=event.get('quality_min_chars', 200),
                min_alpha_ratio=event.get('quality_min_alpha_ratio', 0.30),
                min_unique_ratio=event.get('quality_min_unique_ratio', 0.10)
            )
            quality_checker = ChunkQuality(quality_config)

            accepted_chunks = []
            for chunk in all_chunks:
                if quality_checker.is_good_quality(chunk):
                    accepted_chunks.append(chunk)

            chunks_to_use = accepted_chunks
            rejected_count = len(all_chunks) - len(accepted_chunks)
            logger.info(f"Quality filter: accepted={len(accepted_chunks)}, rejected={rejected_count}")
        else:
            chunks_to_use = all_chunks
            rejected_count = 0

        for i, chunk in enumerate(chunks_to_use):
            chunk['filename'] = filename
            chunk['chunk_index'] = i

        logger.info(f"Processing completed: {len(chunks_to_use)} chunks ready for embeddings")

        return chunks_to_use

    except Exception as e:
        logger.error(f"Error processing markdown: {str(e)}", exc_info=True)
        raise
