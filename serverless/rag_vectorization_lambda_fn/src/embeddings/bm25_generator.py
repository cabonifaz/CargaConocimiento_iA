"""BM25 keyword text generation using Amazon Bedrock Llama 4 Maverick."""

import json
import logging
import math
from typing import List, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger()


class BedrockBM25Generator:
    """
    Generate BM25-optimised keyword text for each chunk using Llama 4 Maverick.

    BM25 text is used alongside semantic vectors for hybrid search in Weaviate.
    If Bedrock fails or returns an empty result for a chunk, the generator falls
    back to the original chunk text — guaranteeing that bm25_text is never empty.
    """

    # Conservative input token budget per batch (leaves room for prompt + output)
    SAFE_INPUT_TOKENS_PER_BATCH = 100_000
    CHARS_PER_TOKEN_ESTIMATE = 4

    def __init__(
        self,
        region: str,
        model_id: str = "us.meta.llama4-maverick-17b-instruct-v1:0",
        timeout_secs: int = 300,
    ) -> None:
        self.region = region
        self.model_id = model_id

        cfg = Config(
            read_timeout=timeout_secs,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.region, config=cfg)

    def generate_bm25_texts(self, chunks: List[str]) -> Tuple[List[str], int, int]:
        """
        Generate BM25-optimised text for each chunk, batching to stay within
        Llama's context window.

        Guarantee: the returned list has the same length as `chunks` and no
        entry is empty — if a chunk's BM25 generation fails or returns blank,
        the original chunk text is used as fallback.

        Args:
            chunks: List of chunk texts.

        Returns:
            Tuple of (bm25_texts, total_input_tokens, total_output_tokens).
        """
        if not chunks:
            return [], 0, 0

        logger.info("Generating BM25 text for %s chunk(s) using %s", len(chunks), self.model_id)

        avg_chars = sum(len(c) for c in chunks) / len(chunks)
        tokens_per_chunk = avg_chars / self.CHARS_PER_TOKEN_ESTIMATE
        batch_size = max(1, int(self.SAFE_INPUT_TOKENS_PER_BATCH / tokens_per_chunk))

        results: List[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        n_batches = math.ceil(len(chunks) / batch_size)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_results, in_tokens, out_tokens = self._generate_batch(batch)
            results.extend(batch_results)
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            logger.info(
                "BM25 batch %s/%s done (%s chunk(s), in=%s, out=%s)",
                i // batch_size + 1, n_batches, len(batch), in_tokens, out_tokens,
            )

        # ── Guarantee: replace any empty result with the original chunk text ──
        # This covers all failure paths: JSON parse errors, API errors, count
        # mismatches, and genuine empty outputs from the model.
        results = [
            bm25 if (bm25 and bm25.strip()) else original
            for bm25, original in zip(results, chunks)
        ]

        fallback_count = sum(1 for bm25, orig in zip(results, chunks) if bm25 == orig)
        if fallback_count:
            logger.warning(
                "%s/%s chunk(s) fell back to original text because BM25 generation returned empty",
                fallback_count, len(chunks),
            )

        logger.info(
            "BM25 generation complete for %s chunk(s) — "
            "total_input_tokens=%s, total_output_tokens=%s",
            len(chunks), total_input_tokens, total_output_tokens,
        )
        return results, total_input_tokens, total_output_tokens

    def _generate_batch(self, chunks: List[str]) -> Tuple[List[str], int, int]:
        """
        Call Llama 4 Maverick to extract keywords for a batch of chunks.

        Returns a tuple of (keywords_list, input_tokens, output_tokens).
        keywords_list entries may be empty strings — the caller applies the
        non-empty guarantee. Token counts are 0 on error (conservative).
        """
        chunks_text = ""
        for idx, chunk in enumerate(chunks, 1):
            preview = chunk[:2000]
            chunks_text += f"\n\n--- Chunk {idx} ---\n{preview}"

        prompt = (
            "Extract the most important keywords from each chunk below. "
            "For each chunk, return ONLY the keywords separated by spaces, "
            "removing stopwords, articles, and common words. "
            "Focus on domain-specific terms, technical vocabulary, and key concepts.\n\n"
            "Return the results in JSON format as an array of strings, one per chunk.\n"
            f"{chunks_text}\n\n"
            "Respond with JSON only, no explanation:"
        )

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 4096,
                    "temperature": 0.1,
                    "top_p": 0.9,
                }).encode(),
            )
            body = json.loads(response["body"].read())
            generated = body.get("generation", "").strip()
            input_tokens = body.get("prompt_token_count", 0)
            output_tokens = body.get("generation_token_count", 0)

            # Strip markdown code fences if present
            if "```json" in generated:
                generated = generated.split("```json")[1].split("```")[0].strip()
            elif "```" in generated:
                generated = generated.split("```")[1].split("```")[0].strip()

            keywords_list = json.loads(generated)

            if not isinstance(keywords_list, list):
                raise ValueError("Response is not a JSON array")

            # Align length to batch size
            while len(keywords_list) < len(chunks):
                keywords_list.append("")
            return [str(k) for k in keywords_list[:len(chunks)]], input_tokens, output_tokens

        except (ClientError, Exception) as e:
            logger.error("BM25 batch generation failed: %s", e)
            # Return empty strings and zero tokens — caller applies non-empty guarantee;
            # zeros avoid inflating cost on error
            return [""] * len(chunks), 0, 0
