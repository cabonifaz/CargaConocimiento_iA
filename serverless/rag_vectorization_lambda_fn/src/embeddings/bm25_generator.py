"""BM25 keyword text generation using Amazon Bedrock Llama 4 Maverick."""

import json
import logging
import math
from typing import List

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

    def generate_bm25_texts(self, chunks: List[str]) -> List[str]:
        """
        Generate BM25-optimised text for each chunk, batching to stay within
        Llama's context window.

        Guarantee: the returned list has the same length as `chunks` and no
        entry is empty — if a chunk's BM25 generation fails or returns blank,
        the original chunk text is used as fallback.

        Args:
            chunks: List of chunk texts.

        Returns:
            List of keyword strings (one per chunk, never empty).
        """
        if not chunks:
            return []

        logger.info(f"Generating BM25 text for {len(chunks)} chunk(s) using {self.model_id}")

        avg_chars = sum(len(c) for c in chunks) / len(chunks)
        tokens_per_chunk = avg_chars / self.CHARS_PER_TOKEN_ESTIMATE
        batch_size = max(1, int(self.SAFE_INPUT_TOKENS_PER_BATCH / tokens_per_chunk))

        results: List[str] = []
        n_batches = math.ceil(len(chunks) / batch_size)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_results = self._generate_batch(batch)
            results.extend(batch_results)
            logger.info(
                f"BM25 batch {i // batch_size + 1}/{n_batches} done "
                f"({len(batch)} chunk(s))"
            )

        # ── Guarantee: replace any empty result with the original chunk text ──
        # This covers all failure paths: JSON parse errors, API errors, count
        # mismatches, and genuine empty outputs from the model.
        results = [
            bm25 if (bm25 and bm25.strip()) else original
            for bm25, original in zip(results, chunks)
        ]

        empty_count = sum(1 for b, r in zip(chunks, results) if r == b)
        if empty_count:
            logger.warning(
                f"{empty_count}/{len(chunks)} chunk(s) fell back to original text "
                "because BM25 generation returned empty"
            )

        logger.info(f"BM25 generation complete for {len(chunks)} chunk(s)")
        return results

    def _generate_batch(self, chunks: List[str]) -> List[str]:
        """
        Call Llama 4 Maverick to extract keywords for a batch of chunks.

        Returns a list of the same length as `chunks`. Any entry may be an
        empty string — the caller applies the non-empty guarantee.
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
            generated = json.loads(response["body"].read()).get("generation", "").strip()

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
            return [str(k) for k in keywords_list[:len(chunks)]]

        except (ClientError, Exception) as e:
            logger.error(f"BM25 batch generation failed: {e}")
            # Return empty strings — the caller's non-empty guarantee will
            # replace them with the original chunk texts
            return [""] * len(chunks)
