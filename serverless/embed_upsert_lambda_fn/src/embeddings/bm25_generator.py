"""BM25 text generation using Amazon Bedrock Llama 4 Maverick model."""

import json
import logging
from typing import List, Tuple
import math

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger()


class BedrockBM25Generator:
    """Generate BM25-optimized text using Bedrock Llama 4 Maverick model."""

    # Llama 4 Maverick supports 1M tokens context window
    MAX_CONTEXT_TOKENS = 1_000_000
    # Conservative estimate for safe batching (reserve space for prompt + output)
    SAFE_INPUT_TOKENS_PER_BATCH = 100_000
    # Estimate ~4 chars per token for input text
    CHARS_PER_TOKEN_ESTIMATE = 4

    def __init__(
        self,
        region: str,
        model_id: str = "us.meta.llama4-maverick-17b-instruct-v1:0",
        timeout_secs: int = 300,
    ) -> None:
        """
        Initialize Bedrock BM25 generator.

        Args:
            region: AWS region for Bedrock
            model_id: Bedrock model ID
            timeout_secs: Request timeout in seconds
        """
        self.region = region
        self.model_id = model_id
        self.timeout_secs = timeout_secs

        cfg = Config(
            read_timeout=self.timeout_secs,
            retries={"max_attempts": 3, "mode": "standard"}
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.region, config=cfg)

    def generate_bm25_texts(self, chunks: List[str]) -> List[str]:
        """
        Generate BM25-optimized text for multiple chunks with batching.

        For large documents (100-200 pages), chunks are processed in batches
        to respect model token limits.

        Args:
            chunks: List of chunk texts

        Returns:
            List of BM25-optimized texts (one per chunk)
        """
        if not chunks:
            return []

        logger.info(f"Generating BM25 text for {len(chunks)} chunks using {self.model_id}")

        # Calculate batch size based on average chunk size
        avg_chunk_size = sum(len(c) for c in chunks) / len(chunks)
        estimated_tokens_per_chunk = avg_chunk_size / self.CHARS_PER_TOKEN_ESTIMATE

        # Calculate how many chunks can fit in one batch
        chunks_per_batch = max(1, int(self.SAFE_INPUT_TOKENS_PER_BATCH / estimated_tokens_per_chunk))

        logger.info(f"Processing in batches of ~{chunks_per_batch} chunks (avg chunk size: {avg_chunk_size:.0f} chars)")

        results = []
        for i in range(0, len(chunks), chunks_per_batch):
            batch = chunks[i:i + chunks_per_batch]
            batch_results = self._generate_batch(batch)
            results.extend(batch_results)

            logger.info(f"Processed batch {i//chunks_per_batch + 1}/{math.ceil(len(chunks)/chunks_per_batch)}: {len(batch)} chunks")

        logger.info(f"Completed BM25 text generation for {len(chunks)} chunks")
        return results

    def _generate_batch(self, chunks: List[str]) -> List[str]:
        """
        Generate BM25 text for a batch of chunks.

        Args:
            chunks: List of chunk texts to process

        Returns:
            List of BM25-optimized texts
        """
        # Create prompt for keyword extraction
        chunks_text = ""
        for idx, chunk in enumerate(chunks, 1):
            # Truncate very long chunks to prevent issues
            chunk_preview = chunk[:2000] if len(chunk) > 2000 else chunk
            chunks_text += f"\n\n--- Chunk {idx} ---\n{chunk_preview}"

        prompt = f"""Extract the most important keywords from each chunk below. For each chunk, return ONLY the keywords separated by spaces, removing stopwords, articles, and common words. Focus on domain-specific terms, technical vocabulary, and key concepts.

Return the results in JSON format as an array of strings, one per chunk.

{chunks_text}

Respond with JSON only, no explanation:"""

        try:
            # Call Bedrock with Llama 4 Maverick
            request_body = {
                "prompt": prompt,
                "max_gen_len": 4096,
                "temperature": 0.1,  # Low temperature for consistent extraction
                "top_p": 0.9,
            }

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body).encode("utf-8")
            )

            response_body = json.loads(response["body"].read().decode("utf-8"))
            generated_text = response_body.get("generation", "").strip()

            # Try to parse JSON response
            try:
                # Remove markdown code blocks if present
                if "```json" in generated_text:
                    generated_text = generated_text.split("```json")[1].split("```")[0].strip()
                elif "```" in generated_text:
                    generated_text = generated_text.split("```")[1].split("```")[0].strip()

                keywords_list = json.loads(generated_text)

                if not isinstance(keywords_list, list):
                    raise ValueError("Response is not a list")

                # Ensure we have the right number of results
                if len(keywords_list) != len(chunks):
                    logger.warning(f"Expected {len(chunks)} results, got {len(keywords_list)}")
                    # Pad or truncate as needed
                    while len(keywords_list) < len(chunks):
                        keywords_list.append("")
                    keywords_list = keywords_list[:len(chunks)]

                return keywords_list

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}. Using fallback extraction.")
                # Fallback: try to extract keywords line by line
                lines = generated_text.strip().split("\n")
                results = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(("#", "-", "Chunk")):
                        results.append(line)

                # Ensure correct number of results
                while len(results) < len(chunks):
                    results.append("")
                return results[:len(chunks)]

        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            # Fallback to empty strings
            return [""] * len(chunks)

        except Exception as e:
            logger.error(f"Unexpected error generating BM25 text: {e}")
            # Fallback to empty strings
            return [""] * len(chunks)
