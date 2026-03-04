"""Cohere Embed Multilingual v3 via AWS Bedrock."""

import json
import logging
import time
from typing import List, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger()


class BedrockCohereEmbedder:
    """Embed texts using Cohere Embed Multilingual v3 on AWS Bedrock."""

    def __init__(
        self,
        region: str,
        model_id: str = "cohere.embed-multilingual-v3",
        timeout_secs: int = 120,
    ) -> None:
        self.region = region
        self.model_id = model_id

        cfg = Config(
            read_timeout=timeout_secs,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.region, config=cfg)

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Embed a list of texts, one Bedrock call per text.

        Args:
            texts: List of text strings to embed.

        Returns:
            Tuple of (vectors, total_embed_tokens).
            total_embed_tokens is the sum of billed input tokens across all calls
            (read from the Bedrock HTTP response header).
        """
        if not texts:
            logger.warning("embed_texts called with empty list")
            return [], 0

        logger.info(f"Embedding {len(texts)} chunk(s) using {self.model_id}")
        vectors: List[List[float]] = []
        total_tokens = 0

        for i, text in enumerate(texts):
            vec, token_count = self._embed_one(text)
            vectors.append(vec)
            total_tokens += token_count
            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} chunk(s)")

        logger.info(
            f"Completed embedding {len(texts)} chunk(s), total_tokens={total_tokens}"
        )
        return vectors, total_tokens

    def _embed_one(self, text: str) -> Tuple[List[float], int]:
        """Embed a single text with exponential backoff on throttling.

        Returns:
            Tuple of (embedding_vector, token_count).
            token_count is read from the Bedrock HTTP response header
            (x-amzn-bedrock-input-token-count), with Cohere body meta as fallback.
        """
        payload = json.dumps({"input_type": "search_document", "texts": [text]}).encode()
        backoff = 1.0

        while True:
            try:
                resp = self.client.invoke_model(modelId=self.model_id, body=payload)
                data = json.loads(resp["body"].read())
                embeddings = data.get("embeddings")

                if not embeddings or not isinstance(embeddings, list):
                    raise RuntimeError("Unexpected Bedrock response: missing 'embeddings'")
                if not isinstance(embeddings[0], list):
                    raise RuntimeError("Unexpected Bedrock response: embedding is not a list")

                # Authoritative source: Bedrock HTTP response header
                http_headers = resp.get("ResponseMetadata", {}).get("HTTPHeaders", {})
                raw = http_headers.get("x-amzn-bedrock-input-token-count")
                token_count = int(raw) if raw is not None else 0
                # Fallback: Cohere body meta.billed_units.input_tokens
                if token_count == 0:
                    meta = data.get("meta", {})
                    token_count = meta.get("billed_units", {}).get("input_tokens", 0)

                return embeddings[0], token_count

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in {"ThrottlingException", "TooManyRequestsException"}:
                    logger.warning(f"Bedrock throttled, retrying in {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue
                raise

            except BotoCoreError as e:
                logger.warning(f"BotoCore error, retrying in {backoff}s: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
