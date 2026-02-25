"""Cohere Embed Multilingual v3 via AWS Bedrock."""

import json
import logging
import time
from typing import List

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

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts, one Bedrock call per text.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors aligned with `texts`.
        """
        if not texts:
            logger.warning("embed_texts called with empty list")
            return []

        logger.info(f"Embedding {len(texts)} chunk(s) using {self.model_id}")
        vectors: List[List[float]] = []

        for i, text in enumerate(texts):
            vectors.append(self._embed_one(text))
            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} chunk(s)")

        logger.info(f"Completed embedding {len(texts)} chunk(s)")
        return vectors

    def _embed_one(self, text: str) -> List[float]:
        """Embed a single text with exponential backoff on throttling."""
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

                return embeddings[0]

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
