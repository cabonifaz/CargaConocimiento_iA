"""Embedder for Cohere Embed Multilingual v3 on AWS Bedrock."""

import json
import time
import logging
from typing import List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger()


class BedrockCohereEmbedder:
    """Embedder for Cohere Embed Multilingual v3 on Bedrock."""

    def __init__(
        self,
        region: str,
        model_id: str = "cohere.embed-multilingual-v3",
        batch_size: int = 1,
        timeout_secs: int = 120,
    ) -> None:
        self.region = region
        self.model_id = model_id
        self.batch_size = batch_size
        self.timeout_secs = timeout_secs

        cfg = Config(
            read_timeout=self.timeout_secs,
            retries={"max_attempts": 3, "mode": "standard"}
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.region, config=cfg)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using Bedrock Cohere Embed Multilingual v3.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            logger.warning("No texts received for embedding")
            return []

        logger.info(f"Embedding {len(texts)} chunks using {self.model_id}")

        vectors: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            for idx, text in enumerate(batch):
                vec = self._embed_one(text)
                vectors.append(vec)

                if (i + idx + 1) % 10 == 0:
                    logger.info(f"Embedded {i + idx + 1}/{len(texts)} chunks")

        logger.info(f"Completed embedding {len(texts)} chunks")
        return vectors

    def _embed_one(self, text: str) -> List[float]:
        """
        Embed a single text using Bedrock Cohere API.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as list of floats
        """
        body = {
            "input_type": "search_document",
            "texts": [text]
        }

        payload = json.dumps(body).encode("utf-8")

        backoff = 1.0
        while True:
            try:
                resp = self.client.invoke_model(modelId=self.model_id, body=payload)
                raw = resp.get("body").read()
                data = json.loads(raw.decode("utf-8"))

                embeddings = data.get("embeddings")
                if not embeddings or not isinstance(embeddings, list):
                    raise RuntimeError("Unexpected response: no 'embeddings' array found")

                emb = embeddings[0]
                if not isinstance(emb, list):
                    raise RuntimeError("Unexpected response: embedding is not a list")

                return emb

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in {"ThrottlingException", "TooManyRequestsException"}:
                    logger.warning(f"Throttled, retrying after {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue
                raise

            except BotoCoreError as e:
                logger.warning(f"BotoCore error, retrying after {backoff}s: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
