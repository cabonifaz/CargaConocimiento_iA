"""S3 client for retrieving chunk JSON files produced by the chunking Lambda."""

import json
import logging
from typing import Any, Dict, List
from urllib.parse import unquote

import boto3

logger = logging.getLogger()


class S3Client:
    """Client for reading chunk JSON files from S3."""

    def __init__(self, results_bucket: str):
        if not results_bucket:
            raise ValueError("results_bucket is required")
        self.results_bucket = results_bucket
        self._client = boto3.client("s3")

    def get_chunks(self, s3_key: str) -> List[Dict[str, Any]]:
        """
        Download and parse a chunks JSON file from S3.

        The chunking Lambda writes a plain JSON array, but legacy structures
        (wrapped in a dict or list-of-dict) are also handled.

        Args:
            s3_key: S3 object key (may be URL-encoded).

        Returns:
            List of chunk dicts.

        Raises:
            ValueError: On unexpected JSON structure or empty result.
        """
        decoded_key = unquote(s3_key)
        logger.info("Retrieving chunks from s3://%s/%s", self.results_bucket, decoded_key)

        try:
            response = self._client.get_object(Bucket=self.results_bucket, Key=decoded_key)
            content = response["Body"].read().decode("utf-8")
        except Exception as e:
            logger.error("Failed to read s3://%s/%s: %s", self.results_bucket, decoded_key, e)
            raise

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {decoded_key}: {e}") from e

        # Chunking Lambda writes a plain array — handle legacy wrapped formats too
        if isinstance(data, list) and data and "result" in data[0]:
            chunks = data[0]["result"]
        elif isinstance(data, dict) and "result" in data:
            chunks = data["result"]
        elif isinstance(data, list):
            chunks = data
        else:
            raise ValueError(f"Unexpected JSON structure in {decoded_key}")

        if not chunks:
            raise ValueError(f"No chunks found in s3://{self.results_bucket}/{decoded_key}")

        logger.info("Retrieved %s chunk(s)", len(chunks))
        return chunks
