"""S3 client for reading .md extraction results and writing .json chunk files."""

import json
import logging
from typing import List

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()


class S3Client:
    """
    S3 operations for the chunking stage.

    results_bucket: bucket where both extraction .md files are read from
                    and chunking .json files are written to.
    """

    def __init__(self, results_bucket: str):
        if not results_bucket:
            raise ValueError("S3_RESULTS_BUCKET is required")
        self.results_bucket = results_bucket
        self._client = None

    @property
    def client(self):
        """Lazy boto3 S3 client (reused across calls within the same invocation)."""
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client

    def download_markdown(self, s3_key: str) -> str:
        """
        Download a .md file from the results bucket.

        Args:
            s3_key: S3 object key (e.g. "ingest-results/extraction/1/5/informe.md")

        Returns:
            Full markdown text as a UTF-8 string.

        Raises:
            ClientError: on S3 API failure.
        """
        try:
            response = self.client.get_object(
                Bucket=self.results_bucket,
                Key=s3_key,
            )
            content = response["Body"].read().decode("utf-8")
            logger.info(
                f"Downloaded markdown ({len(content)} chars) from "
                f"s3://{self.results_bucket}/{s3_key}"
            )
            return content
        except ClientError as e:
            logger.error(f"Failed to download markdown from {s3_key}: {e}")
            raise

    def upload_chunks_json(self, s3_key: str, chunks: List[dict]) -> None:
        """
        Upload a list of chunk dicts as a JSON file to the results bucket.

        Args:
            s3_key:  Destination key (e.g. "ingest-results/chunking/1/5/informe.json")
            chunks:  List of chunk dicts to serialise.
        """
        try:
            body = json.dumps(chunks, ensure_ascii=False).encode("utf-8")
            self.client.put_object(
                Bucket=self.results_bucket,
                Key=s3_key,
                Body=body,
                ContentType="application/json; charset=utf-8",
            )
            logger.info(
                f"Uploaded {len(chunks)} chunks ({len(body)} bytes) to "
                f"s3://{self.results_bucket}/{s3_key}"
            )
        except ClientError as e:
            logger.error(f"Failed to upload chunks JSON to {s3_key}: {e}")
            raise
