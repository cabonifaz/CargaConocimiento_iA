"""S3 client for reading source PDFs and writing extraction results."""

import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()


class S3Client:
    """
    S3 operations for the extraction stage.

    documents_bucket: source bucket where uploaded PDFs are stored.
    results_bucket:   destination bucket where .md extraction results are written.
    """

    def __init__(self, documents_bucket: str, results_bucket: str):
        if not documents_bucket:
            raise ValueError("S3_DOCUMENTS_BUCKET is required")
        if not results_bucket:
            raise ValueError("S3_RESULTS_BUCKET is required")

        self.documents_bucket = documents_bucket
        self.results_bucket = results_bucket
        self._client: Optional[object] = None

    @property
    def client(self):
        """Lazy boto3 S3 client (reused across calls within the same invocation)."""
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client

    def get_presigned_url(self, s3_key: str, expiry_secs: int = 3600) -> str:
        """
        Generate a presigned GET URL for a PDF in the documents bucket.

        Mistral OCR downloads the PDF directly from this URL.
        The expiry is set to 1 hour to allow for large file processing.

        Args:
            s3_key:      S3 object key (e.g. "documents/1/5/informe.pdf")
            expiry_secs: URL lifetime in seconds (default: 3600)

        Returns:
            Presigned HTTPS URL string.
        """
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.documents_bucket, "Key": s3_key},
                ExpiresIn=expiry_secs,
            )
            logger.info(f"Generated presigned URL for s3://{self.documents_bucket}/{s3_key}")
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
            raise

    def upload_markdown(self, s3_key: str, content: str) -> None:
        """
        Upload a markdown string to the results bucket.

        Args:
            s3_key:  Destination key (e.g. "ingest-results/extraction/1/5/informe.md")
            content: Full markdown text to upload.
        """
        try:
            self.client.put_object(
                Bucket=self.results_bucket,
                Key=s3_key,
                Body=content.encode("utf-8"),
                ContentType="text/markdown; charset=utf-8",
            )
            logger.info(
                f"Uploaded markdown ({len(content)} chars) to "
                f"s3://{self.results_bucket}/{s3_key}"
            )
        except ClientError as e:
            logger.error(f"Failed to upload markdown to {s3_key}: {e}")
            raise
