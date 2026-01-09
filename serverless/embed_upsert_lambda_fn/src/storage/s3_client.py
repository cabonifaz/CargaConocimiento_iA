"""S3 client for retrieving chunk JSON files."""

import json
import logging
from typing import List, Dict, Any
from urllib.parse import unquote

import boto3

logger = logging.getLogger()


class S3Client:
    """Client for S3 operations."""

    def __init__(self, bucket_name: str):
        """
        Initialize S3 client.

        Args:
            bucket_name: S3 bucket name where chunks are stored
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')

    def get_chunks(self, key: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from S3 JSON file.

        Args:
            key: S3 object key (path to JSON file, can be URL-encoded)

        Returns:
            List of chunk dictionaries

        Raises:
            ValueError: If JSON structure is unexpected or chunks not found
            Exception: If S3 retrieval fails
        """
        try:
            # Decode URL-encoded characters (e.g., %C3%AD -> í)
            decoded_key = unquote(key)
            logger.info(f"Retrieving chunks from s3://{self.bucket_name}/{decoded_key}")

            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=decoded_key)
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)

            # Handle the structure from chunk-json-example.json
            # The JSON is an array with one object containing a "result" field
            if isinstance(data, list) and len(data) > 0 and 'result' in data[0]:
                chunks = data[0]['result']
            elif isinstance(data, dict) and 'result' in data:
                chunks = data['result']
            elif isinstance(data, list):
                chunks = data
            else:
                raise ValueError(f"Unexpected JSON structure in {key}")

            if not chunks:
                raise ValueError(f"No chunks found in S3 at {decoded_key}")

            logger.info(f"Retrieved {len(chunks)} chunks from S3")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks from S3 {self.bucket_name}/{decoded_key}: {e}")
            raise
