import os
import logging

from botocore.exceptions import ClientError

from ..domain.ports.file_storage import FileStorage

logger = logging.getLogger(__name__)

PDF_EXT = ".pdf"
TMP_PATH = "/tmp"


class S3FileStorageClient(FileStorage):
    def __init__(self, s3_client: object):
        self._s3_client = s3_client

    def is_pdf_by_head(self, bucket: str, key: str) -> bool:
        try:
            head = self._s3_client.head_object(Bucket=bucket, Key=key)
            content_type = (head.get("ContentType") or "").lower()
            if "pdf" in content_type:
                return True
        except ClientError as e:
            logger.warning(f"head_object failed for {key}: {e}")
        return key.lower().endswith(PDF_EXT)

    def get_object_size(self, bucket: str, key: str) -> int:
        response = self._s3_client.get_object_attributes(
            Bucket=bucket,
            Key=key,
            ObjectAttributes=['ObjectSize']
        )
        return int(response['ObjectSize'])

    def delete_object(self, bucket: str, key: str) -> None:
        self._s3_client.delete_object(Bucket=bucket, Key=key)

    def download_to_tmp(self, bucket: str, key: str) -> str:
        local_path = os.path.join(TMP_PATH, key.split("/")[-1])
        self._s3_client.download_file(bucket, key, local_path)
        return local_path