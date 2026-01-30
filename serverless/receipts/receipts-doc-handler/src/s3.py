import logging
import os

logger = logging.getLogger(__name__)


def get_object(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    file_bytes = response["Body"].read()
    content_type = response.get("ContentType", "")
    return file_bytes, content_type


def upload_page(s3_client, bucket, key, body, content_type):
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )
    logger.info({"action": "uploaded_page", "destination": key})


def build_destination_key(prefix, base_name, filename):
    if prefix:
        return f"{prefix}/{base_name}/{filename}"
    return f"{base_name}/{filename}"
