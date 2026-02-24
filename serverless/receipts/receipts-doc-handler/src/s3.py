import logging

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def get_object(s3_client, bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_bytes = response["Body"].read()
        content_type = response.get("ContentType", "")
        logger.info({"action": "get_object_success", "bucket": bucket, "key": key})
        return file_bytes, content_type
    except ClientError as e:
        logger.error({"action": "get_object_failed", "bucket": bucket, "key": key, "error": str(e)})
        raise


def upload_page(s3_client, bucket, key, body, content_type):
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
        logger.info({"action": "upload_page_success", "bucket": bucket, "key": key})
    except ClientError as e:
        logger.error({"action": "upload_page_failed", "bucket": bucket, "key": key, "error": str(e)})
        raise


def build_destination_key(page_number, extension):
    return f"result_pages/pages/page_{page_number:04d}{extension}"
