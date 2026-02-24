import logging
import os

import boto3

from src.pdf import read_pdf, split_page
from src.s3 import build_destination_key, get_object, upload_page

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")

DESTINATION_BUCKET = os.environ["DESTINATION_BUCKET"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def lambda_handler(event, context):
    source_bucket = event["bucket"]
    source_key = event["key"]
    logger.info({"action": "processing_object", "bucket": source_bucket, "key": source_key})

    file_bytes, content_type = get_object(s3_client, source_bucket, source_key)

    ext = os.path.splitext(source_key)[1].lower()

    if ext == ".pdf":
        reader, total_pages = read_pdf(file_bytes)
        keys = _process_pdf(reader, total_pages)
    elif ext in IMAGE_EXTENSIONS:
        keys = _process_image(file_bytes, ext, content_type)
    else:
        logger.warning({"action": "skipped_unsupported", "key": source_key, "extension": ext})
        return {"statusCode": 400, "error": f"Unsupported extension: {ext}"}

    return {
        "statusCode": 200,
        "bucket": DESTINATION_BUCKET,
        "keys": keys,
    }


def _process_pdf(reader, total_pages):
    logger.info({"action": "splitting_pdf", "total_pages": total_pages})
    keys = []

    for page_num in range(total_pages):
        page_bytes = split_page(reader, page_num)
        dest_key = build_destination_key(page_num + 1, ".pdf")

        upload_page(s3_client, DESTINATION_BUCKET, dest_key, page_bytes, "application/pdf")
        keys.append(dest_key)

    return keys


def _process_image(file_bytes, ext, content_type):
    dest_key = build_destination_key(1, ext)

    upload_page(s3_client, DESTINATION_BUCKET, dest_key, file_bytes, content_type)
    return [dest_key]
