import logging
import os
import urllib.parse

import boto3

from src.dynamodb import create_job
from src.pdf import read_pdf, split_page
from src.s3 import build_destination_key, get_object, upload_page
from src.sqs import publish_page_message

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
sqs_client = boto3.client("sqs")
dynamodb_resource = boto3.resource("dynamodb")

DESTINATION_BUCKET = os.environ["DESTINATION_BUCKET"]
SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
JOBS_TABLE_NAME = os.environ["JOBS_TABLE_NAME"]

jobs_table = dynamodb_resource.Table(JOBS_TABLE_NAME)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def lambda_handler(event, context):
    records = event.get("Records", [])
    logger.info({"records_count": len(records)})

    for record in records:
        # obtain object info from event
        source_bucket = record["s3"]["bucket"]["name"]
        source_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        upload_timestamp = record["eventTime"]
        logger.info({"action": "processing_object", "bucket": source_bucket, "key": source_key, "upload_timestamp": upload_timestamp})

        # get file info
        file_bytes, content_type = get_object(s3_client, source_bucket, source_key)
        file_size = len(file_bytes)

        ext = os.path.splitext(source_key)[1].lower()

        # determine total pages
        if ext == ".pdf":
            reader, total_pages = read_pdf(file_bytes)
        elif ext in IMAGE_EXTENSIONS:
            reader = None
            total_pages = 1
        else:
            logger.warning({"action": "skipped_unsupported", "key": source_key, "extension": ext})
            continue

        # create job in DynamoDB
        job_id = create_job(jobs_table, source_key, file_size, upload_timestamp, total_pages)

        if ext == ".pdf":
            _process_pdf(reader, job_id, total_pages)
        elif ext in IMAGE_EXTENSIONS:
            _process_image(file_bytes, ext, content_type, job_id)

    return {"statusCode": 200, "body": f"Processed {len(records)} record(s)"}


def _process_pdf(reader, job_id, total_pages):
    logger.info({"action": "splitting_pdf", "total_pages": total_pages})

    for page_num in range(total_pages):
        page_bytes = split_page(reader, page_num)
        dest_key = build_destination_key(job_id, page_num + 1, ".pdf")

        upload_page(s3_client, DESTINATION_BUCKET, dest_key, page_bytes, "application/pdf")
        publish_page_message(sqs_client, SQS_QUEUE_URL, job_id, page_num + 1, DESTINATION_BUCKET, dest_key)


def _process_image(file_bytes, ext, content_type, job_id):
    dest_key = build_destination_key(job_id, 1, ext)

    upload_page(s3_client, DESTINATION_BUCKET, dest_key, file_bytes, content_type)
    publish_page_message(sqs_client, SQS_QUEUE_URL, job_id, 1, DESTINATION_BUCKET, dest_key)
