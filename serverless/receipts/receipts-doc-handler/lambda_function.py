import io
import json
import logging
import os
import urllib.parse

import boto3
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
sqs_client = boto3.client("sqs")

DESTINATION_BUCKET = os.environ["DESTINATION_BUCKET"]
SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def lambda_handler(event, context):
    records = event.get("Records", [])
    logger.info({"records_count": len(records)})

    for record in records:
        # obtain object info from event
        source_bucket = record["s3"]["bucket"]["name"]
        source_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        logger.info({"action": "processing_object", "bucket": source_bucket, "key": source_key})

        # get file info
        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        file_bytes = response["Body"].read()
        content_type = response.get("ContentType", "")

        ext = os.path.splitext(source_key)[1].lower()
        base_name = os.path.splitext(os.path.basename(source_key))[0]

        # result pages prefix
        complete_prefix = os.path.dirname(source_key)
        prefix = complete_prefix.replace("docs", "result-pages", 1)


        if ext == ".pdf":
            _process_pdf(file_bytes, source_bucket, source_key, base_name, prefix)
        elif ext in IMAGE_EXTENSIONS:
            _process_image(file_bytes, source_bucket, source_key, base_name, prefix, content_type)
        else:
            logger.warning({"action": "skipped_unsupported", "key": source_key, "extension": ext})

    return {"statusCode": 200, "body": f"Processed {len(records)} record(s)"}


def _process_pdf(file_bytes, source_bucket, source_key, base_name, prefix):
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    logger.info({"action": "splitting_pdf", "key": source_key, "total_pages": total_pages})

    for page_num in range(total_pages):
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num])

        page_buffer = io.BytesIO()
        writer.write(page_buffer)
        page_buffer.seek(0)

        dest_key = f"{prefix}/{base_name}/page_{page_num + 1}.pdf" if prefix else f"{base_name}/page_{page_num + 1}.pdf"

        s3_client.put_object(
            Bucket=DESTINATION_BUCKET,
            Key=dest_key,
            Body=page_buffer.getvalue(),
            ContentType="application/pdf",
        )
        logger.info({"action": "uploaded_page", "destination": dest_key, "page": page_num + 1})

        _publish_message(source_bucket, source_key, dest_key, page_num + 1, total_pages)


def _process_image(file_bytes, source_bucket, source_key, base_name, prefix, content_type):
    dest_key = f"{prefix}/{base_name}/{os.path.basename(source_key)}" if prefix else f"{base_name}/{os.path.basename(source_key)}"

    s3_client.put_object(
        Bucket=DESTINATION_BUCKET,
        Key=dest_key,
        Body=file_bytes,
        ContentType=content_type,
    )
    logger.info({"action": "uploaded_image", "destination": dest_key})

    _publish_message(source_bucket, source_key, dest_key, page_number=1, total_pages=1)


def _publish_message(source_bucket, source_key, dest_key, page_number, total_pages):
    message = {
        "source_bucket": source_bucket,
        "source_key": source_key,
        "destination_bucket": DESTINATION_BUCKET,
        "destination_key": dest_key,
        "page_number": page_number,
        "total_pages": total_pages,
    }

    sqs_client.send_message(
        QueueUrl=SQS_QUEUE_URL,
        MessageBody=json.dumps(message),
    )
    logger.info({"action": "published_sqs_message", "destination_key": dest_key})