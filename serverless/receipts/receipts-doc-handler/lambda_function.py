import io
import json
import logging
import os
import urllib.parse
import uuid
from datetime import datetime, timezone

import boto3
from pypdf import PdfReader, PdfWriter

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
        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        file_bytes = response["Body"].read()
        file_size = len(file_bytes)
        content_type = response.get("ContentType", "")

        ext = os.path.splitext(source_key)[1].lower()
        base_name = os.path.splitext(os.path.basename(source_key))[0]

        # result pages prefix
        complete_prefix = os.path.dirname(source_key)
        prefix = complete_prefix.replace("docs", "result-pages", 1)

        # determine total pages
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)
        elif ext in IMAGE_EXTENSIONS:
            reader = None
            total_pages = 1
        else:
            logger.warning({"action": "skipped_unsupported", "key": source_key, "extension": ext})
            continue

        # create job in DynamoDB
        job_id = _create_job(source_key, file_size, upload_timestamp, total_pages)

        if ext == ".pdf":
            _process_pdf(reader, source_key, base_name, prefix, job_id)
        elif ext in IMAGE_EXTENSIONS:
            _process_image(file_bytes, source_key, base_name, prefix, content_type, job_id)

    return {"statusCode": 200, "body": f"Processed {len(records)} record(s)"}


def _process_pdf(reader, source_key, base_name, prefix, job_id):
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

        _publish_message(job_id, page_num + 1, dest_key)


def _process_image(file_bytes, source_key, base_name, prefix, content_type, job_id):
    dest_key = f"{prefix}/{base_name}/{os.path.basename(source_key)}" if prefix else f"{base_name}/{os.path.basename(source_key)}"

    s3_client.put_object(
        Bucket=DESTINATION_BUCKET,
        Key=dest_key,
        Body=file_bytes,
        ContentType=content_type,
    )
    logger.info({"action": "uploaded_image", "destination": dest_key})

    _publish_message(job_id, page_number=1, dest_key=dest_key)


def _create_job(document_key, document_size, uploaded_at, total_pages):
    job_id = str(uuid.uuid4())
    started_processing_at = datetime.now(timezone.utc).isoformat()

    item = {
        "job_id": job_id,
        "status": "INITIATED",
        "document_key": document_key,
        "document_size": document_size,
        "uploaded_at": uploaded_at,
        "started_processing_at": started_processing_at,
        "total_pages": total_pages,
        "ocr_completed_count": 0,
        "llm_completed_count": 0,
        "error_count": 0,
        "current_phase": "OCR",
        "total_mistral_pages": 0,
        "total_bedrock_input_tokens": 0,
        "total_bedrock_output_tokens": 0,
        "estimated_mistral_cost": "0.0",
        "estimated_bedrock_cost": "0.0",
    }

    jobs_table.put_item(Item=item)
    logger.info({"action": "created_job", "job_id": job_id, "document_key": document_key})
    return job_id


def _publish_message(job_id, page_number, dest_key):
    message = {
        "job_id": job_id,
        "page_number": page_number,
        "s3_bucket": DESTINATION_BUCKET,
        "s3_key": dest_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    sqs_client.send_message(
        QueueUrl=SQS_QUEUE_URL,
        MessageBody=json.dumps(message),
    )
    logger.info({"action": "published_sqs_message", "destination_key": dest_key})