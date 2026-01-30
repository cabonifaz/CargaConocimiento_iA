import json
import logging
import os

import boto3

from src.dynamodb import update_job_counters
from src.mistral_ocr import create_client, process_page
from src.s3 import generate_presigned_url, upload_ocr_result
from src.sqs import publish_llm_message

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
sqs_client = boto3.client("sqs")
dynamodb_resource = boto3.resource("dynamodb")

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
OCR_RESULT_DESTINATION_BUCKET = os.environ["OCR_RESULT_DESTINATION_BUCKET"]
LLM_READY_SQS_QUEUE = os.environ["LLM_READY_SQS_QUEUE"]
PROCESSING_JOBS_TABLE = os.environ["PROCESSING_JOBS_TABLE"]

processing_jobs_table = dynamodb_resource.Table(PROCESSING_JOBS_TABLE)

mistral_client = create_client(MISTRAL_API_KEY)


def lambda_handler(event, context):
    records = event.get("Records", [])
    logger.info({"records_count": len(records)})

    for record in records:
        body = json.loads(record["body"])
        job_id = body["job_id"]
        page_number = body["page_number"]
        s3_bucket = body["s3_bucket"]
        s3_key = body["s3_key"]
        logger.info({"action": "processing_page", "job_id": job_id, "page_number": page_number})

        # generate presigned url for the page
        presigned_url = generate_presigned_url(s3_client, s3_bucket, s3_key)

        # call mistral ocr
        extracted_text, duration = process_page(mistral_client, presigned_url)

        # build and upload ocr result
        ocr_result = {
            "page_number": page_number,
            "extracted_text": extracted_text,
            "processing_duration_seconds": round(duration, 3),
        }
        ocr_result_key = upload_ocr_result(
            s3_client, OCR_RESULT_DESTINATION_BUCKET, job_id, page_number, ocr_result
        )

        # update job counters
        update_job_counters(processing_jobs_table, job_id)

        # publish message to llm queue
        publish_llm_message(sqs_client, LLM_READY_SQS_QUEUE, job_id, page_number, ocr_result_key)

        logger.info({"action": "page_processed_successfully", "job_id": job_id, "page_number": page_number})

    return {"statusCode": 200, "body": f"Processed {len(records)} record(s)"}
