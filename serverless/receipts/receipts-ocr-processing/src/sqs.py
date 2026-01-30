import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def publish_llm_message(sqs_client, queue_url, job_id, page_number, ocr_result_s3_key):
    message = {
        "job_id": job_id,
        "page_number": page_number,
        "ocr_result_s3_key": ocr_result_s3_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message),
    )
    logger.info({"action": "published_llm_message", "job_id": job_id, "page_number": page_number})
