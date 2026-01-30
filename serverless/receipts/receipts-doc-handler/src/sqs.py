import json
import logging
from datetime import datetime, timezone

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def publish_page_message(sqs_client, queue_url, job_id, page_number, s3_bucket, s3_key):
    message = {
        "job_id": job_id,
        "page_number": page_number,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message),
        )
        logger.info({"action": "publish_message_success", "job_id": job_id, "page_number": page_number, "s3_key": s3_key})
    except ClientError as e:
        logger.error({"action": "publish_message_failed", "job_id": job_id, "page_number": page_number, "error": str(e)})
        raise
