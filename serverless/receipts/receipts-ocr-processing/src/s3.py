import json
import logging

logger = logging.getLogger(__name__)

PRESIGNED_URL_EXPIRATION = 300


def generate_presigned_url(s3_client, bucket, key):
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=PRESIGNED_URL_EXPIRATION,
    )
    logger.info({"action": "generated_presigned_url", "bucket": bucket, "key": key})
    return url


def upload_ocr_result(s3_client, bucket, job_id, page_number, result):
    key = f"ocr_results/{job_id}/pages/page_{page_number:04d}.json"

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(result),
        ContentType="application/json",
    )
    logger.info({"action": "uploaded_ocr_result", "key": key})
    return key
