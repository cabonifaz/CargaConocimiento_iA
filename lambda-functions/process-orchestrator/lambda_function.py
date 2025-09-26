import os
import logging
import boto3
from src.entrypoints.lambda_entrypoint import LambdaEntrypoint

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')

heavy_queue_url = os.environ.get('HEAVY_QUEUE_URL')
light_queue_url = os.environ.get('LIGHT_QUEUE_URL')

if not all([heavy_queue_url, light_queue_url]):
    raise ValueError("Missing HEAVY_QUEUE_URL or LIGHT_QUEUE_URL environment variables")

_entrypoint = LambdaEntrypoint(
    s3_client=s3_client,
    sqs_client=sqs_client,
    heavy_queue_url=heavy_queue_url,
    light_queue_url=light_queue_url
)


def lambda_handler(event, context):
    return _entrypoint.handle_event(event, context)