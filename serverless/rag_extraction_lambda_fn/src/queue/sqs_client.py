"""SQS publisher for forwarding messages to downstream pipeline queues."""

import json
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()


class SQSPublisher:
    """
    Publishes JSON messages to an SQS queue.

    queue_url: the URL of the destination SQS queue.
    """

    def __init__(self, queue_url: str):
        if not queue_url:
            raise ValueError("queue_url is required")
        self.queue_url = queue_url
        self._client = None

    @property
    def client(self):
        """Lazy boto3 SQS client (reused across calls within the same invocation)."""
        if self._client is None:
            self._client = boto3.client("sqs")
        return self._client

    def publish(self, message: dict) -> str:
        """
        Send a JSON-serialised message to the queue.

        Args:
            message: dict to serialise as the SQS MessageBody.

        Returns:
            SQS MessageId of the sent message.

        Raises:
            ClientError: on SQS API failure.
        """
        body = json.dumps(message)
        try:
            response = self.client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=body,
            )
            message_id = response["MessageId"]
            logger.info(f"Published message {message_id} to {self.queue_url}")
            return message_id
        except ClientError as e:
            logger.error(f"Failed to publish message to {self.queue_url}: {e}")
            raise
