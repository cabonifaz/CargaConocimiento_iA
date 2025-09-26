import json
from typing import Dict, Any

from ..domain.ports.message_queue import MessageQueue


class SqsMessageQueueClient(MessageQueue):
    def __init__(self, sqs_client: object):
        self._sqs_client = sqs_client

    def send_message(self, queue_url: str, message_body: Dict[str, Any], message_attributes: Dict[str, Any]) -> str:
        response = self._sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message_body),
            MessageAttributes=message_attributes
        )
        return response['MessageId']