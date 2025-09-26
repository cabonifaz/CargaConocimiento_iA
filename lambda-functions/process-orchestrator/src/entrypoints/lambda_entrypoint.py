import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from ..domain.commands.process_file_command import ProcessFileCommand
from ..domain.command_handlers.process_file_handler import ProcessFileHandler
from ..adapters.s3_file_storage_client import S3FileStorageClient
from ..adapters.pypdf_pdf_analyzer import PyPdfAnalyzer
from ..adapters.sqs_message_queue_client import SqsMessageQueueClient

logger = logging.getLogger(__name__)


class LambdaEntrypoint:
    def __init__(self, s3_client: Any, sqs_client: Any, heavy_queue_url: str, light_queue_url: str):
        self._heavy_queue_url = heavy_queue_url
        self._light_queue_url = light_queue_url

        file_storage = S3FileStorageClient(s3_client)
        pdf_analyzer = PyPdfAnalyzer()
        message_queue = SqsMessageQueueClient(sqs_client)

        self._handler = ProcessFileHandler(
            file_storage=file_storage,
            pdf_analyzer=pdf_analyzer,
            message_queue=message_queue
        )

    def handle_event(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        results = []

        for record in event.get('Records', []):
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']

            command = ProcessFileCommand(
                bucket=bucket,
                key=key,
                heavy_queue_url=self._heavy_queue_url,
                light_queue_url=self._light_queue_url
            )

            result = self._handler.handle(command)
            results.append(result)

        logger.info(f"Process completed. Results: {results}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'PDF orchestration completed',
                'processed_files': len(results),
                'results': results,
                'timestamp': datetime.utcnow().isoformat()
            })
        }