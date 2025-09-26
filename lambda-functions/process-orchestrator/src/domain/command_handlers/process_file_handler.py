import json
import logging
from datetime import datetime
from typing import Dict, Any

from ..commands.process_file_command import ProcessFileCommand
from ..ports.file_storage import FileStorage
from ..ports.pdf_analyzer import PdfAnalyzer
from ..ports.message_queue import MessageQueue
from ..errors.orchestrator_errors import FileProcessingError, RoutingError

logger = logging.getLogger(__name__)

SIZE_THRESHOLD_BYTES = 2 * 1024 * 1024
PAGES_THRESHOLD = 50


class ProcessFileHandler:
    def __init__(
        self,
        file_storage: FileStorage,
        pdf_analyzer: PdfAnalyzer,
        message_queue: MessageQueue
    ):
        self._file_storage = file_storage
        self._pdf_analyzer = pdf_analyzer
        self._message_queue = message_queue

    def handle(self, command: ProcessFileCommand) -> Dict[str, Any]:
        try:
            logger.info(f"Processing file: {command.bucket}/{command.key}")

            is_pdf = self._file_storage.is_pdf_by_head(command.bucket, command.key)
            if not is_pdf:
                self._file_storage.delete_object(command.bucket, command.key)
                logger.info(f"Deleted non-PDF object: {command.bucket}/{command.key}")
                return {
                    'file': f"{command.bucket}/{command.key}",
                    'status': 'deleted_non_pdf',
                    'queue': None,
                    'routing_reason': 'Non-PDF file removed',
                    'file_size_mb': None
                }

            routing_result = self._determine_routing(command)
            message_id = self._send_to_queue(command, routing_result)

            return {
                'file': f"{command.bucket}/{command.key}",
                'status': 'success',
                'queue': routing_result['queue_type'],
                'message_id': message_id,
                'routing_reason': routing_result['routing_reason'],
                'file_size_mb': round(routing_result['file_size'] / 1024 / 1024, 2) if routing_result['file_size'] else None
            }

        except Exception as e:
            logger.error(f"Error processing {command.bucket}/{command.key}: {e}")
            raise FileProcessingError(command.bucket, command.key, str(e))

    def _determine_routing(self, command: ProcessFileCommand) -> Dict[str, Any]:
        try:
            file_size = self._file_storage.get_object_size(command.bucket, command.key)
            logger.info(f"File size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
        except Exception as e:
            logger.error(f"Error getting file size for {command.key}: {e}")
            return {
                'queue_url': command.heavy_queue_url,
                'queue_type': "heavy",
                'routing_reason': "Error getting size; default heavy",
                'file_size': 0
            }

        if file_size >= SIZE_THRESHOLD_BYTES:
            return {
                'queue_url': command.heavy_queue_url,
                'queue_type': "heavy",
                'routing_reason': f"Large file (>=2MB): {file_size/1024/1024:.2f}MB",
                'file_size': file_size
            }

        try:
            local_path = self._file_storage.download_to_tmp(command.bucket, command.key)
            pages = self._pdf_analyzer.count_pages(local_path)
            logger.info(f"PDF pages: {pages}")

            if pages > PAGES_THRESHOLD:
                return {
                    'queue_url': command.heavy_queue_url,
                    'queue_type': "heavy",
                    'routing_reason': f"Pages > {PAGES_THRESHOLD}: {pages}",
                    'file_size': file_size
                }
            else:
                return {
                    'queue_url': command.light_queue_url,
                    'queue_type': "light",
                    'routing_reason': f"Pages <= {PAGES_THRESHOLD}: {pages}",
                    'file_size': file_size
                }
        except Exception as e:
            logger.error(f"Error counting pages for {command.key}: {e}")
            return {
                'queue_url': command.heavy_queue_url,
                'queue_type': "heavy",
                'routing_reason': "Error counting pages; default heavy",
                'file_size': file_size
            }

    def _send_to_queue(self, command: ProcessFileCommand, routing_result: Dict[str, Any]) -> str:
        try:
            message_body = {
                'bucket': command.bucket,
                'key': command.key,
                'file_size': routing_result['file_size'],
                'routing_reason': routing_result['routing_reason'],
                'queue_type': routing_result['queue_type'],
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'orchestrator'
            }

            message_attributes = {
                'FileSize': {'StringValue': str(routing_result['file_size']), 'DataType': 'Number'},
                'QueueType': {'StringValue': routing_result['queue_type'], 'DataType': 'String'},
                'RoutingReason': {'StringValue': routing_result['routing_reason'][:256], 'DataType': 'String'}
            }

            message_id = self._message_queue.send_message(
                routing_result['queue_url'],
                message_body,
                message_attributes
            )

            logger.info(f"Routing decision: {routing_result['routing_reason']} -> {routing_result['queue_type']} queue")
            logger.info(f"Message sent: {message_id}")

            return message_id

        except Exception as e:
            raise RoutingError(routing_result['queue_type'], str(e))