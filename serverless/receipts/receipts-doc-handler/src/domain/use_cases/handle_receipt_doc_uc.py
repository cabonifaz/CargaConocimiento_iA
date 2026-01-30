import logging
from typing import Optional, cast
from dataclasses import dataclass, astuple
from datetime import datetime
import uuid
from ...utils import destruct_key
from ...ports.pdf_handler import PDFHandlerPort, ReadPDFResponse
from ...ports.storage import StoragePort

@dataclass
class HandleReceiptDocUCResponse:
    id: str
    created_at: datetime
    size: int
    total_pages: int
    original_key: str

class HandleReceiptDocUC:
    def __init__(
        self,
        storage: StoragePort,
        pdf_handler: PDFHandlerPort,
        logger: logging.Logger
    ):
        self._storage = storage
        self._pdf_handler = pdf_handler
        self._logger = logger

    def execute(
        self,
        bucket: str,
        destination_bucket: str,
        raw_key: str,
        parent_prefix: str, 
        destination_parent_prefix: str
        
    ) -> Optional[HandleReceiptDocUCResponse]:
        key_info = destruct_key(
            raw_key=raw_key,
            parent_prefix=parent_prefix,
            destination_parent_prefix=destination_parent_prefix
        )

        key = key_info.key
        prefix = key_info.destination_prefix
        base_name = key_info.base_name

        size = self._storage.get_size(bucket=bucket, key=key)

        if not size:
            self._logger.warning(f"Document not found")
            raise ValueError("")
        
        get_obj_response = self._storage.get_object(bucket=bucket, key=key)

        file_bytes = cast(bytes, get_obj_response["Body"].read() )
        content_type = get_obj_response.get("ContentType", "")

        pdf_reading = self._pdf_handler.read_pdf(bytes=file_bytes)

        total_pages = pdf_reading.total_pages
        reader = pdf_reading.reader

        for page_num in range(total_pages):
            single_page_bytes = self._pdf_handler.write_pdf(reader.pages[page_num])

            destination_key = f"{prefix}/{base_name}/page_{page_num + 1}.pdf" if prefix else f"{base_name}/page_{page_num + 1}.pdf"

            self._storage.put_object(bucket=destination_bucket, key=destination_key, data=single_page_bytes)

        return HandleReceiptDocUCResponse(
            id=uuid.uuid4(),
            created_at=datetime.now(),
            size=size,
            total_pages=total_pages,
            original_key=key
        )




