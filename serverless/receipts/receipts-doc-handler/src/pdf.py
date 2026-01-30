import io
import logging

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def read_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    logger.info({"action": "read_pdf", "total_pages": total_pages})
    return reader, total_pages


def split_page(reader, page_num):
    writer = PdfWriter()
    writer.add_page(reader.pages[page_num])

    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer.getvalue()
