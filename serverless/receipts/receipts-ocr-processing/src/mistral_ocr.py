import logging
import time

from mistralai import Mistral

logger = logging.getLogger(__name__)

OCR_MODEL = "mistral-ocr-latest"


def create_client(api_key):
    return Mistral(api_key=api_key)


def process_page(client, presigned_url):
    start_time = time.time()

    response = client.ocr.process(
        model=OCR_MODEL,
        document={
            "type": "document_url",
            "document_url": presigned_url,
        },
    )

    duration = time.time() - start_time
    logger.info({"action": "ocr_completed", "duration_seconds": round(duration, 3)})

    pages = response.pages
    extracted_text = "\n\n".join(page.markdown for page in pages)

    return extracted_text, duration
