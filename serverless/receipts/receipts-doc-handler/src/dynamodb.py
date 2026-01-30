import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def create_job(jobs_table, document_key, document_size, uploaded_at, total_pages):
    job_id = str(uuid.uuid4())
    started_processing_at = datetime.now(timezone.utc).isoformat()

    item = {
        "job_id": job_id,
        "status": "INITIATED",
        "document_key": document_key,
        "document_size": document_size,
        "uploaded_at": uploaded_at,
        "started_processing_at": started_processing_at,
        "total_pages": total_pages,
        "ocr_completed_count": 0,
        "llm_completed_count": 0,
        "error_count": 0,
        "current_phase": "OCR",
        "total_mistral_pages": 0,
        "total_bedrock_input_tokens": 0,
        "total_bedrock_output_tokens": 0,
        "estimated_mistral_cost": "0.0",
        "estimated_bedrock_cost": "0.0",
    }

    jobs_table.put_item(Item=item)
    logger.info({"action": "created_job", "job_id": job_id, "document_key": document_key})
    return job_id
