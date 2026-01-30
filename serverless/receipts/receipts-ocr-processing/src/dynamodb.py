import logging
from datetime import datetime, timezone
from decimal import Decimal

logger = logging.getLogger(__name__)

MISTRAL_UNIT_COST = Decimal("0.001")


def insert_cost_tracking(cost_tracking_table, job_id, page_number):
    timestamp = datetime.now(timezone.utc).isoformat()
    composite_sort_key = f"OCR#page_{page_number:04d}"

    item = {
        "job_id": job_id,
        "composite_sort_key": composite_sort_key,
        "stage": "OCR",
        "provider": "Mistral",
        "pages_processed": 1,
        "unit_cost": MISTRAL_UNIT_COST,
        "total_cost": MISTRAL_UNIT_COST,
        "timestamp": timestamp,
    }

    cost_tracking_table.put_item(Item=item)
    logger.info({"action": "inserted_cost_tracking", "job_id": job_id, "page_number": page_number})


def update_job_counters(processing_jobs_table, job_id):
    processing_jobs_table.update_item(
        Key={"job_id": job_id},
        UpdateExpression=(
            "ADD ocr_completed_count :inc, "
            "total_mistral_pages :inc, "
            "estimated_mistral_cost :cost"
        ),
        ExpressionAttributeValues={
            ":inc": 1,
            ":cost": MISTRAL_UNIT_COST,
        },
    )
    logger.info({"action": "updated_job_counters", "job_id": job_id})
