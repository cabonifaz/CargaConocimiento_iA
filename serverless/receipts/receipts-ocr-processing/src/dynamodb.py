import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

MISTRAL_UNIT_COST = Decimal("0.001")


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
