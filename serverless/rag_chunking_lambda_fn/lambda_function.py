"""
RAG Chunking Lambda — Stage 2 of the RAG ingestion pipeline.

Triggered by: SQS chunking queue (published by rag_extraction_lambda_fn)
Runtime:      Python 3.11

Flow per SQS record:
  1. Parse message → id_documento, id_proceso, id_empresa, id_area,
                     nombre_documento, ruta_markdown, id_modelo_embedding
  2. Download .md file from S3 (S3_RESULTS_BUCKET / ruta_markdown)
  3. SP_RAG_INGESTA_INICIAR_ETAPA  → state=4 (Segmentando), obtain ID_LOG
  4. Split .md on PAGE_SEPARATOR → pages list
     chunk_markdown(pages) → list of chunk dicts with page/section metadata
  5. Upload JSON array of chunks to S3:
     ingest-results/chunking/{id_empresa}/{id_area}/{nombre_sin_extension}.json
  6. SP_RAG_INGESTA_COMPLETAR_ETAPA → state=5 (En cola vectorización),
     record result key, cost=0.0 (local computation)
  7. Publish message to embedding SQS queue → triggers Stage 3
  On error (after step 3):
     SP_RAG_INGESTA_FALLAR_ETAPA → state=8 (Error)
     re-raise → SQS partial batch failure → DLQ after maxReceiveCount

Environment variables:
  DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
  S3_RESULTS_BUCKET     — bucket where .md inputs are read and .json results written
  EMBEDDING_QUEUE_URL   — SQS URL of the embedding (vectorización) queue
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.database import SQLServerClient
from src.storage import S3Client
from src.queue import SQSPublisher
from src.chunker import chunk_markdown

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Stage constants (must match PARAMETROS ID_MAESTRO=7 and ID_MAESTRO=13) ──
ETAPA_SEGMENTACION = 2        # ID_MAESTRO=13, NUM1=2  (Segmentación)
ESTADO_SEGMENTANDO = 4        # ID_MAESTRO=7,  NUM1=4  (Segmentando)
ESTADO_EN_COLA_VEC = 5        # ID_MAESTRO=7,  NUM1=5  (En cola vectorización)

# Separator written between pages by the extraction Lambda
PAGE_SEPARATOR = "\n\n---\n\n"


def _build_clients() -> tuple:
    """Read env vars and instantiate all clients."""
    db_client = SQLServerClient(
        server=os.environ["DB_SERVER"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        port=int(os.environ.get("DB_PORT", "1433")),
    )

    s3_client = S3Client(
        results_bucket=os.environ["S3_RESULTS_BUCKET"],
    )

    sqs_publisher = SQSPublisher(
        queue_url=os.environ["EMBEDDING_QUEUE_URL"],
    )

    return db_client, s3_client, sqs_publisher


def _process_record(
    record: Dict[str, Any],
    db_client: SQLServerClient,
    s3_client: S3Client,
    sqs_publisher: SQSPublisher,
) -> None:
    """
    Process a single SQS record (one document).

    Raises on failure after marking the process as Error in DB.
    """
    # ── 1. Parse SQS message body ──────────────────────────────────────────
    body = json.loads(record["body"])
    id_documento: int = body["id_documento"]
    id_proceso: int = body["id_proceso"]
    id_empresa: int = body["id_empresa"]
    id_area: int = body["id_area"]
    nombre_documento: str = body["nombre_documento"]
    ruta_markdown: str = body["ruta_markdown"]
    id_modelo_embedding: int = body["id_modelo_embedding"]

    logger.info(
        f"[doc={id_documento}] Starting chunking — "
        f"proceso={id_proceso}, file='{nombre_documento}'"
    )

    # ── 2. Download .md from S3 ────────────────────────────────────────────
    markdown_text = s3_client.download_markdown(ruta_markdown)

    # ── 3. Iniciar etapa en DB ─────────────────────────────────────────────
    id_log: Optional[int] = None
    id_log = db_client.iniciar_etapa(
        id_proceso=id_proceso,
        id_etapa=ETAPA_SEGMENTACION,
        estado_procesando=ESTADO_SEGMENTANDO,
    )

    try:
        # ── 4. Split pages and chunk ───────────────────────────────────────
        pages = markdown_text.split(PAGE_SEPARATOR)
        logger.info(f"[doc={id_documento}] Split into {len(pages)} page(s)")

        chunks = chunk_markdown(pages, nombre_documento)
        logger.info(f"[doc={id_documento}] Produced {len(chunks)} chunk(s)")

        # ── 5. Upload chunks JSON to S3 ────────────────────────────────────
        base_filename = os.path.splitext(nombre_documento)[0]
        result_key = (
            f"ingest-results/chunks/{id_empresa}/{id_area}/{base_filename}.json"
        )
        s3_client.upload_chunks_json(result_key, chunks)

        # ── 6. Completar etapa en DB ───────────────────────────────────────
        db_client.completar_etapa(
            id_log=id_log,
            id_proceso=id_proceso,
            estado_siguiente=ESTADO_EN_COLA_VEC,
            ruta_resultado=result_key,
            costo_usd=0.0,
        )

        # ── 7. Enqueue next stage (embedding) ─────────────────────────────
        sqs_publisher.publish({
            "id_documento": id_documento,
            "id_proceso": id_proceso,
            "id_empresa": id_empresa,
            "id_area": id_area,
            "nombre_documento": nombre_documento,
            "ruta_segmentos": result_key,
            "id_modelo_embedding": id_modelo_embedding,
        })

        logger.info(
            f"[doc={id_documento}] Chunking complete — "
            f"chunks={len(chunks)}, key='{result_key}'"
        )

    except Exception as exc:
        error_msg = str(exc)[:200]
        logger.error(f"[doc={id_documento}] Chunking failed: {error_msg}")

        # Mark process as Error in DB (best-effort — don't mask original exception)
        if id_log is not None:
            try:
                db_client.fallar_etapa(
                    id_log=id_log,
                    id_proceso=id_proceso,
                    mensaje_error=error_msg,
                )
            except Exception as db_exc:
                logger.error(
                    f"[doc={id_documento}] Also failed to call fallar_etapa: {db_exc}"
                )

        raise  # re-raise → SQS partial batch failure → retry → DLQ


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    SQS-triggered Lambda entry point.

    Uses partial batch failure reporting: records that fail are returned in
    batchItemFailures so SQS retries only those records (not the whole batch).
    Requires "Report batch item failures" enabled on the SQS event source mapping.
    """
    records: List[Dict[str, Any]] = event.get("Records", [])
    logger.info(f"Received {len(records)} SQS record(s)")

    # Validate required environment variables early (fail fast on misconfiguration)
    required_env = [
        "DB_SERVER", "DB_NAME", "DB_USER", "DB_PASSWORD",
        "S3_RESULTS_BUCKET", "EMBEDDING_QUEUE_URL",
    ]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")

    db_client, s3_client, sqs_publisher = _build_clients()

    failed_message_ids: List[str] = []

    with db_client:
        for record in records:
            message_id = record.get("messageId", "unknown")
            try:
                _process_record(record, db_client, s3_client, sqs_publisher)
            except Exception as e:
                logger.error(f"Record {message_id} failed: {e}")
                failed_message_ids.append(message_id)

    if failed_message_ids:
        logger.warning(
            f"{len(failed_message_ids)}/{len(records)} record(s) failed: {failed_message_ids}"
        )

    # Return partial batch failures — SQS retries only these records
    return {
        "batchItemFailures": [
            {"itemIdentifier": mid} for mid in failed_message_ids
        ]
    }
