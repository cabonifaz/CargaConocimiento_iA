"""
RAG Extraction Lambda — Stage 1 of the RAG ingestion pipeline.

Triggered by: SQS extraction queue (published by FastAPI /register_ingestion endpoint)
Runtime:      Python 3.11

Flow per SQS record:
  1. Parse message → id_documento, id_proceso, ruta_documento, nombre_documento,
                     id_empresa, id_area, id_modelo_embedding
  2. Generate presigned GET URL for the source PDF in S3
  3. SP_RAG_INGESTA_INICIAR_ETAPA  → state=2 (Extrayendo texto), obtain ID_LOG
  4. Mistral OCR on presigned URL  → list of page markdown strings
  5. Join pages with page-break separator → full markdown document
  6. Upload .md to S3:  ingest-results/extraction/{id_empresa}/{id_area}/{nombre}.md
  7. SP_RAG_INGESTA_COMPLETAR_ETAPA → state=3 (En cola segmentación), record cost & S3 key
  8. Publish message to chunking SQS queue → triggers Stage 2
  On error (after step 3):
     SP_RAG_INGESTA_FALLAR_ETAPA → state=8 (Error)
     re-raise → SQS partial batch failure → DLQ after maxReceiveCount

Environment variables:
  DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
  MISTRAL_API_KEY
  S3_DOCUMENTS_BUCKET   — bucket where uploaded PDFs live
  S3_RESULTS_BUCKET     — bucket where .md results are written
  CHUNKING_QUEUE_URL    — SQS URL of the chunking (segmentation) queue
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

from src.database import SQLServerClient
from src.storage import S3Client
from src.ocr import MistralOCRClient
from src.queue import SQSPublisher

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Stage constants (must match PARAMETROS ID_MAESTRO=7 and ID_MAESTRO=13) ──
ETAPA_EXTRACCION = 1          # ID_MAESTRO=13, NUM1=1  (Extracción)
ESTADO_EXTRAYENDO = 2         # ID_MAESTRO=7,  NUM1=2  (Extrayendo texto)
ESTADO_EN_COLA_SEG = 3        # ID_MAESTRO=7,  NUM1=3  (En cola segmentación)

PAGE_SEPARATOR = "\n\n---\n\n"


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, EndpointConnectionError):
        return "No se pudo conectar con el servicio OCR"
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "NoSuchKey":
            return "El archivo PDF no fue encontrado en el almacenamiento"
        if code == "AccessDenied":
            return "Sin permisos para acceder al archivo en el almacenamiento"
        return "Error de comunicación con los servicios de almacenamiento"
    if isinstance(exc, BotoCoreError):
        return "Error de conexión con los servicios de AWS"
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if "empty pages" in msg:
            return "El documento no contiene páginas de texto reconocible"
        if "Not connected" in msg:
            return "Error de conexión a la base de datos"
    if isinstance(exc, ValueError):
        return "El documento tiene un formato inválido"
    return "Error al extraer el texto del documento"


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
        documents_bucket=os.environ["S3_DOCUMENTS_BUCKET"],
        results_bucket=os.environ["S3_RESULTS_BUCKET"],
    )

    ocr_client = MistralOCRClient(
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    sqs_publisher = SQSPublisher(
        queue_url=os.environ["CHUNKING_QUEUE_URL"],
    )

    return db_client, s3_client, ocr_client, sqs_publisher


def _process_record(
    record: Dict[str, Any],
    db_client: SQLServerClient,
    s3_client: S3Client,
    ocr_client: MistralOCRClient,
    sqs_publisher: SQSPublisher,
    ocr_cost_per_1000_pages: float,
) -> None:
    """
    Process a single SQS record (one document).

    Raises on failure after marking the process as Error in DB.
    """
    # ── 1. Parse SQS message body ──────────────────────────────────────────
    body = json.loads(record["body"])
    id_documento: int = body["id_documento"]
    id_proceso: int = body["id_proceso"]
    ruta_documento: str = body["ruta_documento"]
    nombre_documento: str = body["nombre_documento"]
    id_empresa: int = body["id_empresa"]
    id_area: int = body["id_area"]
    id_modelo_embedding: int = body["id_modelo_embedding"]

    logger.info(
        f"[doc={id_documento}] Starting extraction — "
        f"proceso={id_proceso}, file='{nombre_documento}'"
    )

    # ── 2. Presigned URL for source PDF ────────────────────────────────────
    presigned_url = s3_client.get_presigned_url(ruta_documento)

    # ── 3. Iniciar etapa en DB ─────────────────────────────────────────────
    id_log: Optional[int] = None
    id_log = db_client.iniciar_etapa(
        id_proceso=id_proceso,
        id_etapa=ETAPA_EXTRACCION,
        estado_procesando=ESTADO_EXTRAYENDO,
    )

    try:
        # ── 4. Mistral OCR ─────────────────────────────────────────────────
        pages_markdown, total_pages = ocr_client.extract_text(presigned_url)
        logger.info(f"[doc={id_documento}] OCR produced {total_pages} page(s)")

        # ── 5. Join pages ──────────────────────────────────────────────────
        full_markdown = PAGE_SEPARATOR.join(pages_markdown)

        # ── 6. Build output S3 key and upload ─────────────────────────────
        base_filename = os.path.splitext(nombre_documento)[0]
        result_key = f"ingest-results/extraction/{id_empresa}/{id_area}/{base_filename}.md"
        s3_client.upload_markdown(result_key, full_markdown)

        # ── 7. Calculate cost ──────────────────────────────────────────────
        cost_usd = MistralOCRClient.calculate_cost(total_pages, ocr_cost_per_1000_pages)

        # ── 8. Completar etapa en DB ───────────────────────────────────────
        db_client.completar_etapa(
            id_log=id_log,
            id_proceso=id_proceso,
            estado_siguiente=ESTADO_EN_COLA_SEG,
            ruta_resultado=result_key,
            costo_usd=cost_usd,
        )

        # ── 9. Enqueue next stage (chunking) ───────────────────────────────
        sqs_publisher.publish({
            "id_documento": id_documento,
            "id_proceso": id_proceso,
            "id_empresa": id_empresa,
            "id_area": id_area,
            "nombre_documento": nombre_documento,
            "ruta_markdown": result_key,
            "id_modelo_embedding": id_modelo_embedding,
        })

        logger.info(
            f"[doc={id_documento}] Extraction complete — "
            f"pages={total_pages}, cost=${cost_usd:.6f}, key='{result_key}'"
        )

    except Exception as exc:
        tech_msg = str(exc)
        friendly_msg = _friendly_error(exc)
        logger.error(f"[doc={id_documento}] Extraction failed: {tech_msg}")

        # Mark process as Error in DB (best-effort — don't mask original exception)
        if id_log is not None:
            try:
                db_client.fallar_etapa(
                    id_log=id_log,
                    id_proceso=id_proceso,
                    mensaje_error=friendly_msg,
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
        "MISTRAL_API_KEY", "S3_DOCUMENTS_BUCKET", "S3_RESULTS_BUCKET",
        "CHUNKING_QUEUE_URL",
    ]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")

    db_client, s3_client, ocr_client, sqs_publisher = _build_clients()

    failed_message_ids: List[str] = []

    with db_client:
        ocr_cost_per_1000_pages = db_client.get_ocr_cost_per_1000_pages()

        for record in records:
            message_id = record.get("messageId", "unknown")
            try:
                _process_record(record, db_client, s3_client, ocr_client, sqs_publisher, ocr_cost_per_1000_pages)
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
