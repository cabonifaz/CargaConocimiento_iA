"""
RAG Vectorization Lambda — Stage 3 (final) of the RAG ingestion pipeline.

Environment variables:
  DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
  S3_RESULTS_BUCKET    — bucket where chunk JSON files are stored
  WEAVIATE_URL         — Weaviate Cloud cluster URL
  WEAVIATE_API_KEY     — Weaviate API key
  BEDROCK_REGION       — AWS region for Bedrock API calls
  BEDROCK_MODEL_ID     — Cohere model (default: cohere.embed-multilingual-v3)
  BM25_MODEL_ID        — Llama model (default: us.meta.llama4-maverick-17b-instruct-v1:0)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.database import SQLServerClient
from src.embeddings import BedrockBM25Generator, BedrockCohereEmbedder
from src.storage import S3Client, WeaviateClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Stage constants (must match PARAMETROS ID_MAESTRO=7 and ID_MAESTRO=13) ──
ETAPA_VECTORIZACION = 3       # ID_MAESTRO=13, NUM1=3  (Vectorización)
ESTADO_VECTORIZANDO = 6       # ID_MAESTRO=7,  NUM1=6  (Vectorizando)
ESTADO_CARGADO = 7            # ID_MAESTRO=7,  NUM1=7  (Cargado — terminal success)

# ID_MODELO keys used for PARAMETROS cost lookup (ID_MAESTRO=14, NUM2 column)
_ID_MODELO_COHERE = 4   # Cohere Embed Multilingual v3
_ID_MODELO_LLAMA = 8    # Llama 4 Maverick 17B

# Model costs fetched from DB once at cold start — populated in lambda_handler
# {id_modelo: (cost_input_per_million_tokens, cost_output_per_million_tokens)}
_model_costs: dict = {}


def _load_model_costs(db_client: SQLServerClient) -> None:
    """
    Fetch model costs from PARAMETROS (ID_MAESTRO=14) and cache in _model_costs.

    Called once per Lambda container lifetime. If the dict is already populated
    (warm start) this is a no-op.
    """
    global _model_costs
    if not _model_costs:
        _model_costs = db_client.get_model_costs()
        logger.info("Model costs loaded from DB: %s", _model_costs)


def _calculate_cost(
    embed_tokens: int,
    llama_input_tokens: int,
    llama_output_tokens: int,
) -> float:
    """
    Compute total Bedrock cost using rates from PARAMETROS.

    Rates are stored as cost per million tokens (STRING1/STRING2 columns).
    Returns 0.0 if cost data is unavailable (conservative — never over-reports).
    """
    cohere_in_per_m, _ = _model_costs.get(_ID_MODELO_COHERE, (0.0, 0.0))
    llama_in_per_m, llama_out_per_m = _model_costs.get(_ID_MODELO_LLAMA, (0.0, 0.0))

    return (
        embed_tokens       * (cohere_in_per_m  / 1_000_000)
        + llama_input_tokens  * (llama_in_per_m   / 1_000_000)
        + llama_output_tokens * (llama_out_per_m   / 1_000_000)
    )


def _build_clients(bedrock_region: str, embedding_model: str, bm25_model_id: str) -> tuple:
    """Instantiate all clients from environment variables."""
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

    embedder = BedrockCohereEmbedder(
        region=bedrock_region,
        model_id=embedding_model,
    )

    bm25_generator = BedrockBM25Generator(
        region=bedrock_region,
        model_id=bm25_model_id,
    )

    weaviate_client = WeaviateClient(
        url=os.environ["WEAVIATE_URL"],
        api_key=os.environ["WEAVIATE_API_KEY"],
    )

    return db_client, s3_client, embedder, bm25_generator, weaviate_client


def _process_record(
    record: Dict[str, Any],
    db_client: SQLServerClient,
    s3_client: S3Client,
    embedder: BedrockCohereEmbedder,
    bm25_generator: BedrockBM25Generator,
    weaviate_client: WeaviateClient,
    embedding_model: str,
) -> None:
    """
    Process a single SQS record (one document through the full vectorization flow).

    Raises on failure after marking the process as Error in DB.
    """
    # ── 1. Parse SQS message body ──────────────────────────────────────────
    body = json.loads(record["body"])
    id_documento: int = body["id_documento"]
    id_proceso: int = body["id_proceso"]
    id_empresa: int = body["id_empresa"]
    id_area: int = body["id_area"]
    nombre_documento: str = body["nombre_documento"]
    ruta_segmentos: str = body["ruta_segmentos"]

    logger.info(
        f"[doc={id_documento}] Starting vectorization — "
        f"proceso={id_proceso}, file='{nombre_documento}'"
    )

    # ── 2. Download chunks from S3 ─────────────────────────────────────────
    chunks = s3_client.get_chunks(ruta_segmentos)
    logger.info(f"[doc={id_documento}] Retrieved {len(chunks)} chunk(s)")

    # ── 3. Iniciar etapa en DB ─────────────────────────────────────────────
    id_log: Optional[int] = None
    id_log = db_client.iniciar_etapa(
        id_proceso=id_proceso,
        id_etapa=ETAPA_VECTORIZACION,
        estado_procesando=ESTADO_VECTORIZANDO,
    )

    try:
        texts = [c["text"] for c in chunks]

        # ── 4. BM25 keyword generation ─────────────────────────────────────
        bm25_texts, llama_in_tokens, llama_out_tokens = bm25_generator.generate_bm25_texts(texts)

        # ── 5. Embedding vectors ───────────────────────────────────────────
        vectors, embed_tokens = embedder.embed_texts(texts)

        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(vectors)}, expected {len(chunks)}"
            )

        # ── 6. Weaviate upsert ─────────────────────────────────────────────
        collection_name = str(id_empresa)
        doc_id = f"CONOC-{id_documento}"
        doc_title = os.path.splitext(nombre_documento)[0]

        chunks_written = weaviate_client.upsert_chunks(
            chunks=chunks,
            vectors=vectors,
            bm25_texts=bm25_texts,
            doc_id=doc_id,
            company_id=str(id_empresa),
            area_id=str(id_area),
            doc_title=doc_title,
            embedding_model=embedding_model,
            collection_name=collection_name,
            id_documento=id_documento,
            id_proceso=id_proceso,
        )

        # ── 7. Completar etapa en DB ───────────────────────────────────────
        cost_usd = _calculate_cost(embed_tokens, llama_in_tokens, llama_out_tokens)
        db_client.completar_etapa(
            id_log=id_log,
            id_proceso=id_proceso,
            estado_siguiente=ESTADO_CARGADO,
            ruta_resultado=None,   # vectors live in Weaviate, no S3 output
            costo_usd=cost_usd,
        )

        logger.info(
            f"[doc={id_documento}] Vectorization complete — "
            f"chunks={chunks_written}, embed_tokens={embed_tokens}, "
            f"llama_in={llama_in_tokens}, llama_out={llama_out_tokens}, "
            f"cost=${cost_usd:.6f}"
        )

    except Exception as exc:
        error_msg = str(exc)[:200]
        logger.error(f"[doc={id_documento}] Vectorization failed: {error_msg}")

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
        "S3_RESULTS_BUCKET", "WEAVIATE_URL", "WEAVIATE_API_KEY", "BEDROCK_REGION",
    ]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")

    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    bedrock_region = os.environ.get("BEDROCK_REGION", aws_region)
    embedding_model = os.environ.get("BEDROCK_MODEL_ID", "cohere.embed-multilingual-v3")
    bm25_model_id = os.environ.get("BM25_MODEL_ID", "us.meta.llama4-maverick-17b-instruct-v1:0")

    db_client, s3_client, embedder, bm25_generator, weaviate_client = _build_clients(
        bedrock_region, embedding_model, bm25_model_id
    )

    failed_message_ids: List[str] = []

    try:
        with db_client:
            # Load model costs once per container (no-op on warm start)
            _load_model_costs(db_client)

            for record in records:
                message_id = record.get("messageId", "unknown")
                try:
                    _process_record(
                        record, db_client, s3_client,
                        embedder, bm25_generator, weaviate_client,
                        embedding_model,
                    )
                except Exception as e:
                    logger.error(f"Record {message_id} failed: {e}")
                    failed_message_ids.append(message_id)
    finally:
        weaviate_client.close()

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
