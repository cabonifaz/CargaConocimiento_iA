"""Lambda handler for embedding chunks and upserting to Weaviate."""

import logging
import os
from typing import Dict, Any

from src.embeddings import BedrockCohereEmbedder
from src.storage import S3Client, WeaviateClient
from src.database import SQLServerClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for embedding chunks and upserting to Weaviate.

    Expected event structure:
    {
        "id_carga": 1,                    // int
        "company_id": "1003",             // string
        "area_id": "4",                   // string
        "ruta_segmentos": "prefix/documento.json",
        "embedding_model": "cohere.embed-multilingual-v3"  // optional
    }

    Returns:
    {
        "chunks_written": 42,
        "collection_name": "1003",
        "doc_id": "documento_123",
        "id_carga": 1
    }
    """
    try:
        # ========================================
        # 1. EXTRACT EVENT PARAMETERS
        # ========================================
        id_carga = event.get('id_carga')
        company_id = event.get('company_id')  # Already string
        area_id = event.get('area_id')        # Already string
        ruta_segmentos = event.get('ruta_segmentos')

        if id_carga is None:
            raise ValueError("id_carga is required")
        if not company_id:
            raise ValueError("company_id is required")
        if not area_id:
            raise ValueError("area_id is required")
        if not ruta_segmentos:
            raise ValueError("ruta_segmentos is required")

        # ========================================
        # 2. READ ENVIRONMENT VARIABLES
        # ========================================
        # S3 configuration
        s3_bucket = os.environ.get('S3_CHUNKS_BUCKET')
        if not s3_bucket:
            raise ValueError("S3_CHUNKS_BUCKET environment variable is required")

        # AWS/Bedrock configuration
        aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        bedrock_region = os.environ.get('BEDROCK_REGION', aws_region)
        embedding_model = event.get('embedding_model') or os.environ.get('BEDROCK_MODEL_ID', 'cohere.embed-multilingual-v3')

        # Weaviate configuration
        weaviate_url = os.environ.get('WEAVIATE_URL')
        weaviate_api_key = os.environ.get('WEAVIATE_API_KEY')

        if not weaviate_url or not weaviate_api_key:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY environment variables are required")

        # SQL Server configuration (required)
        db_server = os.environ.get('DB_SERVER')
        db_name = os.environ.get('DB_NAME')
        db_user = os.environ.get('DB_USER')
        db_password = os.environ.get('DB_PASSWORD')
        db_port = int(os.environ.get('DB_PORT', '1433'))

        if not all([db_server, db_name, db_user, db_password]):
            raise ValueError("Database credentials are required: DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD")

        # ========================================
        # 3. RETRIEVE CHUNKS FROM S3
        # ========================================
        s3_client = S3Client(bucket_name=s3_bucket)
        chunks = s3_client.get_chunks(key=ruta_segmentos)

        # Derive doc_id from filename field in chunks (remove .pdf extension)
        filename = chunks[0].get('filename', 'document.pdf') if chunks else 'document.pdf'
        doc_id = filename.replace('.pdf', '').replace('.PDF', '')
        doc_title = doc_id  # Same as doc_id
        collection_name = company_id

        logger.info(f"Processing {len(chunks)} chunks for doc_id={doc_id}, company_id={company_id}, area_id={area_id}, id_carga={id_carga}")

        # ========================================
        # 4. GENERATE EMBEDDINGS
        # ========================================
        embedder = BedrockCohereEmbedder(
            region=bedrock_region,
            model_id=embedding_model,
        )

        texts = [chunk['text'] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        vectors = embedder.embed_texts(texts)

        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embeddings count mismatch: got {len(vectors)}, expected {len(chunks)}")

        logger.info(f"Generated {len(vectors)} embeddings")

        # ========================================
        # 5. EXECUTE SP_CARGA_CONOC_PROCESO_VECTORIZACION
        # ========================================
        try:
            with SQLServerClient(
                server=db_server,
                database=db_name,
                user=db_user,
                password=db_password,
                port=db_port,
            ) as sql_client:
                sql_client.execute_sp_vectorizacion(id_carga=id_carga)
        except Exception as e:
            logger.error(f"Failed to execute SP_CARGA_CONOC_PROCESO_VECTORIZACION: {e}")
            raise  # Re-raise to stop processing if SP fails

        # ========================================
        # 6. UPSERT TO WEAVIATE
        # ========================================
        weaviate_client = WeaviateClient(
            url=weaviate_url,
            api_key=weaviate_api_key,
            bedrock_region=bedrock_region,
        )

        try:
            chunks_written = weaviate_client.upsert_chunks(
                chunks=chunks,
                vectors=vectors,
                doc_id=doc_id,
                company_id=company_id,
                area_id=area_id,
                doc_title=doc_title,
                embedding_model=embedding_model,
                collection_name=collection_name,
            )

            logger.info(f"Successfully processed document: {chunks_written} chunks written to collection '{collection_name}'")

            # ========================================
            # 7. EXECUTE SP_CARGA_CONOC_PROCESO_COMPLETADO
            # ========================================
            if all([db_server, db_name, db_user, db_password]):
                try:
                    with SQLServerClient(
                        server=db_server,
                        database=db_name,
                        user=db_user,
                        password=db_password,
                        port=db_port,
                    ) as sql_client:
                        sql_client.execute_sp_completado(id_carga=id_carga)
                except Exception as e:
                    logger.warning(f"Failed to execute SP_CARGA_CONOC_PROCESO_COMPLETADO: {e}")
                    # Log but don't fail the entire process
            else:
                logger.warning("SQL Server credentials not configured, skipping SP_CARGA_CONOC_PROCESO_COMPLETADO")

            return {
                "chunks_written": chunks_written,
                "collection_name": collection_name,
                "doc_id": doc_id,
                "company_id": company_id,
                "area_id": area_id,
                "id_carga": id_carga,
            }

        finally:
            weaviate_client.close()

    except Exception as e:
        logger.error(f"Error processing chunks: {str(e)}", exc_info=True)
        raise
