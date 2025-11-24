"""Lambda handler for embedding chunks and upserting to Weaviate."""

import logging
import os
from typing import Dict, Any

from embedder import BedrockCohereEmbedder
from weaviate_client import WeaviateClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for embedding chunks and upserting to Weaviate.

    Expected event structure:
    {
        "chunks": [...],  # Output from norm_chunk_lambda_fn
        "company_id": "304",
        "area_id": "1"
    }

    Returns:
    {
        "chunks_written": 42,
        "collection_name": "304",
        "doc_id": "documento_123"
    }
    """
    try:
        # Required fields
        chunks = event.get('chunks', [])
        company_id = event.get('company_id')
        area_id = event.get('area_id')

        if not chunks:
            raise ValueError("chunks array is required")
        if not company_id:
            raise ValueError("company_id is required")
        if not area_id:
            raise ValueError("area_id is required")

        # Derive doc_id from filename field in chunks (remove .pdf extension)
        filename = chunks[0].get('filename', 'document.pdf') if chunks else 'document.pdf'
        doc_id = filename.replace('.pdf', '').replace('.PDF', '')
        doc_title = doc_id  # Same as doc_id

        # Get embedding model from env var only
        embedding_model = os.environ.get('BEDROCK_MODEL_ID', 'cohere.embed-multilingual-v3')

        # Collection name is always company_id
        collection_name = company_id

        logger.info(f"Processing {len(chunks)} chunks for doc_id={doc_id}, company_id={company_id}, area_id={area_id}")

        # Get AWS configuration from environment
        aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        bedrock_region = os.environ.get('BEDROCK_REGION', aws_region)

        # Get Weaviate configuration based on ENVIRONMENT
        environment = os.environ.get('ENVIRONMENT', 'staging')
        logger.info(f"Running in environment: {environment}")
        
        if environment == 'preprod':
            weaviate_url = os.environ.get('WEAVIATE_URL')
            weaviate_api_key = os.environ.get('WEAVIATE_API_KEY')
        else:
            weaviate_url = os.environ.get('WEAVIATE_URL_STAGING')
            weaviate_api_key = os.environ.get('WEAVIATE_API_KEY_STAGING')

        if not weaviate_url or not weaviate_api_key:
            raise RuntimeError(
                f"Weaviate configuration missing for environment '{environment}'. "
                f"Required variables: WEAVIATE_URL{'_STAGING' if environment != 'preprod' else ''} "
                f"and WEAVIATE_API_KEY{'_STAGING' if environment != 'preprod' else ''}"
            )

        # Initialize embedder
        embedder = BedrockCohereEmbedder(
            region=bedrock_region,
            model_id=embedding_model,
        )

        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        vectors = embedder.embed_texts(texts)

        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embeddings count mismatch: got {len(vectors)}, expected {len(chunks)}")

        logger.info(f"Generated {len(vectors)} embeddings")

        # Initialize Weaviate client
        weaviate_client = WeaviateClient(
            url=weaviate_url,
            api_key=weaviate_api_key,
        )

        try:
            # Upsert to Weaviate
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

            return {
                "chunks_written": chunks_written,
                "collection_name": collection_name,
                "doc_id": doc_id,
                "company_id": company_id,
                "area_id": area_id,
            }

        finally:
            weaviate_client.close()

    except Exception as e:
        logger.error(f"Error processing chunks: {str(e)}", exc_info=True)
        raise