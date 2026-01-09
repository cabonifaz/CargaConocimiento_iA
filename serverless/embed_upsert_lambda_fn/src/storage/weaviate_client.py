"""Weaviate client for upserting chunks with vectors."""

import logging
from typing import List, Dict, Any, Optional
from uuid import uuid5, NAMESPACE_URL
from datetime import datetime

import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.exceptions import WeaviateBaseError

from .models import WeaviateChunkMetadata
from ..embeddings import BedrockBM25Generator

logger = logging.getLogger()


class WeaviateClient:
    """Client for managing Weaviate vector store operations."""

    def __init__(
        self,
        url: str,
        api_key: str,
        bedrock_region: str,
        distance: str = "cosine",
        batch_size: int = 100,
        bm25_model_id: str = "us.meta.llama4-maverick-17b-instruct-v1:0",
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.bedrock_region = bedrock_region
        self.distance = distance.lower()
        self.batch_size = batch_size
        self.bm25_model_id = bm25_model_id

        if not self.url or not self.api_key:
            raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY are required")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
            skip_init_checks=True,
            additional_config=AdditionalConfig(timeout=Timeout(init=30)),
        )

        # Initialize Bedrock BM25 generator
        self.bm25_generator = BedrockBM25Generator(
            region=self.bedrock_region,
            model_id=self.bm25_model_id,
        )

    def close(self) -> None:
        """Close Weaviate client connection."""
        try:
            self.client.close()
        except Exception as e:
            logger.warning(f"Error closing Weaviate client: {e}")

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        doc_id: str,
        company_id: str,
        area_id: str,
        doc_title: str,
        embedding_model: str,
        collection_name: str,
    ) -> int:
        """
        Upsert chunks with their vectors to Weaviate.

        Args:
            chunks: List of chunk dictionaries from norm_chunk_lambda_fn
            vectors: List of embedding vectors (aligned with chunks)
            doc_id: Document identifier
            company_id: Company identifier
            area_id: Area identifier
            doc_title: Document title
            embedding_model: Model used for embeddings
            collection_name: Weaviate collection name

        Returns:
            Number of chunks written
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        if not chunks:
            logger.info("No chunks to upsert")
            return 0

        dim = len(vectors[0])
        if any(len(v) != dim for v in vectors):
            raise ValueError("All vectors must have the same dimension")

        logger.info(f"Upserting {len(chunks)} chunks to collection '{collection_name}'")

        self._ensure_collection_exists(collection_name)
        coll = self.client.collections.get(collection_name)

        # Generate BM25 text for all chunks using Bedrock
        logger.info("Generating BM25 text using Bedrock Llama 4 Maverick model")
        chunk_texts = [chunk['text'] for chunk in chunks]
        bm25_texts = self.bm25_generator.generate_bm25_texts(chunk_texts)

        total = 0
        bs = self.batch_size

        for i in range(0, len(chunks), bs):
            batch_chunks = chunks[i:i+bs]
            batch_vecs = vectors[i:i+bs]
            batch_bm25 = bm25_texts[i:i+bs]

            objs: List[DataObject] = []
            id_map: List[tuple] = []

            for chunk, vec, bm25_text in zip(batch_chunks, batch_vecs, batch_bm25):
                # Extract section info from chunk metadata
                section_title = chunk.get('section_title', '')
                section_path = chunk.get('section_path', [])

                metadata = WeaviateChunkMetadata(
                    text=chunk['text'],
                    bm25_text=bm25_text,
                    doc_title=doc_title,
                    section_title=section_title,
                    doc_id=doc_id,
                    company_id=company_id,
                    company=company_id,  # Same as company_id
                    area_id=area_id,
                    area=area_id,  # Same as area_id
                    section_path=section_path,
                    page_start=chunk.get('page_start', 1),
                    page_end=chunk.get('page_end', 1),
                    embedding_model=embedding_model,
                    embedding_dim=len(vec),
                    chunk_id=chunk['chunk_id'],
                    token_count=chunk['token_count'],
                    char_start=chunk['char_start'],
                    char_end=chunk['char_end'],
                    ingested_at=datetime.now().isoformat(),
                )

                props = metadata.to_weaviate_properties()
                uid = str(uuid5(NAMESPACE_URL, f"{doc_id}:{chunk['chunk_id']}"))
                objs.append(DataObject(properties=props, vector=vec, uuid=uid))
                id_map.append((props, vec, uid))

            try:
                coll.data.insert_many(objs)
                total += len(objs)

            except WeaviateBaseError as e:
                logger.warning(f"Batch insert failed, falling back to individual inserts: {e}")
                for props, vec, uid in id_map:
                    try:
                        coll.data.insert(properties=props, uuid=uid, vector=vec)
                    except WeaviateBaseError:
                        coll.data.replace(uuid=uid, properties=props, vector=vec)
                    total += 1

        logger.info(f"Successfully upserted {total} chunks to Weaviate")
        return total

    def _ensure_collection_exists(self, collection_name: str) -> None:
        """Ensure collection exists, create if it doesn't."""
        try:
            self.client.collections.get(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
            return
        except Exception:
            logger.info(f"Creating collection '{collection_name}'")
            pass

        metric = self._metric_from_str(self.distance)
        # Note: Weaviate Cloud may show all fields as "filterable" in the UI by default,
        # but the explicit index_filterable settings below take precedence and optimize performance.
        # Text fields (text, bm25_text) should NOT be filterable for performance reasons.
        # Technical metadata (chunk_id, token_count, char_start, char_end, ingested_at) are
        # typically not used for filtering and are left non-filterable to improve query performance.
        self.client.collections.create(
            name=collection_name,
            description="Chunks de manuales (BYOV)",
            vector_config=Configure.Vectors.self_provided(
                name="default",
                vector_index_config=Configure.VectorIndex.hnsw(distance_metric=metric),
            ),
            inverted_index_config=Configure.inverted_index(
                bm25_k1=1.3,
                bm25_b=0.75,
            ),
            properties=[
                # Large text fields - searchable but not filterable for performance
                Property(name="text", data_type=DataType.TEXT,
                        index_searchable=True, index_filterable=False),
                Property(name="bm25_text", data_type=DataType.TEXT,
                        index_searchable=True, index_filterable=False),
                # Document and section identifiers - filterable for queries
                Property(name="doc_title", data_type=DataType.TEXT,
                        index_searchable=True, index_filterable=True),
                Property(name="section_title", data_type=DataType.TEXT,
                        index_searchable=True, index_filterable=True),
                Property(name="doc_id", data_type=DataType.TEXT,
                        tokenization="field", index_searchable=False, index_filterable=True),
                # Organization identifiers - filterable for multi-tenancy
                Property(name="company_id", data_type=DataType.TEXT,
                        tokenization="field", index_searchable=False, index_filterable=True),
                Property(name="company", data_type=DataType.TEXT,
                        tokenization="field", index_searchable=False, index_filterable=True),
                Property(name="area_id", data_type=DataType.TEXT,
                        tokenization="field", index_searchable=False, index_filterable=True),
                Property(name="area", data_type=DataType.TEXT,
                        tokenization="field", index_searchable=False, index_filterable=True),
                # Hierarchical and range filters
                Property(name="section_path", data_type=DataType.TEXT_ARRAY,
                        index_searchable=False, index_filterable=True),
                Property(name="page_start", data_type=DataType.INT, index_filterable=True),
                Property(name="page_end", data_type=DataType.INT, index_filterable=True),
                # Embedding metadata - filterable for debugging/analytics
                Property(name="embedding_model", data_type=DataType.TEXT,
                        tokenization="field", index_searchable=False, index_filterable=True),
                Property(name="embedding_dim", data_type=DataType.INT,
                        index_filterable=True),
                # Technical metadata - not typically filtered, kept non-filterable for performance
                Property(name="chunk_id", data_type=DataType.TEXT,
                        tokenization="field", index_searchable=False, index_filterable=False),
                Property(name="token_count", data_type=DataType.INT, index_filterable=False),
                Property(name="char_start", data_type=DataType.INT, index_filterable=False),
                Property(name="char_end", data_type=DataType.INT, index_filterable=False),
                Property(name="ingested_at", data_type=DataType.TEXT,
                        index_searchable=False, index_filterable=False)
            ],
        )
        logger.info(f"Collection '{collection_name}' created successfully")

    @staticmethod
    def _metric_from_str(s: str) -> VectorDistances:
        """Convert distance metric string to Weaviate VectorDistances enum."""
        s = (s or "cosine").lower()
        if s == "cosine":
            return VectorDistances.COSINE
        if s in ("dot", "dotproduct", "dot_product"):
            return VectorDistances.DOT
        if s in ("l2", "l2-squared", "euclidean"):
            return VectorDistances.L2_SQUARED
        return VectorDistances.COSINE
