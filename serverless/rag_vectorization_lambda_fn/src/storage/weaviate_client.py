"""Weaviate client for upserting document chunks with vectors."""

import logging
from datetime import datetime
from typing import Any, Dict, List
from uuid import NAMESPACE_URL, uuid5

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.init import AdditionalConfig, Auth, Timeout
from weaviate.exceptions import WeaviateBaseError

from .models import WeaviateChunkMetadata

logger = logging.getLogger()


class WeaviateClient:
    """
    Manages Weaviate collections and chunk upserts for the RAG pipeline.

    BM25 text is generated upstream (in lambda_function.py) and passed in
    explicitly — this class has no Bedrock dependency.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        distance: str = "cosine",
        batch_size: int = 100,
    ) -> None:
        if not url or not api_key:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY are required")

        self.distance = distance.lower()
        self.batch_size = batch_size

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
            skip_init_checks=True,
            additional_config=AdditionalConfig(timeout=Timeout(init=30)),
        )

    def close(self) -> None:
        """Close the Weaviate connection."""
        try:
            self.client.close()
        except Exception as e:
            logger.warning(f"Error closing Weaviate client: {e}")

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        bm25_texts: List[str],
        doc_id: str,
        company_id: str,
        area_id: str,
        doc_title: str,
        embedding_model: str,
        collection_name: str,
        id_documento: int = 0,
        id_proceso: int = 0,
    ) -> int:
        """
        Upsert chunks with their embedding vectors into a Weaviate collection.

        Args:
            chunks:           Chunk dicts from the chunking Lambda.
            vectors:          Embedding vectors aligned with `chunks`.
            bm25_texts:       BM25 keyword texts aligned with `chunks` (non-empty guaranteed).
            doc_id:           Document identifier used as UUID namespace ("CONOC-{id_documento}").
            company_id:       String company ID — also the collection name.
            area_id:          String area ID.
            doc_title:        Document title (filename without extension).
            embedding_model:  Bedrock model ID used to generate vectors.
            collection_name:  Weaviate collection name (= company_id).
            id_documento:     RAG_INGESTA_DOCUMENTOS.ID_DOCUMENTO (for traceability).
            id_proceso:       RAG_INGESTA_PROCESOS.ID_PROCESO (for traceability).

        Returns:
            Number of chunks written.
        """
        if len(chunks) != len(vectors) or len(chunks) != len(bm25_texts):
            raise ValueError(
                f"Length mismatch: chunks={len(chunks)}, "
                f"vectors={len(vectors)}, bm25_texts={len(bm25_texts)}"
            )

        if not chunks:
            logger.info("No chunks to upsert")
            return 0

        dim = len(vectors[0])

        logger.info(
            f"Upserting {len(chunks)} chunk(s) to collection '{collection_name}' "
            f"(doc_id={doc_id}, dim={dim})"
        )

        self._ensure_collection_exists(collection_name)
        coll = self.client.collections.get(collection_name)

        total = 0
        now = datetime.now().isoformat()

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_vecs = vectors[i:i + self.batch_size]
            batch_bm25 = bm25_texts[i:i + self.batch_size]

            objs: List[DataObject] = []
            id_map: List[tuple] = []

            for chunk, vec, bm25 in zip(batch_chunks, batch_vecs, batch_bm25):
                metadata = WeaviateChunkMetadata(
                    text=chunk["text"],
                    bm25_text=bm25,
                    doc_title=doc_title,
                    section_title=chunk.get("section_title", ""),
                    doc_id=doc_id,
                    company_id=company_id,
                    company=company_id,
                    area_id=area_id,
                    area=area_id,
                    section_path=chunk.get("section_path", []),
                    page_start=chunk.get("page_start", 1),
                    page_end=chunk.get("page_end", 1),
                    embedding_model=embedding_model,
                    embedding_dim=len(vec),
                    chunk_id=chunk["chunk_id"],
                    token_count=chunk["token_count"],
                    char_start=chunk["char_start"],
                    char_end=chunk["char_end"],
                    ingested_at=now,
                    id_documento=id_documento,
                    id_proceso=id_proceso,
                    id_status=1,
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

        logger.info(f"Upserted {total} chunk(s) to collection '{collection_name}'")
        return total

    def _ensure_collection_exists(self, collection_name: str) -> None:
        """Create the collection if it does not already exist."""
        try:
            self.client.collections.get(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
            return
        except Exception:
            logger.info(f"Creating collection '{collection_name}'")

        metric = self._metric_from_str(self.distance)

        self.client.collections.create(
            name=collection_name,
            description="RAG document chunks (BYOV)",
            vector_config=Configure.Vectors.self_provided(
                name="default",
                vector_index_config=Configure.VectorIndex.hnsw(distance_metric=metric),
            ),
            inverted_index_config=Configure.inverted_index(
                bm25_k1=1.3,
                bm25_b=0.75,
            ),
            properties=[
                # ── Text content ───────────────────────────────────────────
                Property(name="text", data_type=DataType.TEXT,
                         index_searchable=True, index_filterable=False),
                Property(name="bm25_text", data_type=DataType.TEXT,
                         index_searchable=True, index_filterable=False),
                # ── Document / section identifiers ─────────────────────────
                Property(name="doc_title", data_type=DataType.TEXT,
                         index_searchable=True, index_filterable=True),
                Property(name="section_title", data_type=DataType.TEXT,
                         index_searchable=True, index_filterable=True),
                Property(name="doc_id", data_type=DataType.TEXT,
                         tokenization="field", index_searchable=False, index_filterable=True),
                # ── Organisation ───────────────────────────────────────────
                Property(name="company_id", data_type=DataType.TEXT,
                         tokenization="field", index_searchable=False, index_filterable=True),
                Property(name="company", data_type=DataType.TEXT,
                         tokenization="field", index_searchable=False, index_filterable=True),
                Property(name="area_id", data_type=DataType.TEXT,
                         tokenization="field", index_searchable=False, index_filterable=True),
                Property(name="area", data_type=DataType.TEXT,
                         tokenization="field", index_searchable=False, index_filterable=True),
                # ── Hierarchy / position ───────────────────────────────────
                Property(name="section_path", data_type=DataType.TEXT_ARRAY,
                         index_searchable=False, index_filterable=True),
                Property(name="page_start", data_type=DataType.INT, index_filterable=True),
                Property(name="page_end", data_type=DataType.INT, index_filterable=True),
                # ── Embedding metadata ─────────────────────────────────────
                Property(name="embedding_model", data_type=DataType.TEXT,
                         tokenization="field", index_searchable=False, index_filterable=True),
                Property(name="embedding_dim", data_type=DataType.INT, index_filterable=True),
                # ── Technical metadata (not filtered) ──────────────────────
                Property(name="chunk_id", data_type=DataType.TEXT,
                         tokenization="field", index_searchable=False, index_filterable=False),
                Property(name="token_count", data_type=DataType.INT, index_filterable=False),
                Property(name="char_start", data_type=DataType.INT, index_filterable=False),
                Property(name="char_end", data_type=DataType.INT, index_filterable=False),
                Property(name="ingested_at", data_type=DataType.TEXT,
                         index_searchable=False, index_filterable=False),
                # ── Process traceability ───────────────────────────────────
                Property(name="id_documento", data_type=DataType.INT, index_filterable=True),
                Property(name="id_proceso", data_type=DataType.INT, index_filterable=True),
                # ── Soft delete ────────────────────────────────────────────
                Property(name="id_status", data_type=DataType.INT, index_filterable=True),
            ],
        )
        logger.info(f"Collection '{collection_name}' created successfully")

    @staticmethod
    def _metric_from_str(s: str) -> VectorDistances:
        s = (s or "cosine").lower()
        if s == "cosine":
            return VectorDistances.COSINE
        if s in ("dot", "dotproduct", "dot_product"):
            return VectorDistances.DOT
        if s in ("l2", "l2-squared", "euclidean"):
            return VectorDistances.L2_SQUARED
        return VectorDistances.COSINE
