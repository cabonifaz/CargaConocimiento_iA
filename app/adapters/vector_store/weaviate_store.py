from typing import List, Optional
from uuid import uuid5, NAMESPACE_URL
from typing import Sequence

from app.ports.outbound.chunker import ChunkType
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.exceptions import WeaviateBaseError

from app.ports.outbound.vector_store import VectorStorePort
from app.config.settings import settings

class WeaviateVectorStore(VectorStorePort):
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: Optional[str] = None,
        distance: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.url = url or settings.WEAVIATE_URL
        self.api_key = api_key or settings.WEAVIATE_API_KEY
        self.collection_name = collection or settings.WEAVIATE_COLLECTION
        self.distance = (distance or settings.WEAVIATE_DISTANCE).lower()
        self.batch_size = batch_size or settings.WEAVIATE_BATCH_SIZE

        if not self.url or not self.api_key:
            raise RuntimeError("WEAVIATE_URL / WEAVIATE_API_KEY no configurados.")

        # 🔧 REST-only + skip init checks para evitar el fallo gRPC
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
            skip_init_checks=True,
            additional_config=AdditionalConfig(timeout=Timeout(init=30)),
        )
        self._ensure_collection()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def upsert_chunks(self, doc_id: str, chunks: Sequence[ChunkType], vectors: List[list[float]], company_id: str = "default_company", area: str = "VENTAS", collection_name: Optional[str] = None) -> int:
        print(f"🗃️ [UPSERT] Iniciando upsert a Weaviate collection '{collection_name or self.collection_name}'")

        if len(chunks) != len(vectors):
            raise ValueError("chunks y vectors deben tener la misma longitud.")
        if not chunks:
            print(f"🗃️ [UPSERT] → Sin datos para upsert (0 chunks)")
            return 0

        dim = len(vectors[0])
        if any(len(v) != dim for v in vectors):
            raise ValueError("Todos los vectores deben tener la misma dimensión.")

        # Use the provided collection name or fall back to default
        target_collection = collection_name or self.collection_name
        self._ensure_collection_exists(target_collection)


        coll = self.client.collections.get(target_collection)

        total = 0
        bs = self.batch_size


        for i in range(0, len(chunks), bs):
            batch_chunks = chunks[i:i+bs]
            batch_vecs = vectors[i:i+bs]

            objs: list = []
            id_map: list[tuple[dict, list[float], str]] = []

            for c, vec in zip(batch_chunks, batch_vecs):
                props = {
                    "company_id": company_id,
                    "area": area,
                    "doc_id": doc_id,
                    "chunk_id": c.chunk_id,   # tu hash/64hex queda como propiedad (no como uuid)
                    "text": c.text,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "token_count": c.token_count,
                }
                # UUID RFC-4122 determinístico a partir de doc_id + chunk_id
                uid = str(uuid5(NAMESPACE_URL, f"{doc_id}:{c.chunk_id}"))
                objs.append(DataObject(properties=props, vector=vec, uuid=uid))
                id_map.append((props, vec, uid))

            try:
                # intento rápido en batch
                coll.data.insert_many(objs)

                total += len(objs)
            except WeaviateBaseError:
                # fallback: upsert por ítem (insert → replace si ya existe)
                for props, vec, uid in id_map:
                    try:
                        coll.data.insert(properties=props, uuid=uid, vector=vec)
                    except WeaviateBaseError:
                        # si ya existe u otro conflicto, hacemos replace (sobrescribe todo)
                        coll.data.replace(uuid=uid, properties=props, vector=vec)
                    total += 1
        print(f"🗃️ [UPSERT] → {total} chunks cargados exitosamente")
        return total


    def _ensure_collection(self) -> None:
        self._ensure_collection_exists(self.collection_name)

    def _ensure_collection_exists(self, collection_name: str) -> None:
        try:
            self.client.collections.get(collection_name)
            return
        except Exception:
            pass

        metric = self._metric_from_str(self.distance)
        self.client.collections.create(
            name=collection_name,
            description="Chunks de manuales (BYOV)",
            vector_config=Configure.Vectors.self_provided(
                name="default",
                vector_index_config=Configure.VectorIndex.hnsw(distance_metric=metric),
            ),
            properties=[
                # Metadata fields - optimized for filtering and search
                Property(name="company_id", data_type=DataType.TEXT, index_searchable=True),
                Property(name="area", data_type=DataType.TEXT, index_searchable=True),
                Property(name="doc_id", data_type=DataType.TEXT, index_searchable=True),
                Property(name="chunk_id", data_type=DataType.TEXT, index_searchable=False),  # Used for dedup, not search

                # Main content - fully indexed for vector + BM25 hybrid search
                Property(name="text", data_type=DataType.TEXT, index_searchable=True),

                # Numeric metadata - not indexed for text search but available for filtering
                Property(name="page_start", data_type=DataType.INT),
                Property(name="page_end", data_type=DataType.INT),
                Property(name="char_start", data_type=DataType.INT),
                Property(name="char_end", data_type=DataType.INT),
                Property(name="token_count", data_type=DataType.INT),
            ],
        )

    @staticmethod
    def _metric_from_str(s: str):
        s = (s or "cosine").lower()
        if s == "cosine":
            return VectorDistances.COSINE
        if s in ("dot", "dotproduct", "dot_product"):
            return VectorDistances.DOT
        if s in ("l2", "l2-squared", "euclidean"):
            return VectorDistances.L2_SQUARED
        return VectorDistances.COSINE
