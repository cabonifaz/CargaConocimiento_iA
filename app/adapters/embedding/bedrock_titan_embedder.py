from __future__ import annotations
import json
import time
from typing import List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

from app.ports.outbound.embedder import EmbedderPort
from app.config.settings import settings

class BedrockTitanEmbedder(EmbedderPort):
    """Embedder para Amazon Titan Text Embeddings v2 en Bedrock.
    - Usa AWS_PROFILE si está definido en .env
    - Hace batching cliente (N textos -> N llamadas) con backoff simple en throttling
    """

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        profile: Optional[str] = None,
        dim: Optional[int] = None,
        batch_size: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> None:
        self.region = region or settings.BEDROCK_REGION
        self.model_id = model_id or settings.BEDROCK_MODEL_ID
        self.profile = profile or settings.AWS_PROFILE
        self.dim = dim if dim is not None else settings.BEDROCK_EMBED_DIM
        self.batch_size = batch_size or settings.BEDROCK_BATCH_SIZE
        self.timeout_secs = timeout_secs or settings.BEDROCK_TIMEOUT_SECS

        # Session con perfil (si se especifica)
        if self.profile:
            session = boto3.Session(profile_name=self.profile, region_name=self.region)
        else:
            session = boto3.Session(region_name=self.region)

        cfg = Config(read_timeout=self.timeout_secs, retries={"max_attempts": 3, "mode": "standard"})
        self.client = session.client("bedrock-runtime", region_name=self.region, config=cfg)

    # --------------- API ---------------

    def embed_texts(self, texts: List[str]) -> List[list[float]]:
        if not texts:
            return []

        vectors: List[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Titan embeddings procesa un texto por invocación -> llamamos por cada item del batch
            for t in batch:
                vec = self._embed_one(t)
                vectors.append(vec)
        return vectors

    # --------------- Internos ---------------

    def _embed_one(self, text: str) -> list[float]:
        # Construcción del cuerpo según Titan v2.
        # Para v2, el campo principal es "inputText". La dimensión es opcional.
        body = {"inputText": text}
        if self.dim:
            # Algunas versiones aceptan "dimensions"; otras "embeddingConfig".
            # Usamos "dimensions" si dim está definido; si tu región requiere el otro nombre, cámbialo aquí.
            body["dimensions"] = int(self.dim)

        payload = json.dumps(body).encode("utf-8")

        # Backoff simple en throttling
        backoff = 1.0
        while True:
            try:
                resp = self.client.invoke_model(modelId=self.model_id, body=payload)
                # bedrock-runtime retorna bytes en resp["body"]
                raw = resp.get("body").read()
                data = json.loads(raw.decode("utf-8"))
                emb = data.get("embedding") or data.get("vector")  # por si la clave difiere
                if not isinstance(emb, list):
                    raise RuntimeError("Unexpected response: no 'embedding' vector found")
                return emb
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in {"ThrottlingException", "TooManyRequestsException"}:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue
                raise
            except BotoCoreError:
                # Reintento breve ante errores transitorios
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
