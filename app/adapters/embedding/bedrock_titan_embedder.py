from __future__ import annotations
import json
import time
from typing import List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError
import pathlib

from app.ports.outbound.embedder import EmbedderPort
from app.config.settings import settings

# Add tiktoken for token counting and cost calculation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available, cost calculation will use character estimation")

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
        
        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.COST_PER_1M_TOKENS = 0.02  # $0.02 per 1M tokens for Titan Embeddings V2

    # --------------- API ---------------

    def embed_texts(self, texts: List[str]) -> List[list[float]]:
        if not texts:
            print("-------- NO TEXTS RECEIVED --------")
            return []
        
        print(f"### Bedrock Titan Embedder - embed_texts -> List[list[float]]")

        vectors: List[list[float]] = []
        for i in range(0, len(texts), self.batch_size):            
            batch = texts[i : i + self.batch_size]

            print(f"Procesando batch de {len(batch)} textos (total {len(texts)})...")

            # Titan embeddings procesa un texto por invocación -> llamamos por cada item del batch
            for idx, t in enumerate(batch):
                # Calculate cost before embedding
                tokens = self._estimate_tokens(t)
                cost = self._calculate_cost(tokens)
                
                # Update running totals
                self.total_tokens += tokens
                self.total_cost += cost
                
                # Log cost per chunk
                chunk_idx = i + idx  # Global chunk index
                self._log_chunk_cost(chunk_idx, t, tokens, cost)
                
                vec = self._embed_one(t)
                print(f"\r\033[2K- Vector recibido ({len(vec)} dims). ", end='', flush=True)  # \033[2K limpia la línea
                vectors.append(vec)
                print(f"-> {len(vectors)}/{len(texts)} embeddings procesados.", end='', flush=True)  # \033[2K limpia la línea
        
        # Print final cost summary
        print(f"\n" + "=" * 60)
        print(f"📊 EMBEDDING COST SUMMARY")
        print(f"=" * 60)
        print(f"Total Chunks: {len(texts)}")
        print(f"Total Tokens: {self.total_tokens:,}")
        print(f"Total Cost: ${self.total_cost:.8f}")
        print(f"Average Tokens per Chunk: {self.total_tokens / len(texts):.1f}")
        print(f"Average Cost per Chunk: ${self.total_cost / len(texts):.8f}")
        print(f"Model: {self.model_id}")
        print(f"=" * 60)
        
        print("\n---")
        return vectors

    # --------------- Cost Calculation ---------------
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using tiktoken or fallback to character-based estimation."""
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base (GPT-4) as approximation for Titan embeddings
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as e:
                print(f"Warning: tiktoken failed ({e}), using character estimation")
        
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return max(1, len(text) // 4)
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost for given number of tokens."""
        return (tokens / 1_000_000) * self.COST_PER_1M_TOKENS
    
    def _log_chunk_cost(self, chunk_idx: int, text: str, tokens: int, cost: float):
        """Log cost information for individual chunk."""
        print(f"\n💰 CHUNK {chunk_idx + 1} COST:")
        print(f"   Tokens: {tokens:,}")
        print(f"   Cost: ${cost:.8f}")
        print(f"   Text length: {len(text)} chars")
        print(f"   Running total: {self.total_tokens:,} tokens, ${self.total_cost:.8f}")
        print("-" * 50)

    # --------------- Internos ---------------

    def _embed_one(self, text: str) -> list[float]:
        # Construcción del cuerpo según Titan v2.
        # Para v2, el campo principal es "inputText". La dimensión es opcional.
        body: dict[str, str | int] = {"inputText": text}
        if self.dim:
            # Algunas versiones aceptan "dimensions"; otras "embeddingConfig".
            # Usamos "dimensions" si dim está definido; si tu región requiere el otro nombre, cámbialo aquí.
            body["dimensions"] = int(self.dim)

        payload = json.dumps(body).encode("utf-8")

        # Backoff simple en throttling
        backoff = 1.0
        while True:
            try:
                print(f"\r\033[2K- Invocando modelo {self.model_id}...", end='', flush=True)  # \033[2K limpia la línea
                resp = self.client.invoke_model(modelId=self.model_id, body=payload)
                # bedrock-runtime retorna bytes en resp["body"]
                raw = resp.get("body").read()
                print(f"\r\033[2K- Respuesta recibida ({len(raw)} bytes). Procesando...", end='', flush=True)  # \033[2K limpia la línea
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
