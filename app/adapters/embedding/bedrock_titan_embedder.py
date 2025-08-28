from __future__ import annotations
import json
import time
from typing import List, Optional
from datetime import datetime
import os
from pathlib import Path

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
        
        # Per-PDF tracking
        self.current_pdf_start_time = None
        self.current_pdf_tokens = 0
        self.current_pdf_cost = 0.0
        self.current_pdf_chunks = 0
        
        # Setup logging
        self.logs_dir = Path("reports/embedding_costs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.logs_dir / f"embedding_costs_{current_date}.txt"
        
        # Initialize log file with header if it doesn't exist
        if not self.log_file.exists():
            self._write_log_header()

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
                chunk_start_time = time.time()
                
                # Calculate cost before embedding
                tokens = self._estimate_tokens(t)
                cost = self._calculate_cost(tokens)
                
                # Update running totals
                self.total_tokens += tokens
                self.total_cost += cost
                
                # Update per-PDF totals
                self.current_pdf_tokens += tokens
                self.current_pdf_cost += cost
                self.current_pdf_chunks += 1
                
                # Process embedding
                vec = self._embed_one(t)
                
                # Calculate processing time
                chunk_end_time = time.time()
                processing_time = chunk_end_time - chunk_start_time
                
                # Log cost per chunk with timing
                chunk_idx = i + idx  # Global chunk index
                self._log_chunk_cost(chunk_idx, t, tokens, cost, processing_time)
                
                vectors.append(vec)
        
        # Print and log final cost summary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Console output
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
        
        # File logging
        summary = f"""
{'=' * 80}
📊 EMBEDDING BATCH SUMMARY - {timestamp}
{'=' * 80}
Total Chunks: {len(texts)}
Total Tokens: {self.total_tokens:,}
Total Cost: ${self.total_cost:.8f}
Average Tokens per Chunk: {self.total_tokens / len(texts):.1f}
Average Cost per Chunk: ${self.total_cost / len(texts):.8f}
Model: {self.model_id}
Region: {self.region}
{'=' * 80}

"""
        self._log_to_file(summary)
        
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
    
    def _write_log_header(self):
        """Write header to log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
==========================================================================
EMBEDDING COST LOG - {timestamp}
Model: {self.model_id}
Region: {self.region}
Cost per 1M tokens: ${self.COST_PER_1M_TOKENS}
==========================================================================

"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def _log_to_file(self, message: str):
        """Append message to log file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def start_pdf_tracking(self, pdf_name: str, pdf_size_mb: float):
        """Start tracking metrics for a new PDF."""
        self.current_pdf_start_time = time.time()
        self.current_pdf_tokens = 0
        self.current_pdf_cost = 0.0
        self.current_pdf_chunks = 0
        
        # Log PDF start
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"\n📄 STARTING PDF: {pdf_name} ({pdf_size_mb:.1f} MB) - {timestamp}"
        print(msg)
        self._log_to_file(f"{msg}\n{'=' * 80}")
    
    def end_pdf_tracking(self, pdf_name: str, pdf_size_mb: float):
        """End tracking and log PDF summary."""
        if self.current_pdf_start_time is None:
            return
            
        end_time = time.time()
        processing_time = end_time - self.current_pdf_start_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Calculate rates
        mb_per_second = pdf_size_mb / processing_time if processing_time > 0 else 0
        chunks_per_second = self.current_pdf_chunks / processing_time if processing_time > 0 else 0
        tokens_per_second = self.current_pdf_tokens / processing_time if processing_time > 0 else 0
        
        # Console output
        print(f"\n📊 PDF COMPLETED: {pdf_name}")
        print(f"   File size: {pdf_size_mb:.1f} MB")
        print(f"   Processing time: {processing_time:.1f}s")
        print(f"   Chunks processed: {self.current_pdf_chunks}")
        print(f"   Tokens: {self.current_pdf_tokens:,}")
        print(f"   Cost: ${self.current_pdf_cost:.8f}")
        print(f"   Speed: {mb_per_second:.2f} MB/s, {chunks_per_second:.1f} chunks/s, {tokens_per_second:.0f} tokens/s")
        print("=" * 80)
        
        # File logging
        summary = f"""
📊 PDF SUMMARY: {pdf_name} - {timestamp}
================================================================================
File size: {pdf_size_mb:.1f} MB
Processing time: {processing_time:.1f}s
Chunks processed: {self.current_pdf_chunks}
Tokens: {self.current_pdf_tokens:,}
Cost: ${self.current_pdf_cost:.8f}
Processing speeds:
  - {mb_per_second:.2f} MB/s
  - {chunks_per_second:.1f} chunks/s  
  - {tokens_per_second:.0f} tokens/s
================================================================================

"""
        self._log_to_file(summary)
    
    def _log_chunk_cost(self, chunk_idx: int, text: str, tokens: int, cost: float, processing_time: float):
        """Log cost information for individual chunk with timing and size info."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        text_size_kb = len(text.encode('utf-8')) / 1024  # Size in KB
        chars_per_second = len(text) / processing_time if processing_time > 0 else 0
        tokens_per_second = tokens / processing_time if processing_time > 0 else 0
        
        # Console output
        console_msg = f"\n💰 CHUNK {chunk_idx + 1} COST & PERFORMANCE:"
        print(console_msg)
        print(f"   Tokens: {tokens:,}")
        print(f"   Cost: ${cost:.8f}")
        print(f"   Text: {len(text)} chars ({text_size_kb:.1f} KB)")
        print(f"   Time: {processing_time:.2f}s")
        print(f"   Speed: {chars_per_second:.0f} chars/s, {tokens_per_second:.0f} tokens/s")
        print(f"   Running total: {self.total_tokens:,} tokens, ${self.total_cost:.8f}")
        print("-" * 55)
        
        # File logging
        file_msg = f"""[{timestamp}] CHUNK {chunk_idx + 1}:
   Tokens: {tokens:,}
   Cost: ${cost:.8f}
   Text length: {len(text)} chars ({text_size_kb:.1f} KB)
   Processing time: {processing_time:.2f}s
   Speed: {chars_per_second:.0f} chars/s, {tokens_per_second:.0f} tokens/s
   Text preview: {text[:100]}{'...' if len(text) > 100 else ''}
   Running total: {self.total_tokens:,} tokens, ${self.total_cost:.8f}
{'-' * 80}"""
        
        self._log_to_file(file_msg)

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
