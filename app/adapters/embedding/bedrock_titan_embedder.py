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
        
        # Chunk-by-chunk data accumulation for final table
        self.chunks_data = []
        
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
                
                # Accumulate chunk data for final table
                chunk_idx = i + idx  # Global chunk index
                self._accumulate_chunk_data(chunk_idx, t, tokens, cost, processing_time)
                
                vectors.append(vec)
        
        # Print consolidated chunk table and summary
        self._print_chunks_table()
        
        # Print final cost summary
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
        
        # Solo log a archivo, no a consola (ya se imprime en el nivel superior)
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"\n📄 STARTING PDF: {pdf_name} ({pdf_size_mb:.1f} MB) - {timestamp}"
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
        
        # Solo resumen compacto en consola
        print(f"⚡ [EMBEDDING] → Completado en {processing_time:.1f}s ({self.current_pdf_chunks} chunks, {tokens_per_second:.0f} tok/s)")
        
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
    
    def _accumulate_chunk_data(self, chunk_idx: int, text: str, tokens: int, cost: float, processing_time: float):
        """Accumulate chunk data for final consolidated table."""
        text_size_kb = len(text.encode('utf-8')) / 1024  # Size in KB
        chars_per_second = len(text) / processing_time if processing_time > 0 else 0
        tokens_per_second = tokens / processing_time if processing_time > 0 else 0
        
        chunk_data = {
            'chunk_id': chunk_idx + 1,
            'tokens': tokens,
            'cost': cost,
            'text_preview': text[:50] + '...' if len(text) > 50 else text,
            'text_length': len(text),
            'text_size_kb': text_size_kb,
            'processing_time': processing_time,
            'chars_per_second': chars_per_second,
            'tokens_per_second': tokens_per_second
        }
        
        self.chunks_data.append(chunk_data)
        
        # Progress indicator compacto para muchos chunks
        if chunk_idx % 10 == 0 or chunk_idx < 5:  # Solo mostrar cada 10 chunks o los primeros 5
            print(f"✓ Chunk {chunk_idx + 1} processed: {tokens:,} tokens, ${cost:.8f}, {processing_time:.2f}s")
        elif (chunk_idx + 1) % 50 == 0:  # Resumen cada 50
            print(f"✓ Processed {chunk_idx + 1} chunks...")
    
    def _print_chunks_table(self):
        """Print consolidated table with all chunks data."""
        if not self.chunks_data:
            return
            
        print(f"\n" + "=" * 100)
        print(f"📋 CHUNKS COST & PERFORMANCE TABLE")
        print(f"=" * 100)
        
        # Table header with better spacing
        header = f"{'ID':>3} │ {'Tokens':>7} │ {'Cost ($)':>10} │ {'Text Details':<20} │ {'Time':>6} │ {'Speed':>8}"
        separator = f"{'─':─>3}─┼─{'─':─>7}─┼─{'─':─>10}─┼─{'─':─<20}─┼─{'─':─>6}─┼─{'─':─>8}─"
        
        print(header)
        print(separator)
        
        # Table rows with better formatting
        for chunk in self.chunks_data:
            # Format text details with chars and KB info
            text_details = f"{chunk['text_length']:,} chars ({chunk['text_size_kb']:.1f} KB)"
            
            # Format cost with fewer decimals for readability
            cost_str = f"${chunk['cost']:.6f}" if chunk['cost'] >= 0.000001 else f"${chunk['cost']:.8f}"
            
            # Format speed
            speed_str = f"{chunk['tokens_per_second']:.0f} t/s"
            
            row = (
                f"{chunk['chunk_id']:>3} │ "
                f"{chunk['tokens']:>7,} │ "
                f"{cost_str:>10} │ "
                f"{text_details:<20} │ "
                f"{chunk['processing_time']:>6.2f} │ "
                f"{speed_str:>8}"
            )
            print(row)
        
        print("=" * 100)
        
        # Calculate and show totals row
        total_tokens = sum(chunk['tokens'] for chunk in self.chunks_data)
        total_cost = sum(chunk['cost'] for chunk in self.chunks_data)
        total_chars = sum(chunk['text_length'] for chunk in self.chunks_data)
        total_kb = sum(chunk['text_size_kb'] for chunk in self.chunks_data)
        avg_time = sum(chunk['processing_time'] for chunk in self.chunks_data) / len(self.chunks_data)
        avg_speed = sum(chunk['tokens_per_second'] for chunk in self.chunks_data) / len(self.chunks_data)
        
        total_cost_str = f"${total_cost:.6f}" if total_cost >= 0.000001 else f"${total_cost:.8f}"
        total_text_details = f"{total_chars:,} chars ({total_kb:.1f} KB)"
        
        totals_row = (
            f"{'TOT':>3} │ "
            f"{total_tokens:>7,} │ "
            f"{total_cost_str:>10} │ "
            f"{total_text_details:<20} │ "
            f"{avg_time:>6.2f} │ "
            f"{avg_speed:.0f} t/s"
        )
        print(totals_row)
        print("=" * 100)
        
        # Log to file with same improved format
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        table_log = f"\n{'=' * 100}\n📋 CHUNKS TABLE - {timestamp}\n{'=' * 100}\n"
        table_log += header + "\n" + separator + "\n"
        
        for chunk in self.chunks_data:
            text_details = f"{chunk['text_length']:,} chars ({chunk['text_size_kb']:.1f} KB)"
            
            cost_str = f"${chunk['cost']:.6f}" if chunk['cost'] >= 0.000001 else f"${chunk['cost']:.8f}"
            speed_str = f"{chunk['tokens_per_second']:.0f} t/s"
            
            row = (
                f"{chunk['chunk_id']:>3} │ "
                f"{chunk['tokens']:>7,} │ "
                f"{cost_str:>10} │ "
                f"{text_details:<20} │ "
                f"{chunk['processing_time']:>6.2f} │ "
                f"{speed_str:>8}"
            )
            table_log += row + "\n"
        
        table_log += "=" * 100 + "\n"
        table_log += totals_row + "\n"
        table_log += "=" * 100 + "\n\n"
        self._log_to_file(table_log)

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
