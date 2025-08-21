import argparse
from pathlib import Path

from app.adapters.blob.local_fs import LocalFileSystemBlob
from app.adapters.extract.pymupdf_text_extractor import PyMuPDFTextExtractor
from app.adapters.tokenizer.simple_regex_tokenizer import SimpleRegexTokenizer
from app.adapters.embedding.bedrock_titan_embedder import BedrockTitanEmbedder

from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput
)
from app.application.use_cases.extract_normalize_chunk_global_pdf import (
    ExtractNormalizeChunkGlobalPdf, ExtractNormalizeChunkGlobalInput
)
from app.application.use_cases.chunk_global_all import (
    ChunkGlobalAll, ChunkGlobalAllInput
)
from app.application.use_cases.embed_chunks_from_pdf import (
    EmbedChunksFromPdf, EmbedChunksFromPdfInput
)

from app.domain.services.text_normalizer import TextNormalizer
from app.domain.services.chunker_global import GlobalTokenChunker, GlobalChunkerConfig
from app.domain.services.chunk_quality import QualityConfig



def main():
    parser = argparse.ArgumentParser(description="Herramientas RAG (piloto)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- extract (debug puntual) ---
    ext = sub.add_parser("extract", help="Extraer texto de un PDF bajo company_files/")
    ext.add_argument(
        "relative_path",
        type=str,
        help=("ruta del PDF: puede ser solo el nombre (dentro de company_files), "
              "o incluir el prefijo 'company_files/', o una ruta absoluta dentro de esa carpeta")
    )
    ext.add_argument("--max-pages", type=int, default=None)

    # --- chunk-global (1 archivo) ---
    gchunk = sub.add_parser("chunk-global", help="Extraer, normalizar y chunkear GLOBAL un PDF")
    gchunk.add_argument("relative_path", type=str)
    gchunk.add_argument("--max-pages", type=int, default=None)
    gchunk.add_argument("--target", type=int, default=512, help="tokens por chunk")
    gchunk.add_argument("--overlap", type=int, default=64, help="tokens de solapamiento")
    gchunk.add_argument("--min-toks", type=int, default=50, help="umbral mínimo de tokens")
    gchunk.add_argument("--sep", type=str, default="\n\n\f\n\n", help="separador entre páginas en el texto unido")

    # --- chunk-global-all (todos los PDFs + validación + reporte) ---
    gall = sub.add_parser("chunk-global-all", help="Chunking global + validación para TODOS los PDFs")
    gall.add_argument("--max-pages", type=int, default=None)
    gall.add_argument("--non-recursive", action="store_true")
    # params de chunker
    gall.add_argument("--target", type=int, default=512)
    gall.add_argument("--overlap", type=int, default=64)
    gall.add_argument("--min-toks", type=int, default=50)
    gall.add_argument("--sep", type=str, default="\n\n\f\n\n")
    # quality gates
    gall.add_argument("--q-min-toks", type=int, default=50)
    gall.add_argument("--q-min-chars", type=int, default=200)
    gall.add_argument("--q-alpha-min", type=float, default=0.30)
    gall.add_argument("--q-uniq-min", type=float, default=0.10)
    gall.add_argument("--no-report", action="store_true", help="no generar archivo .jsonl")
    gall.add_argument("--dry-run", action="store_true", help="solo validar, no embebe ni sube (por ahora)")
    # embedding
    embed = sub.add_parser("embed-dry", help="Pipeline hasta embeddings (sin upsert)")
    embed.add_argument("relative_path", type=str)
    embed.add_argument("--max-pages", type=int, default=None)
    embed.add_argument("--target", type=int, default=512)
    embed.add_argument("--overlap", type=int, default=64)
    embed.add_argument("--min-toks", type=int, default=50)
    embed.add_argument("--sep", type=str, default="\n\n\f\n\n")
    embed.add_argument("--q-min-toks", type=int, default=50)
    embed.add_argument("--q-min-chars", type=int, default=200)
    embed.add_argument("--q-alpha-min", type=float, default=0.30)
    embed.add_argument("--q-uniq-min", type=float, default=0.10)

    args = parser.parse_args()

    if args.cmd == "extract":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        uc = ExtractTextFromPdf(blob, extractor)

        out = uc.execute(ExtractTextInput(relative_path=Path(args.relative_path), max_pages=args.max_pages))
        print(f"Archivo: {out.source_path}")
        print(f"Páginas extraídas: {out.result.page_count}")
        for i, page in enumerate(out.result.pages, start=1):
            print(f"\n--- Página {i} ---\n{page[:1000]}")  # primeras 1000 chars para inspección

    elif args.cmd == "chunk-global":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        tokenizer = SimpleRegexTokenizer()
        chunker = GlobalTokenChunker(tokenizer, GlobalChunkerConfig(
            target_tokens=args.target,
            overlap_tokens=args.overlap,
            min_tokens=args.min_toks,
            page_separator=args.sep,
        ))
        normalizer = TextNormalizer()
        uc = ExtractNormalizeChunkGlobalPdf(blob, extractor, normalizer, chunker)
        out = uc.execute(ExtractNormalizeChunkGlobalInput(
            relative_path=Path(args.relative_path),
            max_pages=args.max_pages,
        ))
        print(f"Archivo: {out.source_path}  (páginas: {out.page_count})")
        print(f"Chunks generados: {len(out.chunks)}")
        for i, c in enumerate(out.chunks, start=1):
            preview = c.text[:160].replace("\n", " ")
            print(f"#{i:03d} pages={c.page_start}-{c.page_end} toks={c.token_count} "
                  f"chars={c.char_start}-{c.char_end} id={c.chunk_id[:12]}  {preview}...")

    elif args.cmd == "chunk-global-all":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        tokenizer = SimpleRegexTokenizer()
        normalizer = TextNormalizer()
        chunker = GlobalTokenChunker(tokenizer, GlobalChunkerConfig(
            target_tokens=args.target,
            overlap_tokens=args.overlap,
            min_tokens=args.min_toks,
            page_separator=args.sep,
        ))

        uc = ChunkGlobalAll(blob, extractor, normalizer, chunker)
        out = uc.execute(ChunkGlobalAllInput(
            max_pages=args.max_pages,
            recursive=not args.non_recursive,
            chunker_cfg=GlobalChunkerConfig(
                target_tokens=args.target,
                overlap_tokens=args.overlap,
                min_tokens=args.min_toks,
                page_separator=args.sep,
            ),
            quality_cfg=QualityConfig(
                min_tokens=args.q_min_toks,
                min_chars=args.q_min_chars,
                min_alpha_ratio=args.q_alpha_min,
                min_unique_ratio=args.q_uniq_min,
            ),
            dry_run=args.dry_run or True,          # por ahora siempre dry-run
            report_jsonl=not args.no_report,
        ))

        print(f"Archivos procesados: {len(out.reports)}")
        for r in out.reports[:10]:  # muestra los primeros 10
            print(f"- {r.file_path.name}: pages={r.pages} total={r.chunks_total} ok={r.chunks_accepted} "
                  f"avgTok={r.avg_tokens:.1f} minTok={r.min_tokens} maxTok={r.max_tokens} "
                  f"warns={';'.join(r.warnings) or '—'}")
        if out.report_path:
            print(f"Reporte JSONL: {out.report_path}")

    elif args.cmd == "embed-dry":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        tokenizer = SimpleRegexTokenizer()
        normalizer = TextNormalizer()

        # chunker global
        from app.domain.services.chunker_global import GlobalTokenChunker, GlobalChunkerConfig
        chunker = GlobalTokenChunker(tokenizer, GlobalChunkerConfig(
            target_tokens=args.target,
            overlap_tokens=args.overlap,
            min_tokens=args.min_toks,
            page_separator=args.sep,
        ))

        # embedder bedrock
        embedder = BedrockTitanEmbedder()  # usa env: AWS_PROFILE, BEDROCK_REGION, BEDROCK_MODEL_ID, etc.

        uc = EmbedChunksFromPdf(blob, extractor, normalizer, chunker, embedder)
        out = uc.execute(EmbedChunksFromPdfInput(
            relative_path=Path(args.relative_path),
            max_pages=args.max_pages,
            chunker_cfg=GlobalChunkerConfig(
                target_tokens=args.target,
                overlap_tokens=args.overlap,
                min_tokens=args.min_toks,
                page_separator=args.sep,
            ),
            quality_cfg=QualityConfig(
                min_tokens=args.q_min_toks,
                min_chars=args.q_min_chars,
                min_alpha_ratio=args.q_alpha_min,
                min_unique_ratio=args.q_uniq_min,
            ),
        ))

        print(f"Archivo: {out.source_path} (páginas: {out.page_count})")
        print(f"Chunks aceptados: {out.used_text_count}")
        if out.vectors:
            dim = len(out.vectors[0])
            print(f"Embeddings recibidos: {len(out.vectors)}  |  Dimensión: {dim}")
        else:
            print("No se generaron embeddings (0 textos después del filtro).")

if __name__ == "__main__":
    main()
