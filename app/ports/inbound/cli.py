import argparse
from pathlib import Path

from app.adapters.blob.local_fs import LocalFileSystemBlob
from app.adapters.extract.pymupdf_text_extractor import PyMuPDFTextExtractor
from app.adapters.tokenizer.simple_regex_tokenizer import SimpleRegexTokenizer
from app.adapters.embedding.bedrock_titan_embedder import BedrockTitanEmbedder
from app.adapters.vector_store.weaviate_store import WeaviateVectorStore

from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput
)
from app.application.use_cases.extract_and_normalize_pdf import (
    ExtractAndNormalizePdf, ExtractAndNormalizeInput
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
from app.application.use_cases.embed_and_upsert_pdf import (
    EmbedAndUpsertPdf, EmbedAndUpsertPdfInput
)
from app.application.use_cases.embed_and_upsert_company_pdfs import (
    EmbedAndUpsertCompanyPdfs, EmbedAndUpsertCompanyPdfsInput
)
from app.domain.services.md_text_normalizer import MdTextNormalizer
from app.ports.outbound.chunker import ChunkerConfig
from app.adapters.chunker.chunk_global import ChunkGlobalAdapter
from app.adapters.chunker.chunk_global_md import ChunkGlobalMdAdapter
from app.domain.services.chunk_quality import QualityConfig
from app.config.settings import settings


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
    ext.add_argument("--generate-report", action="store_true", help="Generar reporte de extracción")

    # --- normalize (debug puntual) ---
    norm = sub.add_parser("normalize", help="Normalizar texto extraído (debug)")
    norm.add_argument("relative_path", type=str, help="ruta al archivo de texto a normalizar")
    norm.add_argument("--generate-report", action="store_true", help="Generar reporte de normalización")

    # --- chunk-global (1 archivo) ---
    gchunk = sub.add_parser("chunk-global", help="Extraer, normalizar y chunkear GLOBAL un PDF")
    gchunk.add_argument("relative_path", type=str)
    gchunk.add_argument("--max-pages", type=int, default=None)
    gchunk.add_argument("--target", type=int, default=512, help="tokens por chunk")
    gchunk.add_argument("--overlap", type=int, default=64, help="tokens de solapamiento")
    gchunk.add_argument("--min-toks", type=int, default=50, help="umbral mínimo de tokens")
    gchunk.add_argument("--sep", type=str, default="\n\n\f\n\n", help="separador entre páginas en el texto unido")
    
    # --- chunk-global-md (1 archivo) ---
    gchunkmd = sub.add_parser("chunk-global-md", help="Extraer, normalizar y chunkear GLOBAL un PDF")
    gchunkmd.add_argument("relative_path", type=str)
    gchunkmd.add_argument("--max-pages", type=int, default=None)
    gchunkmd.add_argument("--target", type=int, default=512, help="tokens por chunk")
    gchunkmd.add_argument("--overlap", type=int, default=64, help="tokens de solapamiento")
    gchunkmd.add_argument("--min-toks", type=int, default=50, help="umbral mínimo de tokens")
    gchunkmd.add_argument("--sep", type=str, default="\n\n\f\n\n", help="separador entre páginas en el texto unido")
    gchunkmd.add_argument("--generate-report", action="store_true", help="Generar reporte de extracción y normalización")

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

    # --- embed-dry (hasta embeddings, sin upsert) ---
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

    # --- embed-weaviate (un PDF: embeddings + upsert BYOV) ---
    ew = sub.add_parser("embed-weaviate", help="Embed + upsert BYOV en Weaviate (un PDF)")
    ew.add_argument("relative_path", type=str)
    ew.add_argument("--max-pages", type=int, default=None)
    ew.add_argument("--target", type=int, default=512)
    ew.add_argument("--overlap", type=int, default=64)
    ew.add_argument("--min-toks", type=int, default=50)
    ew.add_argument("--sep", type=str, default="\n\n\f\n\n")
    ew.add_argument("--q-min-toks", type=int, default=50)
    ew.add_argument("--q-min-chars", type=int, default=200)
    ew.add_argument("--q-alpha-min", type=float, default=0.30)
    ew.add_argument("--q-uniq-min", type=float, default=0.10)
    ew.add_argument("--doc-id", type=str, default=None)
    ew.add_argument("--company-id", type=str, default="default_company", help="Company identifier")
    ew.add_argument("--generate-report", action="store_true", help="Generar reportes de extracción, normalización y chunking")

    # --- embed-weaviate-company (todos los PDFs de una carpeta de empresa) ---
    ewc = sub.add_parser("embed-weaviate-company", help="Embed + upsert todos los PDFs de una carpeta de empresa")
    ewc.add_argument("company_id", type=str, help="Nombre de la carpeta de la empresa (será usado como company_id y collection)")
    ewc.add_argument("--max-pages", type=int, default=None)
    ewc.add_argument("--target", type=int, default=512)
    ewc.add_argument("--overlap", type=int, default=64)
    ewc.add_argument("--min-toks", type=int, default=50)
    ewc.add_argument("--sep", type=str, default="\n\n\f\n\n")
    ewc.add_argument("--q-min-toks", type=int, default=50)
    ewc.add_argument("--q-min-chars", type=int, default=200)
    ewc.add_argument("--q-alpha-min", type=float, default=0.30)
    ewc.add_argument("--q-uniq-min", type=float, default=0.10)
    ewc.add_argument("--generate-report", action="store_true", help="Generar reportes de extracción, normalización y chunking")

    # --- bedrock-check (sanity de conexión) ---
    br = sub.add_parser("bedrock-check", help="Probar conexión a Bedrock Titan con un texto")
    br.add_argument("--text", type=str, required=True, help="Texto a embeddear")

    args = parser.parse_args()

    if args.cmd == "extract":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        uc = ExtractTextFromPdf(blob, extractor)

        out = uc.execute(ExtractTextInput(relative_path=Path(args.relative_path), max_pages=args.max_pages, generate_report=args.generate_report))
        print(f"Archivo: {out.source_path}")
        print(f"Páginas extraídas: {out.result.page_count}")
        for i, page in enumerate(out.result.pages, start=1):
            print(f"\n--- Página {i} ---\n{page[:50]}...")  # primeras 1000 chars para inspección

    elif args.cmd == "normalize":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        normalizer = MdTextNormalizer()
        uc = ExtractAndNormalizePdf(blob, extractor, normalizer)
        out = uc.execute(ExtractAndNormalizeInput(relative_path=Path(args.relative_path), generate_report=args.generate_report))
        print(f"Archivo: {out.source_path}  (páginas: {out.page_count})")
        print(f"Texto normalizado!")  # primeras 1000 chars para inspección

    elif args.cmd == "chunk-global":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        tokenizer = SimpleRegexTokenizer()
        chunker = ChunkGlobalAdapter(
            tokenizer,
            ChunkerConfig(
                target_tokens=args.target,
                overlap_tokens=args.overlap,
                min_tokens=args.min_toks,
                page_separator=args.sep,
            ),
        )
        normalizer = MdTextNormalizer()
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
            
    elif args.cmd == "chunk-global-md":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        tokenizer = SimpleRegexTokenizer()
        chunker = ChunkGlobalMdAdapter(
            tokenizer,
            ChunkerConfig(
                target_tokens=args.target,
                overlap_tokens=args.overlap,
                min_tokens=args.min_toks,
                page_separator=args.sep,
            ),
        )
        normalizer = MdTextNormalizer()
        uc = ExtractNormalizeChunkGlobalPdf(blob, extractor, normalizer, chunker)
        out = uc.execute(ExtractNormalizeChunkGlobalInput(
            relative_path=Path(args.relative_path),
            max_pages=args.max_pages,
            generate_report=args.generate_report
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
        normalizer = MdTextNormalizer()
        chunker = ChunkGlobalAdapter(
            tokenizer,
            ChunkerConfig(
                target_tokens=args.target,
                overlap_tokens=args.overlap,
                min_tokens=args.min_toks,
                page_separator=args.sep,
            ),
        )

        uc = ChunkGlobalAll(blob, extractor, normalizer, chunker)
        out = uc.execute(ChunkGlobalAllInput(
            max_pages=args.max_pages,
            recursive=not args.non_recursive,
            chunker_cfg=ChunkerConfig(
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
        normalizer = MdTextNormalizer()

        chunker = ChunkGlobalMdAdapter(
            tokenizer,
            ChunkerConfig(
                target_tokens=args.target,
                overlap_tokens=args.overlap,
                min_tokens=args.min_toks,
                page_separator=args.sep,
            ),
        )

        embedder = BedrockTitanEmbedder()  # usa env: AWS_PROFILE, BEDROCK_REGION, BEDROCK_MODEL_ID, etc.

        uc = EmbedChunksFromPdf(blob, extractor, normalizer, chunker, embedder)
        out = uc.execute(EmbedChunksFromPdfInput(
            relative_path=Path(args.relative_path),
            max_pages=args.max_pages,
            chunker_cfg=ChunkerConfig(
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

    elif args.cmd == "embed-weaviate":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        tokenizer = SimpleRegexTokenizer()
        normalizer = MdTextNormalizer()
        chunker = ChunkGlobalMdAdapter(
            tokenizer,
            ChunkerConfig(
                target_tokens=args.target,
                overlap_tokens=args.overlap,
                min_tokens=args.min_toks,
                page_separator=args.sep,
            ),
        )
        embedder = BedrockTitanEmbedder()
        store = WeaviateVectorStore()

        try:
            uc = EmbedAndUpsertPdf(blob, extractor, normalizer, chunker, embedder, store)
            out = uc.execute(EmbedAndUpsertPdfInput(
                relative_path=Path(args.relative_path),
                max_pages=args.max_pages,
                chunker_cfg=ChunkerConfig(
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
                doc_id=args.doc_id,
                company_id=getattr(args, 'company_id', 'default_company'),
                generate_report=args.generate_report
            ))

            print(f"Archivo: {out.source_path}")
            print(f"Páginas: {out.page_count} | Chunks aceptados: {out.used_text_count}")
            print(f"Weaviate upsert -> escritos: {out.written} | colección: {settings.WEAVIATE_COLLECTION} | doc_id: {out.doc_id} | company_id: {getattr(args, 'company_id', 'default_company')}")
        finally:
            store.close()

    elif args.cmd == "embed-weaviate-company":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        tokenizer = SimpleRegexTokenizer()
        normalizer = MdTextNormalizer()
        chunker = ChunkGlobalMdAdapter(
            tokenizer,
            ChunkerConfig(
                target_tokens=args.target,
                overlap_tokens=args.overlap,
                min_tokens=args.min_toks,
                page_separator=args.sep,
            ),
        )
        embedder = BedrockTitanEmbedder()
        store = WeaviateVectorStore()

        try:
            uc = EmbedAndUpsertCompanyPdfs(blob, extractor, normalizer, chunker, embedder, store)
            out = uc.execute(EmbedAndUpsertCompanyPdfsInput(
                company_id=args.company_id,
                max_pages=args.max_pages,
                chunker_cfg=ChunkerConfig(
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

            print(f"Empresa: {out.company_id}")
            print(f"Archivos procesados: {out.successful_files}/{out.total_files}")
            print(f"Total chunks escritos: {out.total_chunks_written}")
            print(f"Colección: {args.company_id}")
            
            for report in out.reports:
                status = "✅" if report.success else "❌"
                if report.success:
                    print(f"{status} {report.file_path.name} -> {report.chunks_written} chunks (doc_id: {report.doc_id})")
                else:
                    print(f"{status} {report.file_path.name} -> Error: {report.error_message}")
        finally:
            store.close()
            
    elif args.cmd == "bedrock-check":
        embedder = BedrockTitanEmbedder()
        vec = embedder.embed_texts([args.text])[0]
        print(f"Texto: {args.text}")
        print(f"Dimensión: {len(vec)}")
        print(f"Primeros 10 valores: {vec[:10]}")


if __name__ == "__main__":
    main()
