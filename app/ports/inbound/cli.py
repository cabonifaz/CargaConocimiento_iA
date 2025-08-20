import argparse
from pathlib import Path

from app.adapters.blob.local_fs import LocalFileSystemBlob
from app.adapters.extract.pymupdf_text_extractor import PyMuPDFTextExtractor
from app.adapters.tokenizer.simple_regex_tokenizer import SimpleRegexTokenizer

from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput
)
from app.application.use_cases.extract_text_from_all_pdfs import (
    ExtractAllPdfs, ExtractAllPdfsInput
)
from app.application.use_cases.extract_and_normalize_pdf import (
    ExtractAndNormalizePdf, ExtractAndNormalizeInput
)
from app.application.use_cases.extract_and_normalize_all_pdfs import (
    ExtractAndNormalizeAllPdfs, ExtractAndNormalizeAllInput
)
from app.application.use_cases.extract_normalize_chunk_pdf import (
    ExtractNormalizeChunkPdf, ExtractNormalizeChunkInput
)
from app.domain.services.chunker import TokenChunker, ChunkerConfig
from app.domain.services.text_normalizer import NormalizerConfig, TextNormalizer

def main():
    parser = argparse.ArgumentParser(description="Herramientas RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Extraer texto de un PDF
    ext = sub.add_parser("extract", help="Extraer texto de un PDF bajo company_files/")
    ext.add_argument(
        "relative_path", 
        type=str, 
        help="ruta del PDF: puede ser solo el nombre (dentro de company_files), "
            "o incluir el prefijo 'company_files/', o una ruta absoluta dentro de esa carpeta"
    )
    ext.add_argument("--max-pages", type=int, default=None)

    # Extraer texto de todos los PDFs
    allp = sub.add_parser("extract-all", help="Extraer texto de todos los PDFs bajo company_files/")
    allp.add_argument("--max-pages", type=int, default=None)
    allp.add_argument("--non-recursive", action="store_true")

    # Normalizar texto de un PDF
    norm = sub.add_parser("normalize", help="Extraer y normalizar un PDF bajo company_files/")
    norm.add_argument("relative_path", type=str, help="ruta del PDF (ver ayuda extract)")
    norm.add_argument("--max-pages", type=int, default=None)
    norm.add_argument("--join", action="store_true", help="unir todas las páginas en un único texto")
    norm.add_argument("--no-fix-hyphens", action="store_true")
    norm.add_argument("--no-join-soft-breaks", action="store_true")

    # Normalizar todos los PDFs
    normall = sub.add_parser("normalize-all", help="Extraer y normalizar todos los PDFs")
    normall.add_argument("--max-pages", type=int, default=None)
    normall.add_argument("--non-recursive", action="store_true")
    normall.add_argument("--join", action="store_true")

    # Chunking
    chunk = sub.add_parser("chunk", help="Extraer, normalizar y chunkear un PDF")
    chunk.add_argument("relative_path", type=str)
    chunk.add_argument("--max-pages", type=int, default=None)
    chunk.add_argument("--target", type=int, default=512, help="tokens por chunk")
    chunk.add_argument("--overlap", type=int, default=64, help="tokens de solapamiento")
    chunk.add_argument("--min-toks", type=int, default=50, help="umbral mínimo de tokens por chunk")


    args = parser.parse_args()

    if args.cmd == "extract":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        uc = ExtractTextFromPdf(blob, extractor)

        out = uc.execute(ExtractTextInput(relative_path=Path(args.relative_path), max_pages=args.max_pages))
        print(f"Archivo: {out.source_path}")
        print(f"Páginas extraídas: {out.result.page_count}")
        for i, page in enumerate(out.result.pages, start=1):
            print(f"\n--- Página {i} ---\n{page[:1000]}")  # muestra primeras 1000 chars para inspección
    elif args.cmd == "extract-all":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        uc = ExtractAllPdfs(blob, extractor)

        out = uc.execute(ExtractAllPdfsInput(
            max_pages=args.max_pages,
            recursive=not args.non_recursive
        ))
        print(f"Procesados {len(out.results)} PDFs")
        for r in out.results:
            print(f"- {r.source_path} -> {r.result.page_count} páginas extraídas")
    elif args.cmd == "normalize":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        cfg = NormalizerConfig(
            fix_broken_hyphens=not args.no_fix_hyphens,
            join_soft_linebreaks=not args.no_join_soft_breaks,
        )
        uc = ExtractAndNormalizePdf(blob, extractor)
        out = uc.execute(ExtractAndNormalizeInput(
            relative_path=Path(args.relative_path),
            max_pages=args.max_pages,
            normalizer_cfg=cfg,
            join_pages=args.join,
        ))
        print(f"Archivo: {out.source_path}  (páginas: {out.page_count})")
        if args.join:
            print("\n--- Texto normalizado (primeros 1500 chars) ---\n")
            print((out.normalized_text or "")[:1500])
        else:
            for i, page in enumerate(out.normalized_pages or [], start=1):
                print(f"\n--- Página {i} ---\n{page[:1000]}")
    elif args.cmd == "normalize-all":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        uc = ExtractAndNormalizeAllPdfs(blob, extractor)
        out = uc.execute(ExtractAndNormalizeAllInput(
            max_pages=args.max_pages,
            recursive=not args.non_recursive,
            join_pages=args.join,
        ))
        print(f"Normalizados {len(out.results)} PDFs")
        for r in out.results:
            print(f"- {r.source_path} ({r.page_count} páginas)")
    elif args.cmd == "chunk":
        blob = LocalFileSystemBlob()
        extractor = PyMuPDFTextExtractor()
        tokenizer = SimpleRegexTokenizer()
        chunker = TokenChunker(tokenizer, ChunkerConfig(
            target_tokens=args.target,
            overlap_tokens=args.overlap,
            min_tokens=args.min_toks,
        ))
        normalizer = TextNormalizer()
        uc = ExtractNormalizeChunkPdf(blob, extractor, normalizer, chunker)
        out = uc.execute(ExtractNormalizeChunkInput(relative_path=Path(args.relative_path),
                                                max_pages=args.max_pages))
        print(f"Archivo: {out.source_path}  (páginas: {out.page_count})")
        print(f"Chunks generados: {len(out.chunks)}")
        for i, c in enumerate(out.chunks, start=1):
            preview = c.text[:160].replace("\n", " ")
            print(f"#{i:03d} page={c.page} toks={c.token_count} id={c.chunk_id[:12]}  {preview}...")


if __name__ == "__main__":
    main()
