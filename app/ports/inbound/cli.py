import argparse
from pathlib import Path

from app.adapters.blob.local_fs import LocalFileSystemBlob
from app.adapters.extract.pymupdf_text_extractor import PyMuPDFTextExtractor
from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput
)
from app.application.use_cases.extract_text_from_all_pdfs import (
    ExtractAllPdfs, ExtractAllPdfsInput
)

def main():
    parser = argparse.ArgumentParser(description="Herramientas RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ext = sub.add_parser("extract", help="Extraer texto de un PDF bajo company_files/")
    ext.add_argument(
        "relative_path", 
        type=str, 
        help="ruta del PDF: puede ser solo el nombre (dentro de company_files), "
            "o incluir el prefijo 'company_files/', o una ruta absoluta dentro de esa carpeta"
    )
    ext.add_argument("--max-pages", type=int, default=None)

    allp = sub.add_parser("extract-all", help="Extraer texto de todos los PDFs bajo company_files/")
    allp.add_argument("--max-pages", type=int, default=None)
    allp.add_argument("--non-recursive", action="store_true")

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


if __name__ == "__main__":
    main()
