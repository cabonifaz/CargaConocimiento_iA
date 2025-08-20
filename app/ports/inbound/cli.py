import argparse
from pathlib import Path

from app.adapters.blob.local_fs import LocalFileSystemBlob
from app.adapters.extract.pymupdf_text_extractor import PyMuPDFTextExtractor
from app.application.use_cases.extract_text_from_pdf import (
    ExtractTextFromPdf, ExtractTextInput
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

if __name__ == "__main__":
    main()
