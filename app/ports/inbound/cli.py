import argparse
from app.adapters.blob.local_fs import LocalFileSystemBlob
from app.application.use_cases.list_local_pdfs import (
    ListLocalPdfs, ListLocalPdfsInput
)

def main():
    parser = argparse.ArgumentParser(description="Listar PDFs en company_files/")
    parser.add_argument("--non-recursive", action="store_true", help="No recorrer subcarpetas")
    args = parser.parse_args()

    uc = ListLocalPdfs(LocalFileSystemBlob())
    out = uc.execute(ListLocalPdfsInput(recursive=not args.non_recursive))

    if not out.files:
        print("No se encontraron PDFs válidos en company_files/")
        return

    print(f"Encontrados {len(out.files)} PDF(s):")
    for f in out.files:
        print(f"- {f.path}  ({f.size_bytes/1024:.1f} KiB)  mtime={f.modified_at.isoformat(timespec='seconds')}")

if __name__ == "__main__":
    main()
