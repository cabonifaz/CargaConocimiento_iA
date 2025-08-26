from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime
import os

from app.ports.outbound.blob_storage import BlobStoragePort, FileInfo
from app.config.settings import settings

class LocalFileSystemBlob(BlobStoragePort):
    def __init__(
        self,
        exclude_globs: list[str] | None = None,
        max_pdf_mb: int | None = None,
    ) -> None:
        self.exclude_globs = exclude_globs or settings.EXCLUDE_GLOBS
        self.max_pdf_mb = max_pdf_mb or settings.MAX_PDF_MB
        self.base_dir = settings.COMPANY_FILES_DIR

    def list_pdfs(self, base_dir: Path, recursive: bool = True) -> Iterable[FileInfo]:
        if not base_dir.exists() or not base_dir.is_dir():
            return []

        # 1) recolectar candidatos por extensión .pdf (case-insensitive)
        candidates = (
            base_dir.rglob("*.pdf") if recursive else base_dir.glob("*.pdf")
        )

        # 2) aplicar exclusiones (globs)
        def excluded(p: Path) -> bool:
            return any(p.match(pattern) for pattern in self.exclude_globs)

        # 3) filtrar por tamaño y validar magic header %PDF
        max_bytes = self.max_pdf_mb * 1024 * 1024

        valid: list[FileInfo] = []
        for p in candidates:
            if excluded(p):
                continue
            try:
                st = p.stat()
                if st.st_size == 0 or st.st_size > max_bytes:
                    continue

                # Validación rápida del encabezado PDF
                # Leer primeros bytes sin cargar todo el archivo
                with p.open("rb") as fh:
                    header = fh.read(5)  # b'%PDF-'
                    if not header.startswith(b"%PDF"):
                        continue

                valid.append(
                    FileInfo(
                        path=p,
                        size_bytes=st.st_size,
                        modified_at=datetime.fromtimestamp(st.st_mtime),
                    )
                )
            except (OSError, PermissionError):
                # ignora archivos inaccesibles
                continue

        # orden opcional por fecha de modificación descendente
        valid.sort(key=lambda fi: fi.modified_at, reverse=True)
        return valid

    def read_bytes(self, path: Path, max_bytes: Optional[int] = None) -> bytes:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(path))
        # lectura acotada si se solicita
        if max_bytes is not None:
            with path.open("rb") as fh:
                return fh.read(max_bytes)
        bytes = path.read_bytes()
        print(f"{len(bytes)} bytes obtenidos de {path}")
        return bytes
