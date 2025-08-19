from typing import Protocol, Iterable, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class FileInfo:
    path: Path
    size_bytes: int
    modified_at: datetime

class BlobStoragePort(Protocol):
    def list_pdfs(self, base_dir: Path, recursive: bool = True) -> Iterable[FileInfo]:
        ...

    def read_bytes(self, path: Path, max_bytes: Optional[int] = None) -> bytes:
        ...
