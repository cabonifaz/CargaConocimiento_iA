from dataclasses import dataclass
from typing import Iterable
from app.ports.outbound.blob_storage import BlobStoragePort, FileInfo
from app.config.settings import settings

@dataclass
class ListLocalPdfsInput:
    recursive: bool = True

@dataclass
class ListLocalPdfsOutput:
    files: list[FileInfo]

class ListLocalPdfs:
    def __init__(self, blob_storage: BlobStoragePort):
        self.blob = blob_storage

    def execute(self, params: ListLocalPdfsInput) -> ListLocalPdfsOutput:
        base = settings.COMPANY_FILES_DIR
        files = list(self.blob.list_pdfs(base, recursive=params.recursive))
        return ListLocalPdfsOutput(files=files)
