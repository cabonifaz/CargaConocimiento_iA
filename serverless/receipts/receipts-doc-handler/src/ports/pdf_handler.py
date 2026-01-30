from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ReadPDFResponse:
    reader: Any
    total_pages: int

class PDFHandlerPort(ABC):
    @abstractmethod
    def read_pdf(self, bytes: bytes) -> Optional[ReadPDFResponse]:
        pass

    @abstractmethod
    def write_pdf(self, content) -> bytes:
        pass