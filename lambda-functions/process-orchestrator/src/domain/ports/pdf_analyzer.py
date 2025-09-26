from abc import ABC, abstractmethod


class PdfAnalyzer(ABC):
    @abstractmethod
    def count_pages(self, pdf_path: str) -> int:
        pass