from pypdf import PdfReader

from ..domain.ports.pdf_analyzer import PdfAnalyzer


class PyPdfAnalyzer(PdfAnalyzer):
    def count_pages(self, pdf_path: str) -> int:
        reader = PdfReader(pdf_path)
        return len(reader.pages)