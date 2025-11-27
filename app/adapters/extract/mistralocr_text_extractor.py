from app.ports.outbound.text_extractor import TextExtractorPort, TextExtractionResult
from mistralai import Mistral
from typing import Optional
import base64

class MistralOCRTextExtractor(TextExtractorPort):
    def __init__(self, mistral_client: Mistral, mistral_model: str = "mistral-ocr-latest"):
        self.mistral_client = mistral_client
        self.mistral_model = mistral_model
        print(f"[Mistral OCR] Extractor inicializado con modelo: {mistral_model}")

    def extract_from_bytes(self, data: bytes, max_pages: Optional[int] = None) -> TextExtractionResult:
        try:
            base64_pdf = base64.b64encode(data).decode('utf-8')

            print(f"[Mistral OCR] Procesando PDF ({len(data)} bytes)...")

            ocr_response = self.mistral_client.ocr.process(
                model=self.mistral_model,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                include_image_base64=False
            )

            if ocr_response is None:
                raise ValueError("Mistral OCR retorno None - verifica la API key y la cuota disponible")

            if not hasattr(ocr_response, 'pages'):
                raise ValueError(f"Respuesta de Mistral OCR no contiene 'pages': {type(ocr_response)}")

            pages = [page.markdown for page in ocr_response.pages]
            if max_pages is not None:
                pages = pages[:max_pages]

            print(f"[Mistral OCR] Extraccion exitosa: {len(pages)} paginas")

            return TextExtractionResult(
                pages=pages,
                page_count=len(ocr_response.pages),
            )
        except Exception as e:
            print(f"[Mistral OCR] ERROR: {str(e)}")
            raise