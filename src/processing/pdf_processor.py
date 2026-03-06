from dataclasses import dataclass
from typing import List

from ..config import Config


@dataclass
class PageInfo:
    page_num: int
    text: str
    doc_type: str | None = None
    page_in_doc: int = 0


@dataclass
class LogicalDocument:
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str


class PDFProcessor:
    """Extract and analyze PDFs into logical documents."""

    def __init__(self):
        self.pages_info: List[PageInfo] = []
        self.logical_docs: List[LogicalDocument] = []

    def extract_text(self, pdf_file: str) -> List[PageInfo]:
        """Extract text from PDF pages, with optional OCR fallback."""
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_file)
        pages_info = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if not text.strip() and Config.USE_OCR:
                import io

                import pytesseract
                from PIL import Image

                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            pages_info.append(PageInfo(page_num=i, text=text))

        self.pages_info = pages_info
        return pages_info

    def classify_documents(self) -> List[LogicalDocument]:
        """Simple placeholder classification into one logical document per page."""
        self.logical_docs = [
            LogicalDocument(
                doc_id=f"doc_{i}",
                doc_type="Report",
                page_start=p.page_num,
                page_end=p.page_num,
                text=p.text,
            )
            for i, p in enumerate(self.pages_info)
        ]
        return self.logical_docs
