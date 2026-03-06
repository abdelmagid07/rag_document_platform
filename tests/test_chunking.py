from src.processing.chunking import chunk_document
from src.processing.pdf_processor import LogicalDocument


def test_chunk_document_returns_chunks_for_text():
    doc = LogicalDocument(
        doc_id="d1",
        doc_type="Report",
        page_start=1,
        page_end=1,
        text="word " * 1200,
    )

    chunks = chunk_document(doc)

    assert len(chunks) >= 2
    assert chunks[0].doc_id == "d1"
