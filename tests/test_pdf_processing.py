from src.processing.pdf_processor import LogicalDocument, PageInfo, PDFProcessor


def test_classify_documents_maps_pages_to_logical_docs():
    processor = PDFProcessor()
    processor.pages_info = [PageInfo(page_num=0, text="hello")]

    docs = processor.classify_documents()

    assert len(docs) == 1
    assert isinstance(docs[0], LogicalDocument)
    assert docs[0].doc_id == "doc_0"
