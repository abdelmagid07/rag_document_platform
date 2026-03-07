from .ingestion import PDFProcessor
from .chunking import chunk_all_documents
from .retriever import IntelligentRetriever
from .answer_gen import generate_answer

class EnhancedDocumentStore:
    def __init__(self):
        self.processor = PDFProcessor()
        self.retriever = IntelligentRetriever()
        self.chunks = []

    def process_pdf(self, pdf_file):
        pages = self.processor.extract_text(pdf_file)
        docs = self.processor.classify_documents()
        self.chunks = chunk_all_documents(docs)
        # Generate embeddings
        for c in self.chunks:
            c.embedding = self.retriever.embedder.get_text_embedding(c.text)
        # Build vector index
        self.retriever.vector_store.build_index(self.chunks)

    def query(self, question: str):
        results = self.retriever.retrieve(question)
        return generate_answer(question, results)