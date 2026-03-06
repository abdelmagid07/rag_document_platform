from typing import List
from dataclasses import dataclass
from .ingestion import LogicalDocument
from .config import Config

@dataclass
class ChunkMetadata:
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    embedding: list = None

def chunk_document(logical_doc: LogicalDocument) -> List[ChunkMetadata]:
    """Simple sliding window chunking with overlap."""
    words = logical_doc.text.split()
    chunks = []
    stride = Config.CHUNK_SIZE - Config.CHUNK_OVERLAP
    for i, start_idx in enumerate(range(0, len(words), stride)):
        end_idx = min(start_idx + Config.CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start_idx:end_idx])
        chunks.append(ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=i,
            page_start=logical_doc.page_start,
            page_end=logical_doc.page_end,
            text=chunk_text
        ))
        if end_idx == len(words):
            break
    return chunks

def chunk_all_documents(logical_docs: List[LogicalDocument]) -> List[ChunkMetadata]:
    """Chunk all logical documents."""
    all_chunks = []
    for doc in logical_docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks