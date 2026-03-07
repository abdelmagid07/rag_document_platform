from ..retrieval.vector_store import get_vector_store


def write_vectors(chunks: list[str], embeddings: list[list[float]], document_id: str):
    """
    Write chunk embeddings into the FAISS vector store and persist to disk.
    
    Args:
        chunks: list of text chunks
        embeddings: list of embedding vectors (parallel to chunks)
        document_id: the document these chunks belong to
    """
    store = get_vector_store()

    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "doc_id": document_id,
            "chunk_id": i,
            "text": chunk,
        })

    store.insert(embeddings, metadata)
    store.save()  # persist after every ingestion