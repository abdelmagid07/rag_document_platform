from ..retrieval.vector_store import get_vector_store


def retrieve_documents(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Retrieve the top-k most relevant document chunks for a query embedding.
    
    Returns list of dicts with: doc_id, chunk_id, text, score
    """
    store = get_vector_store()
    return store.search(embedding=query_embedding, top_k=top_k)