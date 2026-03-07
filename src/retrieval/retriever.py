from .vector_store import VectorStore

vector_store = VectorStore()


def retrieve_documents(query_embedding, top_k):

    results = vector_store.search(
        embedding=query_embedding,
        top_k=top_k
    )

    documents = []

    for r in results:
        documents.append({
            "doc_id": r["doc_id"],
            "text": r["text"],
            "score": r["score"]
        })

    return documents