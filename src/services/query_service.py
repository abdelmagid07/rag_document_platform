from ..retrieval.embedder import embed_query
from ..retrieval.retriever import retrieve_documents
from ..generation.generator import generate_answer


async def run_query(query: str, top_k: int):

    # Embed query
    query_embedding = embed_query(query)

    # Retrieve documents
    retrieved_docs = retrieve_documents(query_embedding, top_k)

    # Generate answer
    answer = generate_answer(query, retrieved_docs)

    # Format sources
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "score": doc["score"]
        })

    return answer, sources