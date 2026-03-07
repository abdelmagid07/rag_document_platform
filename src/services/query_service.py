import time
import json
from .embedding_service import get_embeddings
from ..retrieval.pg_vector_store import PgVectorStore
from ..generation.generator import generate_answer, generate_answer_stream
from ..api.metrics import metrics_store
from .logger import logger


async def run_query(query: str, top_k: int):
    start_total = time.time()
    
    # Embed query 
    query_embedding = await get_embeddings(query)

    # Retrieve documents
    start_retr = time.time()
    retrieved_docs = await PgVectorStore.search(query_embedding, top_k)
    retr_latency = (time.time() - start_retr) * 1000
    metrics_store.record("retrieval", retr_latency)

    # Total Latency 
    total_latency = (time.time() - start_total) * 1000
    metrics_store.record("total", total_latency)

    # Generate answer 
    answer = await generate_answer(query, retrieved_docs)

    # Structured Logging
    logger.info(
        f"Query processed: {query[:50]}...",
        extra={"props": {
            "query": query,
            "top_k": top_k,
            "retrieved_count": len(retrieved_docs),
            "latency_total_ms": round(total_latency, 2),
            "latency_retrieval_ms": round(retr_latency, 2),
            "doc_ids": [doc["doc_id"] for doc in retrieved_docs]
        }}
    )

    # Format sources
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "score": doc["score"]
        })

    return answer, sources


async def run_query_stream(query: str, top_k: int):
    """
    Asynchronous generator for SSE streaming.
    Yields JSON-encoded strings.
    """
    yield f"data: {json.dumps({'status': 'searching'})}\n\n"
    start_total = time.time()
    
    # Embed query
    query_embedding = await get_embeddings(query)

    # Retrieve documents
    start_retr = time.time()
    retrieved_docs = await PgVectorStore.search(query_embedding, top_k)
    retr_latency = (time.time() - start_retr) * 1000
    metrics_store.record("retrieval", retr_latency)

    # Yield sources immediately to UI
    sources = [{
        "doc_id": doc["doc_id"],
        "text": doc["text"],
        "score": doc["score"]
    } for doc in retrieved_docs]
    
    yield f"data: {json.dumps({'sources': sources})}\n\n"

    # Total Latency 
    total_latency = (time.time() - start_total) * 1000
    metrics_store.record("total", total_latency)

    # Generate answer stream
    async for token in generate_answer_stream(query, retrieved_docs):
        yield f"data: {json.dumps({'token': token})}\n\n"
    
    # Final metadata event
    yield f"data: {json.dumps({'latency_ms': round(total_latency, 2), 'done': True})}\n\n"

    # Structured Logging
    logger.info(
        f"Streamed query processed: {query[:50]}...",
        extra={"props": {
            "query": query,
            "top_k": top_k,
            "retrieved_count": len(retrieved_docs),
            "latency_total_ms": round(total_latency, 2),
            "latency_retrieval_ms": round(retr_latency, 2),
            "doc_ids": [doc["doc_id"] for doc in retrieved_docs]
        }}
    )