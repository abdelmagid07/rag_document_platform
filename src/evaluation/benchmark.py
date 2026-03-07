import time
import asyncio
import numpy as np

from datasets import load_dataset
from ..services.embedding_service import get_embeddings
from ..retrieval.vector_store import VectorStore
from ..api.metrics import recall_at_k, mean_reciprocal_rank
from ..services.logger import logger

async def run_benchmark(sample_size: int = 25):
    """
    Highly optimized evaluation benchmark.
    Uses batch embedding and matrix math for speed.
    """
    logger.info(f"Starting HotpotQA Benchmark (size={sample_size})...")
    start_bench = time.time()
    
    # 1. Load Dataset (Streaming)
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    
    # Efficiently take samples
    samples = []
    logger.info(f"Fetching {sample_size} samples from HotpotQA...")
    for i, item in enumerate(dataset):
        samples.append(item)
        if len(samples) >= sample_size:
            break
            
    logger.info(f"Dataset loaded: {len(samples)} samples.")

    # Extract Contexts and Batch Embed
    all_chunks = []
    chunk_metadata = []
    for i, sample in enumerate(samples):
        for title, sentences in zip(sample["context"]["title"], sample["context"]["sentences"]):
            text = " ".join(sentences)
            all_chunks.append(text)
            chunk_metadata.append({"title": title, "sample_idx": i})

    logger.info(f"Generating embeddings for {len(all_chunks)} chunks in batches...")
    # Sub-batch to prevent large memory spikes or hangs
    sub_batch_size = 500
    all_corpus_embeddings = []
    for j in range(0, len(all_chunks), sub_batch_size):
        batch_texts = all_chunks[j:j + sub_batch_size]
        logger.info(f"Embedding chunk batch {j // sub_batch_size + 1}/{(len(all_chunks) // sub_batch_size) + 1}...")
        batch_emb = await get_embeddings(batch_texts)
        all_corpus_embeddings.append(batch_emb)
    
    corpus_matrix = np.vstack(all_corpus_embeddings)

    # Store in temporary Postgres table for evaluation
    from src.retrieval.pg_vector_store import PgVectorStore
    from src.services.database import Database
    import uuid
    import hashlib
    
    logger.info("Clearing previous benchmark data...")
    # Clean up benchmark documents
    await Database.execute("DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE filename = 'benchmark_eval')")
    await Database.execute("DELETE FROM documents WHERE filename = 'benchmark_eval'")

    logger.info(f"Ingesting {len(all_chunks)} eval chunks into Postgres...")
    
    # Track title to UUID mapping for recall verification
    doc_id_to_title = {}
    
    # Pre-group chunks by document to enable batch inserts
    from collections import defaultdict
    docs_to_ingest = defaultdict(list) # doc_uuid -> list of (embedding, text)
    
    for i, meta in enumerate(chunk_metadata):
        title = meta['title']
        doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, title))
        docs_to_ingest[doc_uuid].append((corpus_matrix[i], all_chunks[i]))
        doc_id_to_title[doc_uuid] = title

    logger.info(f"Ingesting into {len(docs_to_ingest)} documents...")
    
    # Insert document metadata in one batch
    doc_metadata = [(uid, "benchmark_eval") for uid in doc_id_to_title.keys()]
    await Database.executemany(
        "INSERT INTO documents (id, filename) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        doc_metadata
    )

    # Insert chunks in document-level batches
    for doc_uuid, items in docs_to_ingest.items():
        embs = np.array([item[0] for item in items])
        meta = [{"text": item[1]} for item in items]
        await PgVectorStore.insert(embs, meta, doc_uuid)

    # Batch Embed Queries
    questions = [s["question"] for s in samples]
    logger.info(f"Generating embeddings for {len(questions)} queries...")
    query_matrix = await get_embeddings(questions)

    # Search in Postgres and Calculate Metrics
    logger.info("Performing similarity search in Postgres...")
    recalls = []
    mrrs = []
    
    for i, sample in enumerate(samples):
        if i % 20 == 0:
            logger.info(f"Processing sample {i}/{len(samples)}...")
            
        relevant_titles = set(sample["supporting_facts"]["title"])
        
        # Search using PgVectorStore
        results = await PgVectorStore.search(query_matrix[i], top_k=5)
        
        # Debug: check first result
        if results:
            first = results[0]
            logger.info(f"Query {i} top result: score={first['score']:.4f}, doc_id={first['doc_id']}")
        
        # Look up title from our local mapping
        retrieved_titles = [doc_id_to_title.get(res["doc_id"]) for res in results]
        
        # Debug: log retrieved titles
        valid_titles = [t for t in retrieved_titles if t]
        logger.info(f"Query {i} retrieved titles: {valid_titles}")
        logger.info(f"Query {i} relevant titles: {list(relevant_titles)}")

        # Filter out any None values
        retrieved_titles = valid_titles
        
        # Recall@5
        recall = 1.0 if any(t in relevant_titles for t in retrieved_titles) else 0.0
        recalls.append(recall)
        
        # MRR
        mrr = 0.0
        for rank, t in enumerate(retrieved_titles):
            if rank < 5:  # Consider only top 5 as before
                if t in relevant_titles:
                    mrr = 1.0 / (rank + 1)
                    break
        mrrs.append(mrr)

    avg_recall = sum(recalls) / len(recalls)
    avg_mrr = sum(mrrs) / len(mrrs)
    total_time = time.time() - start_bench

    logger.info(f"Benchmark Complete in {total_time:.1f}s. Recall@5: {avg_recall:.2f}, MRR: {avg_mrr:.2f}")
    
    return {
        "sample_size": sample_size,
        "avg_recall_at_5": round(avg_recall, 4),
        "mrr": round(avg_mrr, 4),
        "total_time_sec": round(total_time, 2)
    }

if __name__ == "__main__":
    import asyncio
    import sys
    
    # Allow passing sample size as an argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    asyncio.run(run_benchmark(size))
