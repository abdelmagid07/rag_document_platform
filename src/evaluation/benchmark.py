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

    # 3. Batch Embed Queries
    questions = [s["question"] for s in samples]
    logger.info(f"Generating embeddings for {len(questions)} queries...")
    query_matrix = await get_embeddings(questions)

    # Matrix Similarity Search (Vectorized)
    logger.info("Calculating similarities...")
    
    # Calculate Distances
    q_sq = np.sum(query_matrix**2, axis=1, keepdims=True)
    c_sq = np.sum(corpus_matrix**2, axis=1, keepdims=True).T
    distances = np.sqrt(np.maximum(q_sq - 2 * np.dot(query_matrix, corpus_matrix.T) + c_sq, 0))

    # Calculate Metrics
    logger.info(f"Calculating metrics for {len(samples)} samples...")
    recalls = []
    mrrs = []
    
    for i, sample in enumerate(samples):
        if i % 20 == 0:
            logger.info(f"Processing sample {i}/{len(samples)}...")
            
        relevant_titles = set(sample["supporting_facts"]["title"])
        
        # Get top 5 indices for this query
        top_k_indices = np.argsort(distances[i])[:5]
        retrieved_titles = [chunk_metadata[idx]["title"] for idx in top_k_indices]
        
        # Recall@5
        recall = 1.0 if any(t in relevant_titles for t in retrieved_titles) else 0.0
        recalls.append(recall)
        
        # MRR
        mrr = 0.0
        for rank, t in enumerate(retrieved_titles):
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
