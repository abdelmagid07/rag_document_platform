import time
import asyncio
from datasets import load_dataset
from ..services.embedding_service import get_embeddings
from ..retrieval.vector_store import VectorStore
from ..api.metrics import recall_at_k, mean_reciprocal_rank
from ..services.logger import logger

async def run_benchmark(sample_size: int = 10):
    """
    Run evaluation benchmark using HotpotQA dataset.
    """
    logger.info(f"Starting HotpotQA Benchmark (size={sample_size})...")
    
    # Load HotpotQA (streaming)
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    samples = list(dataset.take(sample_size))

    # Initialize temporary vector store 
    temp_store = VectorStore(dim=384)
    
    # Preparation: Ingest all contexts
    all_chunks = []
    all_metadata = []
    
    for i, sample in enumerate(samples):
        # HotpotQA context is a dict of lists
        for title, sentences in zip(sample["context"]["title"], sample["context"]["sentences"]):
            full_text = " ".join(sentences)
            all_chunks.append(full_text)
            all_metadata.append({
                "doc_id": f"eval_{i}_{title}",
                "text": full_text
            })

    logger.info(f"Ingesting {len(all_chunks)} chunks...")
    # Awaiting async embeddings
    embeddings = await get_embeddings(all_chunks)
    temp_store.insert(embeddings, all_metadata)

    # Evaluation
    recalls = []
    mrrs = []
    
    for sample in samples:
        query = sample["question"]
        relevant_titles = set(sample["supporting_facts"]["title"])
        
        # Awaiting async embeddings
        query_emb = await get_embeddings(query)
        results = temp_store.search(query_emb, top_k=5)
        
        retrieved_titles = [res["doc_id"].split("_", 2)[2] for res in results]
        
        recall = 1.0 if any(t in relevant_titles for t in retrieved_titles) else 0.0
        recalls.append(recall)
        
        rank = 0
        for idx, t in enumerate(retrieved_titles):
            if t in relevant_titles:
                rank = 1.0 / (idx + 1)
                break
        mrrs.append(rank)

    avg_recall = sum(recalls) / len(recalls)
    avg_mrr = sum(mrrs) / len(mrrs)

    logger.info(f"Benchmark Complete. Recall@5: {avg_recall:.2f}, MRR: {avg_mrr:.2f}")
    
    return {
        "sample_size": sample_size,
        "avg_recall_at_5": round(avg_recall, 4),
        "mrr": round(avg_mrr, 4)
    }

if __name__ == "__main__":
    asyncio.run(run_benchmark(5))
