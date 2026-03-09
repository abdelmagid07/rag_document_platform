import asyncio
import time
import numpy as np
from datasets import load_dataset
from collections import defaultdict

from src.services.embedding_service import get_embeddings
from src.retrieval.pg_vector_store import PgVectorStore
from src.services.database import Database
from src.services.logger import logger


TOP_K = 5
EMBED_BATCH = 256


def recall_at_k(relevant, retrieved, k):
    retrieved = retrieved[:k]
    return len(set(relevant) & set(retrieved)) / len(relevant)


def mean_reciprocal_rank(relevant, retrieved):
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


async def batch_embed(texts, batch_size=EMBED_BATCH):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = await get_embeddings(batch)
        embeddings.append(emb)
    return np.vstack(embeddings)


async def run_benchmark(sample_size=500):

    logger.info(f"Starting HotpotQA Benchmark (n={sample_size})")
    start = time.time()

    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split="validation",
        streaming=True
    )

    samples = []
    for item in dataset:
        samples.append(item)
        if len(samples) >= sample_size:
            break

    logger.info(f"Loaded {len(samples)} queries")

    # Build Corpus
    corpus_chunks = []
    metadata = []

    for i, sample in enumerate(samples):
        for title, sentences in zip(
            sample["context"]["title"],
            sample["context"]["sentences"]
        ):

            text = " ".join(sentences)

            corpus_chunks.append(text)

            metadata.append({
                "title": title,
                "sample_idx": i
            })

    logger.info(f"Corpus size: {len(corpus_chunks)} chunks")

    # Embed Corpus
    logger.info("Embedding corpus...")
    corpus_embeddings = await batch_embed(corpus_chunks)

    # Reset Benchmark Tables
    await Database.execute(
        "DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE filename='benchmark')"
    )

    await Database.execute(
        "DELETE FROM documents WHERE filename='benchmark'"
    )

    # Insert Documents
    import uuid

    doc_map = {}
    grouped = defaultdict(list)

    for i, meta in enumerate(metadata):

        title = meta["title"]

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, title))

        doc_map[doc_id] = title

        grouped[doc_id].append((corpus_embeddings[i], corpus_chunks[i]))

    logger.info(f"Inserting {len(grouped)} documents")

    docs = [(doc_id, "benchmark") for doc_id in doc_map.keys()]

    await Database.executemany(
        "INSERT INTO documents (id, filename) VALUES ($1,$2) ON CONFLICT DO NOTHING",
        docs
    )

    for doc_id, rows in grouped.items():

        embs = np.array([r[0] for r in rows])

        meta = [{"text": r[1]} for r in rows]

        await PgVectorStore.insert(embs, meta, doc_id)

    # Embed Queries
    questions = [s["question"] for s in samples]

    logger.info("Embedding queries...")

    query_embeddings = await batch_embed(questions)


    # Evaluation
    recalls = []
    mrrs = []

    for i, sample in enumerate(samples):

        relevant_titles = set(sample["supporting_facts"]["title"])

        results = await PgVectorStore.search(
            query_embeddings[i],
            top_k=TOP_K
        )

        retrieved_titles = [
            doc_map.get(r["doc_id"])
            for r in results
            if r["doc_id"] in doc_map
        ]

        rec = recall_at_k(relevant_titles, retrieved_titles, TOP_K)

        mrr = mean_reciprocal_rank(relevant_titles, retrieved_titles)

        recalls.append(rec)
        mrrs.append(mrr)

    avg_recall = np.mean(recalls)
    avg_mrr = np.mean(mrrs)

    runtime = time.time() - start

    logger.info("Benchmark Complete")
    logger.info(f"Recall@{TOP_K}: {avg_recall:.4f}")
    logger.info(f"MRR: {avg_mrr:.4f}")
    logger.info(f"Runtime: {runtime:.2f}s")

    return {
        "samples": sample_size,
        "recall@5": float(avg_recall),
        "mrr": float(avg_mrr),
        "runtime_sec": runtime
    }


if __name__ == "__main__":

    import sys

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    asyncio.run(run_benchmark(size))