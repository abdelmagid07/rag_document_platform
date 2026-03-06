import time
from collections import deque
from statistics import median
from typing import Iterable


class LatencyTracker:
    """Tracks query latencies and exposes rolling percentile summaries."""

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.samples_ms = deque(maxlen=window_size)

    def add_sample(self, latency_ms: float) -> None:
        self.samples_ms.append(latency_ms)

    def summary(self) -> dict:
        if not self.samples_ms:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0}

        ordered = sorted(self.samples_ms)
        p50_idx = int(0.50 * (len(ordered) - 1))
        p95_idx = int(0.95 * (len(ordered) - 1))
        return {
            "count": len(ordered),
            "p50_ms": ordered[p50_idx],
            "p95_ms": ordered[p95_idx],
            "median_ms": median(ordered),
        }


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float:
    top_k = list(retrieved_ids)[:k]
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved_ids: Iterable[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for idx, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / idx
    return 0.0


class TimedBlock:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
