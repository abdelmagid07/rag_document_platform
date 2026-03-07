import time
from collections import defaultdict
import statistics


class LatencyTracker:
    """Track query latency metrics (p50, p95, p99)."""

    def __init__(self):
        self._latencies: list[float] = []

    def record(self, latency_ms: float):
        """Record a single latency measurement in milliseconds."""
        self._latencies.append(latency_ms)

    def summary(self) -> dict:
        """Return latency summary with p50, p95, p99 percentiles."""
        if not self._latencies:
            return {
                "total_queries": 0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        return {
            "total_queries": n,
            "p50_ms": round(sorted_lat[int(n * 0.50)], 2),
            "p95_ms": round(sorted_lat[int(min(n * 0.95, n - 1))], 2),
            "p99_ms": round(sorted_lat[int(min(n * 0.99, n - 1))], 2),
        }


def recall_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    """Compute Recall@k: fraction of relevant items found in top-k retrieved."""
    top_k = retrieved[:k]
    if not relevant:
        return 0.0
    return len(set(top_k) & set(relevant)) / len(relevant)


def mean_reciprocal_rank(retrieved: list, relevant: list) -> float:
    """Compute MRR: reciprocal of the rank of the first relevant item."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0
