import time
import json
from pathlib import Path
from typing import Dict, List

METRICS_FILE = Path("data/metrics.json")

class MetricsStore:
    """Manages persistent metrics for the RAG platform."""

    def __init__(self):
        self.latencies: Dict[str, List[float]] = {
            "total": [],
            "embedding": [],
            "retrieval": []
        }
        self.load()

    def record(self, stage: str, latency_ms: float):
        if stage in self.latencies:
            self.latencies[stage].append(latency_ms)
            self.save()

    def get_percentiles(self, stage: str) -> dict:
        data = self.latencies.get(stage, [])
        if not data:
            return {"count": 0, "p50": 0, "p95": 0, "p99": 0}
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        return {
            "count": n,
            "p50": round(sorted_data[int(n * 0.50)], 2),
            "p95": round(sorted_data[int(min(n * 0.95, n - 1))], 2),
            "p99": round(sorted_data[int(min(n * 0.99, n - 1))], 2),
        }

    def summary(self) -> dict:
        return {stage: self.get_percentiles(stage) for stage in self.latencies}

    def save(self):
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_FILE, "w") as f:
            json.dump(self.latencies, f)

    def load(self):
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r") as f:
                    self.latencies.update(json.load(f))
            except Exception:
                pass

# Singleton
metrics_store = MetricsStore()

def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int = 5) -> float:
    if not relevant_ids: return 0.0
    top_k = set(retrieved_ids[:k])
    matches = len(top_k.intersection(set(relevant_ids)))
    return matches / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float:
    for i, res_id in enumerate(retrieved_ids):
        if res_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0
