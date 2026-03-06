from src.api.metrics import mean_reciprocal_rank, recall_at_k


def test_metrics_recall_and_mrr_basic():
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c2"]

    assert recall_at_k(retrieved, relevant, k=2) == 1.0
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5
