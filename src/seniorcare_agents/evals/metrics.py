import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / max(1, k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / max(1, len(relevant))


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    return next((1 / rank for rank, value in enumerate(retrieved, 1) if value in relevant), 0.0)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    actual = sum(
        (1 / math.log2(rank + 1))
        for rank, value in enumerate(retrieved[:k], 1)
        if value in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return actual / ideal if ideal else 0.0
