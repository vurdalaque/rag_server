from __future__ import annotations

import math


def _binary_relevant(relevance: int, threshold: int) -> bool:
    return relevance >= threshold


def judge_precision_at_k(relevances: list[int], k: int, threshold: int = 2) -> float:
    if k <= 0:
        return 0.0
    top_k = relevances[:k]
    hits = sum(1 for rel in top_k if _binary_relevant(rel, threshold))
    return hits / k


def _dcg_at_k(relevances: list[int], k: int) -> float:
    clipped = relevances[:k]
    total = 0.0
    for rank, rel in enumerate(clipped, start=1):
        total += (2**rel - 1) / math.log2(rank + 1)
    return total


def judge_ndcg_at_k(relevances: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    dcg = _dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
