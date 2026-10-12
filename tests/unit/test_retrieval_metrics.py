from __future__ import annotations

import math

import pytest

from retrieval_quality_metrics import judge_ndcg_at_k, judge_precision_at_k


def test_precision_perfect_ranking() -> None:
    relevances = [3, 3, 2, 1, 0]
    assert judge_precision_at_k(relevances, k=3) == 1.0
    assert judge_precision_at_k(relevances, k=5) == 0.6


def test_precision_all_irrelevant() -> None:
    relevances = [0, 1, 0, 1]
    assert judge_precision_at_k(relevances, k=4) == 0.0
    assert judge_precision_at_k(relevances, k=2, threshold=3) == 0.0


def test_precision_partial_relevance() -> None:
    relevances = [3, 0, 2, 1]
    assert judge_precision_at_k(relevances, k=4) == 0.5
    assert judge_precision_at_k(relevances, k=2) == 0.5


def test_precision_k_larger_than_list() -> None:
    relevances = [3, 2]
    assert judge_precision_at_k(relevances, k=5) == 0.4


def test_precision_empty_list() -> None:
    assert judge_precision_at_k([], k=3) == 0.0


def test_precision_respects_custom_threshold() -> None:
    relevances = [2, 1, 2]
    assert judge_precision_at_k(relevances, k=3, threshold=2) == pytest.approx(2 / 3)
    assert judge_precision_at_k(relevances, k=3, threshold=3) == 0.0


def test_precision_non_positive_k() -> None:
    assert judge_precision_at_k([3, 3], k=0) == 0.0
    assert judge_precision_at_k([3, 3], k=-1) == 0.0


def test_ndcg_perfect_ranking() -> None:
    relevances = [3, 2, 1, 0]
    assert judge_ndcg_at_k(relevances, k=4) == pytest.approx(1.0)


def test_ndcg_all_irrelevant() -> None:
    relevances = [0, 0, 0]
    assert judge_ndcg_at_k(relevances, k=3) == 0.0


def test_ndcg_partial_ranking() -> None:
    relevances = [1, 3, 0, 2]
    ideal_dcg = _manual_dcg([3, 2, 1, 0], 4)
    actual_dcg = _manual_dcg(relevances, 4)
    expected = actual_dcg / ideal_dcg
    assert judge_ndcg_at_k(relevances, k=4) == pytest.approx(expected)
    assert 0.0 < judge_ndcg_at_k(relevances, k=4) < 1.0


def test_ndcg_k_larger_than_list() -> None:
    relevances = [3, 1]
    full = judge_ndcg_at_k(relevances, k=2)
    padded_equivalent = judge_ndcg_at_k(relevances, k=10)
    assert padded_equivalent == pytest.approx(full)


def test_ndcg_empty_list() -> None:
    assert judge_ndcg_at_k([], k=5) == 0.0


def test_ndcg_non_positive_k() -> None:
    assert judge_ndcg_at_k([3, 2], k=0) == 0.0
    assert judge_ndcg_at_k([3, 2], k=-2) == 0.0


def test_ndcg_graded_gain_formula() -> None:
    relevances = [2]
    expected = (2**2 - 1) / math.log2(2)
    assert judge_ndcg_at_k(relevances, k=1) == pytest.approx(1.0)
    assert _manual_dcg(relevances, 1) == pytest.approx(expected)


def _manual_dcg(relevances: list[int], k: int) -> float:
    total = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / math.log2(rank + 1)
    return total
