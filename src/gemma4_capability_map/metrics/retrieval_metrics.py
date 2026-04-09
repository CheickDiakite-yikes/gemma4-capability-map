from __future__ import annotations

import math


def recall_at_k(expected_doc_ids: list[str], actual_doc_ids: list[str]) -> float:
    if not expected_doc_ids:
        return 0.0
    hits = len(set(expected_doc_ids).intersection(actual_doc_ids))
    return hits / len(set(expected_doc_ids))


def reciprocal_rank(expected_doc_ids: list[str], actual_doc_ids: list[str]) -> float:
    expected = set(expected_doc_ids)
    for index, doc_id in enumerate(actual_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / index
    return 0.0


def ndcg(expected_doc_ids: list[str], actual_doc_ids: list[str]) -> float:
    expected = set(expected_doc_ids)
    dcg = 0.0
    for index, doc_id in enumerate(actual_doc_ids, start=1):
        if doc_id in expected:
            dcg += 1.0 / math.log2(index + 1)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, len(expected_doc_ids) + 1))
    return dcg / ideal if ideal else 0.0

