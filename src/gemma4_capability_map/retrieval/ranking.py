from __future__ import annotations

import math
import re

from gemma4_capability_map.schemas import Document, RetrievalHit


def rank_documents(
    query: str,
    documents: list[Document],
    top_k: int,
    dim: int,
    quantization: str,
) -> list[RetrievalHit]:
    query_terms = _tokenize(query)
    hits: list[RetrievalHit] = []
    dim_bonus = 1.0 - max(0, 768 - dim) / 2048
    quant_penalty = {"none": 1.0, "int8": 0.98, "int4": 0.94}.get(quantization, 1.0)
    for document in documents:
        doc_terms = _tokenize(document.content)
        overlap = len(query_terms & doc_terms)
        if not overlap:
            continue
        idfish = math.log(1 + overlap)
        score = overlap * idfish * dim_bonus * quant_penalty
        hits.append(RetrievalHit(doc_id=document.doc_id, content=document.content, score=score, metadata=document.metadata))
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

