from __future__ import annotations

from gemma4_capability_map.retrieval.chunking import chunk_document
from gemma4_capability_map.schemas import Document


def build_corpus_index(corpora: dict[str, list[Document]], chunk_size: int = 120) -> dict[str, list[Document]]:
    indexed: dict[str, list[Document]] = {}
    for corpus_id, documents in corpora.items():
        indexed[corpus_id] = []
        for document in documents:
            indexed[corpus_id].extend(chunk_document(document, chunk_size=chunk_size))
    return indexed

