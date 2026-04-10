from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any

from gemma4_capability_map.models.base import Retriever
from gemma4_capability_map.models.runtime_utils import load_hf_auth_token, resolve_model_source, should_use_local_files_only
from gemma4_capability_map.retrieval.ranking import rank_documents
from gemma4_capability_map.schemas import Document, RetrievalHit


_HF_EMBEDDER_CACHE: dict[tuple[str, str], object] = {}


class EmbeddingGemmaRetriever(Retriever):
    def __init__(self, model_id: str, backend: str = "heuristic", device: str = "auto") -> None:
        self.model_id = model_id
        self.backend = backend
        self.device = device
        self._corpora: dict[str, list[Document]] = defaultdict(list)
        self._model = None
        self._effective_model_id = model_id
        self._runtime_device: str | None = None
        self._document_embedding_cache: dict[tuple[tuple[str, ...], int, str], list[list[float]]] = {}

    def ensure_loaded(self) -> dict[str, Any]:
        if self.backend == "hf":
            self._ensure_hf_loaded()
            self._encode_texts([_query_text("warmup")])
        return self.runtime_info()

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_model": self.model_id,
            "effective_model": self._effective_model_id,
            "loaded": self._model is not None,
            "configured_device": self.device,
            "runtime_device": self._runtime_device,
        }

    def set_corpora(self, corpora: dict[str, list[dict]]) -> None:
        self._corpora = {
            corpus_id: [document if isinstance(document, Document) else Document.model_validate(document) for document in documents]
            for corpus_id, documents in corpora.items()
        }
        self._document_embedding_cache.clear()

    def search(
        self,
        query: str,
        corpus_id: str,
        top_k: int,
        dim: int,
        quantization: str,
    ) -> list[RetrievalHit]:
        documents = self._corpora.get(corpus_id, [])
        if self.backend == "hf":
            return self._search_hf(query, documents, top_k=top_k, dim=dim, quantization=quantization)
        return rank_documents(query=query, documents=documents, top_k=top_k, dim=dim, quantization=quantization)

    def _search_hf(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
        dim: int,
        quantization: str,
    ) -> list[RetrievalHit]:
        self._ensure_hf_loaded()

        query_embedding = _project_embedding(self._encode_texts([_query_text(query)])[0], dim=dim, quantization=quantization)
        document_embeddings = self._document_embeddings(documents, dim=dim, quantization=quantization)
        hits: list[RetrievalHit] = []
        for document, embedding in zip(documents, document_embeddings, strict=True):
            score = _dot(query_embedding, embedding)
            hits.append(RetrievalHit(doc_id=document.doc_id, content=document.content, score=score, metadata=document.metadata))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def _document_embeddings(self, documents: list[Document], dim: int, quantization: str) -> list[list[float]]:
        cache_key = (tuple(document.doc_id for document in documents), dim, quantization)
        cached = self._document_embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        embeddings = self._encode_texts([_document_text(document.content) for document in documents])
        projected = [_project_embedding(embedding, dim=dim, quantization=quantization) for embedding in embeddings]
        self._document_embedding_cache[cache_key] = projected
        return projected

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            encoded = self._model.encode(texts, normalize_embeddings=True)
        except TypeError:
            encoded = self._model.encode(texts)
        return [_normalize(_to_list(vector)) for vector in encoded]

    def _ensure_hf_loaded(self) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the hf extra to use the EmbeddingGemma HF backend.") from exc

        if self._model is not None:
            return

        token = load_hf_auth_token()
        resolved_source = resolve_model_source(self.model_id)
        runtime_device = _pick_embedding_device(torch, preferred=self.device)
        cache_key = (resolved_source, runtime_device)
        cached = _HF_EMBEDDER_CACHE.get(cache_key)
        if cached is None:
            cached = SentenceTransformer(
                resolved_source,
                token=token,
                local_files_only=should_use_local_files_only(resolved_source),
                device=runtime_device,
            )
            _HF_EMBEDDER_CACHE[cache_key] = cached
        self._model = cached
        self._effective_model_id = resolved_source
        self._runtime_device = runtime_device


def _query_text(query: str) -> str:
    return f"query: {query}"


def _document_text(content: str) -> str:
    return f"document: {content}"


def _project_embedding(embedding: list[float], dim: int, quantization: str) -> list[float]:
    target_dim = min(max(dim, 1), len(embedding))
    projected = list(embedding[:target_dim])
    if quantization == "int8":
        projected = _quantize(projected, levels=127)
    elif quantization == "int4":
        projected = _quantize(projected, levels=7)
    return _normalize(projected)


def _quantize(values: list[float], levels: int) -> list[float]:
    max_abs = max((abs(value) for value in values), default=0.0)
    if max_abs == 0.0:
        return values
    scale = levels / max_abs
    return [round(value * scale) / scale for value in values]


def _normalize(values: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(left * right for left, right in zip(a, b, strict=True)))


def _to_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]


def _pick_embedding_device(torch: Any, preferred: str = "auto") -> str:
    if preferred != "auto":
        if preferred == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("Requested mps device, but MPS is not available.")
        if preferred == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("Requested cuda device, but CUDA is not available.")
        if preferred == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported device selection: {preferred}")
    return "cpu"
