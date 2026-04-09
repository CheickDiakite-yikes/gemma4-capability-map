from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from gemma4_capability_map.schemas import Message, ModelTurn, RetrievalHit, ToolCall, ToolSpec


class Runner(ABC):
    def __init__(self, model_id: str, backend: str = "heuristic") -> None:
        self.model_id = model_id
        self.backend = backend

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        media: list[str],
        tool_specs: list[ToolSpec],
        thinking: bool,
        max_new_tokens: int | None = None,
    ) -> ModelTurn:
        raise NotImplementedError


class Retriever(Protocol):
    model_id: str
    backend: str

    def set_corpora(self, corpora: dict[str, list[dict]]) -> None:
        ...

    def search(
        self,
        query: str,
        corpus_id: str,
        top_k: int,
        dim: int,
        quantization: str,
    ) -> list[RetrievalHit]:
        ...


class Executor(Protocol):
    def step(self, state: dict, tool_call: ToolCall):
        ...
