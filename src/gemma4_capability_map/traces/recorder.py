from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.schemas import RunTrace


class TraceRecorder:
    def __init__(self) -> None:
        self.traces: list[RunTrace] = []

    def add(self, trace: RunTrace) -> None:
        self.traces.append(trace)

    def write(self, path: str | Path) -> None:
        dump_jsonl(path, self.traces)

