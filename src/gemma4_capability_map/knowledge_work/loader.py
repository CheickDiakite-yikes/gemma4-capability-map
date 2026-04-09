from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.knowledge_work.schemas import Episode


def load_episodes(path: str | Path) -> list[Episode]:
    return load_jsonl(path, Episode)
