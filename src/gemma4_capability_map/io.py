from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

import yaml

T = TypeVar("T")


def load_jsonl(path: str | Path, model_cls: type[T] | None = None) -> list[T] | list[dict[str, Any]]:
    records: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(model_cls.model_validate(payload) if model_cls else payload)
    return records


def dump_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.model_dump(mode="json", by_alias=True) if hasattr(record, "model_dump") else record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
