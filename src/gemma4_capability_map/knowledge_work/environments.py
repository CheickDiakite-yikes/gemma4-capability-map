from __future__ import annotations

from pathlib import Path


def workspace_root(base: str | Path, workspace_id: str) -> Path:
    return Path(base) / workspace_id


def ensure_workspace(base: str | Path, workspace_id: str) -> Path:
    target = workspace_root(base, workspace_id)
    target.mkdir(parents=True, exist_ok=True)
    return target
