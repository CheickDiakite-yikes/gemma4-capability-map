from __future__ import annotations

import os
from ast import literal_eval
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOADED_KEYS: set[tuple[Path, bool]] = set()


def load_project_env(root: Path | None = None, *, force: bool = False, override: bool = False) -> dict[str, str]:
    root_path = (root or _PROJECT_ROOT).resolve()
    cache_key = (root_path, override)
    if cache_key in _LOADED_KEYS and not force:
        return {}

    loaded: dict[str, str] = {}
    for name in (".env.local", ".env"):
        env_path = root_path / name
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
    _LOADED_KEYS.add(cache_key)
    return loaded


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()
    if "=" not in stripped:
        return None
    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _parse_env_value(raw_value.strip())


def _parse_env_value(value: str) -> str:
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            parsed = literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
        return str(parsed)
    return value
