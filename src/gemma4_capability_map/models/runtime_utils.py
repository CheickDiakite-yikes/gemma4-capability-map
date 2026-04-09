from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gemma4_capability_map.env import load_project_env


load_project_env()


_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
_OFFLINE_ENV_VARS = ("GEMMA4_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_MLX_AUTO_MODEL_MAP = {
    "google/gemma-4-e2b-it": "mlx-community/gemma-4-e2b-it-4bit",
    "google/gemma-4-e4b-it": "mlx-community/gemma-4-e4b-it-4bit",
}
_LOCAL_MODEL_PATH_ENV_MAP = {
    "google/gemma-4-e2b-it": "GEMMA4_E2B_PATH",
    "google/gemma-4-e4b-it": "GEMMA4_E4B_PATH",
    "google/functiongemma-270m": "FUNCTIONGEMMA_PATH",
    "google/functiongemma-270m-it": "FUNCTIONGEMMA_PATH",
    "google/embeddinggemma-300m": "EMBEDDINGGEMMA_PATH",
}


def load_hf_auth_token() -> str | None:
    for env_var in _HF_TOKEN_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value
    return None


def detect_hf_token_source() -> str | None:
    for env_var in _HF_TOKEN_ENV_VARS:
        if os.getenv(env_var):
            return env_var
    return None


def is_offline_mode_enabled() -> bool:
    return any(_is_truthy(os.getenv(env_var)) for env_var in _OFFLINE_ENV_VARS)


def resolve_model_source(model_id: str) -> str:
    explicit_path = _existing_path(model_id)
    if explicit_path is not None:
        return explicit_path

    env_var = _LOCAL_MODEL_PATH_ENV_MAP.get(model_id.lower())
    if env_var:
        env_path = _existing_path(os.getenv(env_var))
        if env_path is not None:
            return env_path

    for root_env_var in ("GEMMA_MODEL_ROOT", "GEMMA4_MODEL_ROOT"):
        root = _existing_path(os.getenv(root_env_var))
        if root is None:
            continue
        root_path = Path(root)
        candidates = (
            root_path / model_id.split("/")[-1],
            root_path / model_id.replace("/", "--"),
            root_path / model_id.replace("/", "__"),
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
    return model_id


def should_use_local_files_only(model_source: str) -> bool:
    return is_offline_mode_enabled() or _existing_path(model_source) is not None


def resolve_mlx_model_id(model_id: str) -> str:
    resolved_source = resolve_model_source(model_id)
    if resolved_source != model_id:
        return resolved_source
    return _MLX_AUTO_MODEL_MAP.get(model_id.lower(), model_id)


def probe_mlx_runtime(timeout_seconds: int = 20) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        probe = subprocess.Popen(
            _mlx_probe_command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = probe.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(probe.pid, signal.SIGKILL)
        stdout, stderr = probe.communicate()
        return {
            "ok": False,
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "returncode": probe.returncode,
            "stdout": (stdout or "").strip(),
            "stderr": (stderr or "").strip(),
            "timeout_seconds": timeout_seconds,
            "error": "timeout",
        }
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    stdout = (stdout or "").strip()
    stderr = (stderr or "").strip()
    if probe.returncode == 0:
        try:
            payload = json.loads(stdout.splitlines()[-1]) if stdout else {"ok": False}
        except json.JSONDecodeError:
            payload = {"ok": False, "parse_error": stdout}
        payload.update({"elapsed_ms": elapsed_ms, "returncode": probe.returncode})
        return payload
    return {
        "ok": False,
        "elapsed_ms": elapsed_ms,
        "returncode": probe.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def recommended_local_reasoner_backend(timeout_seconds: int = 20) -> str:
    return recommended_reasoner_backend_from_probe(probe_mlx_runtime(timeout_seconds=timeout_seconds))


def recommended_reasoner_backend_from_probe(probe: dict[str, Any]) -> str:
    return "mlx" if probe.get("ok") else "hf"


def _mlx_probe_command() -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import json\n"
            "import mlx.core as mx\n"
            "payload = {'ok': mx.zeros((1,)).tolist() == [0.0]}\n"
            "print(json.dumps(payload))\n"
        ),
    ]


def _existing_path(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return None


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}
