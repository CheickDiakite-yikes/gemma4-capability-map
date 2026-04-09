from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from gemma4_capability_map.benchmark import ROOT
from gemma4_capability_map.hardware import detect_hardware_profile
from gemma4_capability_map.models.hf_service import read_service_state, service_paths_for
from gemma4_capability_map.models.runtime_utils import (
    detect_hf_token_source,
    is_offline_mode_enabled,
    load_hf_auth_token,
    probe_mlx_runtime,
    recommended_reasoner_backend_from_probe,
    resolve_model_source,
)


def main() -> None:
    output_dir = ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "backend_preflight.json"
    markdown_path = output_dir / "backend_preflight.md"

    mlx_probe = probe_mlx_runtime()
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "hardware": detect_hardware_profile().model_dump(mode="json"),
        "auth": {
            "token_present": bool(load_hf_auth_token()),
            "token_source": detect_hf_token_source(),
        },
        "offline_mode_enabled": is_offline_mode_enabled(),
        "model_sources": {
            "google/gemma-4-E2B-it": resolve_model_source("google/gemma-4-E2B-it"),
            "google/gemma-4-E4B-it": resolve_model_source("google/gemma-4-E4B-it"),
            "google/functiongemma-270m-it": resolve_model_source("google/functiongemma-270m-it"),
            "google/embeddinggemma-300m": resolve_model_source("google/embeddinggemma-300m"),
        },
        "hf_service": _default_service_snapshot(),
        "hf_import_probe": _load_existing_probe(output_dir / "hf_import_probe.json"),
        "specialist_access_probe": _load_existing_probe(output_dir / "specialist_access_probe.json"),
        "mlx_runtime_probe": mlx_probe,
    }
    payload["recommended_local_reasoner_backend"] = recommended_reasoner_backend_from_probe(mlx_probe)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote backend preflight to {json_path}")
    print(f"Wrote backend preflight markdown to {markdown_path}")


def _render_markdown(payload: dict[str, object]) -> str:
    probe = payload["mlx_runtime_probe"]
    assert isinstance(probe, dict)
    model_sources = payload["model_sources"]
    assert isinstance(model_sources, dict)
    lines = [
        "# Backend Preflight",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Recommended local reasoner backend: `{payload['recommended_local_reasoner_backend']}`",
        f"- HF token present: `{payload['auth']['token_present']}`",
        f"- HF token source: `{payload['auth']['token_source']}`",
        f"- Offline mode enabled: `{payload['offline_mode_enabled']}`",
        "",
        "## MLX runtime probe",
        "",
        f"- ok: `{probe.get('ok')}`",
        f"- elapsed_ms: `{probe.get('elapsed_ms')}`",
        f"- returncode: `{probe.get('returncode')}`",
    ]
    if probe.get("stderr"):
        lines.append(f"- stderr: `{str(probe['stderr'])[:300]}`")
    hf_import_probe = payload.get("hf_import_probe")
    if isinstance(hf_import_probe, dict):
        lines.extend(
            [
                "",
                "## HF import probe",
                "",
                f"- ok: `{hf_import_probe.get('ok')}`",
                f"- elapsed_ms: `{hf_import_probe.get('elapsed_ms')}`",
                f"- torch_ms: `{hf_import_probe.get('torch_ms')}`",
                f"- transformers_ms: `{hf_import_probe.get('transformers_ms')}`",
            ]
        )
    specialist_access_probe = payload.get("specialist_access_probe")
    if isinstance(specialist_access_probe, dict):
        lines.extend(
            [
                "",
                "## Specialist access probe",
                "",
            ]
        )
        for row in specialist_access_probe.get("models", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('model')}`: access=`{row.get('access')}` api=`{row.get('api_status')}` config=`{row.get('config_status')}` gated=`{row.get('gated')}`"
            )
    service_snapshot = payload.get("hf_service")
    if isinstance(service_snapshot, dict):
        lines.extend(
            [
                "",
                "## HF reasoner service",
                "",
                f"- status: `{service_snapshot.get('status')}`",
                f"- socket_path: `{service_snapshot.get('paths', {}).get('socket_path')}`",
                f"- state_path: `{service_snapshot.get('paths', {}).get('state_path')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Resolved model sources",
            "",
        ]
    )
    for model_id, source in model_sources.items():
        lines.append(f"- `{model_id}` -> `{source}`")
    return "\n".join(lines) + "\n"


def _default_service_snapshot() -> dict[str, object]:
    paths = service_paths_for("google/gemma-4-E2B-it", "mps")
    state = read_service_state(paths["state_path"]) or {}
    status = state.get("status", "missing")
    pid = state.get("pid")
    alive = _pid_is_running(pid)
    if status != "missing" and not alive and status != "ready":
        status = "stale"
    return {
        "status": status,
        "pid_alive": alive,
        "state": state,
        "paths": paths,
    }


def _load_existing_probe(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pid_is_running(pid: object) -> bool:
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        os.kill(numeric_pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
