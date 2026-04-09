from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import requests

from gemma4_capability_map.benchmark import ROOT
from gemma4_capability_map.models.runtime_utils import detect_hf_token_source, load_hf_auth_token


DEFAULT_MODELS = [
    "google/gemma-4-E2B-it",
    "google/functiongemma-270m-it",
    "google/embeddinggemma-300m",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "tables"))
    parser.add_argument("--model", action="append", dest="models", default=[])
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "specialist_access_probe.json"
    markdown_path = output_dir / "specialist_access_probe.md"

    payload = probe_models(models)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote specialist access probe to {json_path}")
    print(f"Wrote specialist access probe markdown to {markdown_path}")


def probe_models(models: list[str]) -> dict[str, object]:
    token = load_hf_auth_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rows: list[dict[str, object]] = []
    for model in models:
        api_url = f"https://huggingface.co/api/models/{model}"
        api_response = requests.get(api_url, headers=headers, timeout=30)
        model_info = api_response.json() if api_response.headers.get("content-type", "").startswith("application/json") else {}
        config_head = requests.head(
            f"https://huggingface.co/{model}/resolve/main/config.json",
            headers=headers,
            allow_redirects=True,
            timeout=30,
        )
        rows.append(
            {
                "model": model,
                "api_status": api_response.status_code,
                "config_status": config_head.status_code,
                "gated": model_info.get("gated"),
                "private": model_info.get("private"),
                "pipeline_tag": model_info.get("pipeline_tag"),
                "access": classify_access(api_response.status_code, config_head.status_code, model_info),
            }
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "auth_token_present": bool(token),
        "auth_token_source": detect_hf_token_source(),
        "models": rows,
    }


def classify_access(api_status: int, config_status: int, model_info: dict[str, object]) -> str:
    gated = bool(model_info.get("gated"))
    if api_status == 200 and config_status == 200:
        return "available"
    if gated and config_status == 403:
        return "gated_denied"
    if api_status == 401 or config_status == 401:
        return "unauthorized"
    if api_status == 404 or config_status == 404:
        return "not_found"
    if config_status == 403:
        return "forbidden"
    return "unknown"


def render_markdown(payload: dict[str, object]) -> str:
    rows = payload.get("models", [])
    assert isinstance(rows, list)
    lines = [
        "# Specialist Access Probe",
        "",
        f"- Generated at: `{payload.get('generated_at')}`",
        f"- HF token present: `{payload.get('auth_token_present')}`",
        f"- HF token source: `{payload.get('auth_token_source')}`",
        "",
        "| Model | Access | API | config | gated |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(
            f"| `{row.get('model')}` | `{row.get('access')}` | `{row.get('api_status')}` | `{row.get('config_status')}` | `{row.get('gated')}` |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
