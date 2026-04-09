from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    hf_warm = _load_json(ROOT / "results" / "raw" / "warm_harness_hf_google__gemma_4_E2B_it_local_asset.json")
    mlx_warm = _load_json(ROOT / "results" / "raw" / "warm_harness_mlx_google__gemma_4_E2B_it.json")
    hf_alpha = _load_first_summary(ROOT / "results" / "alpha_slice" / "e2b_multimodal_fix" / "summary.json")
    mlx_alpha = _load_first_summary(ROOT / "results" / "alpha_slice" / "e2b_multimodal_mlx_48_v2" / "summary.json")
    e4b_probe = _load_json(ROOT / "results" / "raw" / "hf_smoke_e4b.json")

    report = {
        "decision": {
            "apple_silicon_default_reasoner_backend": "mlx",
            "reason": "MLX E2B matches the validated multimodal alpha slice on this Mac with materially lower load and warm inference latency than HF.",
            "hf_role": "native thinking validation and larger or slower experimental runs",
        },
        "hf_e2b": _warm_summary(hf_warm, hf_alpha),
        "mlx_e2b": _warm_summary(mlx_warm, mlx_alpha),
        "hf_e4b_probe": {
            "status": e4b_probe.get("status"),
            "load_elapsed_ms": _stage_elapsed_ms(e4b_probe, "runner_loaded"),
            "runtime_device": _stage_value(e4b_probe, "runner_loaded", "runtime_device"),
            "load_mode": _stage_value(e4b_probe, "runner_loaded", "load_mode"),
        },
    }

    json_path = ROOT / "results" / "tables" / "local_backend_report.json"
    md_path = ROOT / "results" / "tables" / "local_backend_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_first_summary(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    if isinstance(payload, list) and payload:
        return payload[0]
    return payload


def _warm_summary(warm_payload: dict[str, object], alpha_summary: dict[str, object]) -> dict[str, object]:
    summary = warm_payload["summary"]
    if not isinstance(summary, dict):
        return {}
    text_reasoning = summary.get("text_reasoning", {})
    image_understanding = summary.get("image_understanding", {})
    return {
        "load_elapsed_ms": warm_payload["load"]["elapsed_ms"],
        "warm_text_latency_ms": text_reasoning.get("warm_avg_latency_ms"),
        "warm_image_latency_ms": image_understanding.get("warm_avg_latency_ms"),
        "alpha_success_rate": alpha_summary.get("success_rate"),
        "alpha_avg_latency_ms": alpha_summary.get("avg_latency_ms"),
        "alpha_runs": alpha_summary.get("runs"),
    }


def _stage_elapsed_ms(payload: dict[str, object], stage_name: str) -> int | None:
    for stage in payload.get("stages", []):
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            value = stage.get("elapsed_ms")
            return int(value) if isinstance(value, int | float) else None
    return None


def _stage_value(payload: dict[str, object], stage_name: str, key: str) -> object:
    for stage in payload.get("stages", []):
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            return stage.get(key)
    return None


def _markdown_report(report: dict[str, object]) -> str:
    decision = report["decision"]
    hf_e2b = report["hf_e2b"]
    mlx_e2b = report["mlx_e2b"]
    hf_e4b = report["hf_e4b_probe"]
    return "\n".join(
        [
            "# Local Backend Report",
            "",
            f"Default Apple Silicon local reasoner backend: `{decision['apple_silicon_default_reasoner_backend']}`",
            "",
            f"Reason: {decision['reason']}",
            "",
            f"HF role: {decision['hf_role']}",
            "",
            "## E2B comparison",
            "",
            "| Backend | Load ms | Warm text ms | Warm image ms | Alpha success | Alpha avg latency ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| HF | {hf_e2b['load_elapsed_ms']} | {hf_e2b['warm_text_latency_ms']} | {hf_e2b['warm_image_latency_ms']} | {hf_e2b['alpha_success_rate']} | {hf_e2b['alpha_avg_latency_ms']} |",
            f"| MLX | {mlx_e2b['load_elapsed_ms']} | {mlx_e2b['warm_text_latency_ms']} | {mlx_e2b['warm_image_latency_ms']} | {mlx_e2b['alpha_success_rate']} | {mlx_e2b['alpha_avg_latency_ms']} |",
            "",
            "## E4B local probe",
            "",
            f"- Status: `{hf_e4b['status']}`",
            f"- Load elapsed ms: `{hf_e4b['load_elapsed_ms']}`",
            f"- Runtime device: `{hf_e4b['runtime_device']}`",
            f"- Load mode: `{hf_e4b['load_mode']}`",
        ]
    )


if __name__ == "__main__":
    main()
