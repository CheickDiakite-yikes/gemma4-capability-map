from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gemma4_capability_map.assets import default_benchmark_image_path
from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.models.runtime_utils import detect_hf_token_source, load_hf_auth_token, resolve_mlx_model_id
from gemma4_capability_map.schemas import Message
from gemma4_capability_map.hardware import detect_hardware_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "hf_service", "mlx"], default="hf")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--image-path", default=default_benchmark_image_path())
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    requested_model = args.model
    effective_model = resolve_mlx_model_id(requested_model) if args.backend == "mlx" else requested_model
    output_path = Path(args.output_path or _default_output_path(args.backend))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    auth_token = load_hf_auth_token()
    results: dict[str, object] = {
        "backend": args.backend,
        "requested_model": requested_model,
        "effective_model": effective_model,
        "device": args.device,
        "checks": [],
        "stages": [],
        "status": "starting",
        "hardware": detect_hardware_profile().model_dump(mode="json"),
        "auth": {
            "token_present": bool(auth_token),
            "token_source": detect_hf_token_source(),
        },
    }
    _write_progress(output_path, results)

    runner = Gemma4Runner(requested_model, backend=args.backend, max_new_tokens=args.max_new_tokens, device=args.device)
    _record_stage(results, output_path, "runner_initialized", started_at=time.perf_counter(), extra=runner.runtime_info())
    results["status"] = "runner_initialized"
    _write_progress(output_path, results)
    load_started = time.perf_counter()
    image_ref = str(Path(args.image_path).expanduser().resolve())
    try:
        runtime = runner.ensure_loaded(media=[] if args.skip_image else [image_ref])
        _record_stage(results, output_path, "runner_loaded", started_at=load_started, extra=runtime)
        results["status"] = "runner_loaded"
    except Exception as exc:
        _record_stage(
            results,
            output_path,
            "runner_load_failed",
            started_at=load_started,
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        results["status"] = "failed"
        results["error"] = f"{type(exc).__name__}: {exc}"
        _write_progress(output_path, results)
        print(json.dumps(results, indent=2))
        return
    _write_progress(output_path, results)

    if args.probe_only:
        results["status"] = "completed"
        _write_progress(output_path, results)
        print(json.dumps(results, indent=2))
        return

    checks = [
        {
            "name": "text_reasoning",
            "messages": [Message(role="system", content="You are a concise assistant."), Message(role="user", content="What is 19 plus 23?")],
            "media": [],
            "thinking": args.thinking,
        }
    ]
    if not args.skip_image:
        checks.append(
            {
                "name": "image_understanding",
                "messages": [Message(role="user", content="What security setting is disabled in this screenshot? Answer in one sentence.")],
                "media": [image_ref],
                "thinking": False,
            }
        )

    for check in checks:
        print(f"running {check['name']}...", flush=True)
        results["status"] = f"running:{check['name']}"
        _write_progress(output_path, results)
        started = time.perf_counter()
        try:
            turn = runner.generate(
                messages=check["messages"],
                media=check["media"],
                tool_specs=[],
                thinking=bool(check["thinking"]),
            )
            results["checks"].append(
                {
                    "name": check["name"],
                    "ok": True,
                    "latency_ms": turn.latency_ms,
                    "prompt_tokens": turn.prompt_tokens,
                    "completion_tokens": turn.completion_tokens,
                    "thinking_text": turn.thinking_text,
                    "final_answer": turn.final_answer or turn.raw_model_output,
                }
            )
        except Exception as exc:
            results["checks"].append(
                {
                    "name": check["name"],
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        _record_stage(
            results,
            output_path,
            f"check:{check['name']}",
            started_at=started,
            extra={"ok": bool(results["checks"][-1]["ok"])},
        )
        _write_progress(output_path, results)

    results["status"] = "completed"
    _write_progress(output_path, results)
    print(json.dumps(results, indent=2))


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_output_path(backend: str) -> str:
    if backend == "mlx":
        return "results/raw/mlx_smoke.json"
    return "results/raw/hf_smoke.json"


def _record_stage(
    results: dict[str, object],
    output_path: Path,
    name: str,
    started_at: float,
    extra: dict[str, object] | None = None,
) -> None:
    stages = results.setdefault("stages", [])
    if not isinstance(stages, list):
        return
    payload = {
        "name": name,
        "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
    }
    if extra:
        payload.update(extra)
    stages.append(payload)
    _write_progress(output_path, results)


if __name__ == "__main__":
    main()
