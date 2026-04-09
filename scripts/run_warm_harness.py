from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from gemma4_capability_map.assets import default_benchmark_image_path
from gemma4_capability_map.hardware import detect_hardware_profile
from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.models.runtime_utils import detect_hf_token_source, load_hf_auth_token, resolve_mlx_model_id
from gemma4_capability_map.schemas import Message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "hf_service", "mlx"], default="hf")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--image-path", default=default_benchmark_image_path())
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    requested_model = args.model
    effective_model = resolve_mlx_model_id(requested_model) if args.backend == "mlx" else requested_model
    output_path = Path(args.output_path or _default_output_path(args.backend, effective_model))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "backend": args.backend,
        "requested_model": requested_model,
        "effective_model": effective_model,
        "device": args.device,
        "repeats": args.repeats,
        "hardware": detect_hardware_profile().model_dump(mode="json"),
        "auth": {
            "token_present": bool(load_hf_auth_token()),
            "token_source": detect_hf_token_source(),
        },
        "load": {},
        "checks": [],
        "summary": {},
    }

    runner = Gemma4Runner(requested_model, backend=args.backend, max_new_tokens=args.max_new_tokens, device=args.device)
    load_started = time.perf_counter()
    image_ref = str(Path(args.image_path).expanduser().resolve())
    runtime = runner.ensure_loaded(media=[] if args.skip_image else [image_ref])
    payload["load"] = {
        "elapsed_ms": int((time.perf_counter() - load_started) * 1000),
        **runtime,
    }

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

    summary: dict[str, object] = {}
    for check in checks:
        runs = []
        for index in range(args.repeats):
            turn = runner.generate(
                messages=check["messages"],
                media=check["media"],
                tool_specs=[],
                thinking=bool(check["thinking"]),
            )
            runs.append(
                {
                    "iteration": index + 1,
                    "latency_ms": turn.latency_ms,
                    "prompt_tokens": turn.prompt_tokens,
                    "completion_tokens": turn.completion_tokens,
                    "final_answer": turn.final_answer or turn.raw_model_output,
                }
            )
        payload["checks"].append({"name": check["name"], "runs": runs})
        latencies = [int(run["latency_ms"]) for run in runs]
        warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
        summary[check["name"]] = {
            "first_latency_ms": latencies[0],
            "warm_avg_latency_ms": round(sum(warm_latencies) / len(warm_latencies), 2),
            "warm_median_latency_ms": round(statistics.median(warm_latencies), 2),
            "best_latency_ms": min(latencies),
        }

    payload["summary"] = summary
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _default_output_path(backend: str, model_id: str) -> str:
    safe_model = model_id.replace("/", "__").replace("-", "_")
    return f"results/raw/warm_harness_{backend}_{safe_model}.json"


if __name__ == "__main__":
    main()
