from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.benchmark import load_tasks, load_variants, run_benchmark
from gemma4_capability_map.schemas import RunTrace, Task, Variant
from gemma4_capability_map.traces.exporters import export_leaderboard_csv
from gemma4_capability_map.traces.replay import summarize_traces
from gemma4_capability_map.traces.recorder import TraceRecorder


DEFAULT_TASK_IDS = [
    "think_001_math",
    "think_003_policy_reasoning",
    "think_006_screenshot_security",
    "think_007_doc_image_summary",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["monolith", "hybrid", "modular"], default="monolith")
    parser.add_argument("--backend", choices=["oracle", "heuristic", "hf", "mlx"], default="hf")
    parser.add_argument("--reasoner-backend", choices=["oracle", "heuristic", "hf", "mlx"], default=None)
    parser.add_argument("--reasoner-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--reasoner", action="append", dest="reasoners", default=[])
    parser.add_argument("--router", default="google/functiongemma-270m-it")
    parser.add_argument("--retriever", default="google/embeddinggemma-300m")
    parser.add_argument("--task-id", action="append", dest="task_ids", default=[])
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--output-dir", default="results/alpha_slice")
    args = parser.parse_args()

    task_ids = args.task_ids or DEFAULT_TASK_IDS
    reasoners = args.reasoners or ["google/gemma-4-E2B-it"]
    tasks = load_tasks(track="thinking", task_ids=task_ids)
    variants = load_variants(tasks, include_generated=args.variants)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for reasoner in reasoners:
        safe_model = reasoner.replace("/", "__").replace("-", "_")
        trace_path = output_dir / f"{safe_model}_traces.jsonl"
        leaderboard_path = output_dir / f"{safe_model}_leaderboard.csv"
        summary_path = output_dir / f"{safe_model}_summary.json"
        progress_path = output_dir / f"{safe_model}_progress.json"
        progress_traces: list[RunTrace] = []

        def persist(status: str, completed_runs: int, total_runs: int, error: str | None = None) -> dict[str, Any]:
            if progress_traces:
                recorder = TraceRecorder()
                for trace in progress_traces:
                    recorder.add(trace)
                recorder.write(trace_path)
                export_leaderboard_csv(progress_traces, leaderboard_path)

            summary = _build_summary(progress_traces, model=reasoner)
            summary["status"] = status
            summary["completed_runs"] = completed_runs
            summary["total_runs"] = total_runs
            summary["remaining_runs"] = max(total_runs - completed_runs, 0)
            if progress_traces:
                summary["last_run_id"] = progress_traces[-1].run_id
                summary["last_task_id"] = progress_traces[-1].task_id
                summary["last_variant_id"] = progress_traces[-1].variant_id
            if error:
                summary["error"] = error
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            progress_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        def on_trace(trace: RunTrace, completed_runs: int, total_runs: int) -> None:
            progress_traces.append(trace)
            summary = persist("running", completed_runs=completed_runs, total_runs=total_runs)
            print(
                f"[{reasoner}] completed {completed_runs}/{total_runs} "
                f"task={trace.task_id} variant={trace.variant_id} "
                f"success={trace.metrics.get('success', 0.0)} "
                f"latency_ms={trace.metrics.get('latency_ms', 0)}"
            )
            print(f"[{reasoner}] progress summary: {json.dumps(summary, ensure_ascii=False)}")

        total_runs = _count_planned_runs(tasks, variants)
        persist("starting", completed_runs=0, total_runs=total_runs)
        try:
            traces = run_benchmark(
                tasks=tasks,
                variants=variants,
                pipeline_name=args.pipeline,
                backend=args.backend,
                reasoner_backend=args.reasoner_backend,
                router_backend=None,
                retriever_backend=None,
                reasoner_id=reasoner,
                router_id=args.router,
                retriever_id=args.retriever,
                reasoner_device=args.reasoner_device,
                reasoner_max_new_tokens=args.max_new_tokens,
                planning_max_new_tokens=None,
                final_max_new_tokens=None,
                limit=total_runs,
                thinking_enabled=args.thinking,
                on_trace=on_trace,
            )
        except Exception as exc:
            summary = persist("failed", completed_runs=len(progress_traces), total_runs=total_runs, error=str(exc))
            summary_rows.append(summary)
            raise

        summary = persist("completed", completed_runs=len(traces), total_runs=total_runs)
        summary_rows.append(summary)
        print(f"Wrote {len(traces)} traces for {reasoner} to {trace_path}")

    combined_summary_path = output_dir / "summary.json"
    combined_summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"Wrote combined summary to {combined_summary_path}")


def _build_summary(traces: list[RunTrace], model: str) -> dict[str, object]:
    summary: dict[str, object] = summarize_traces(traces)
    summary["model"] = model
    if not traces:
        summary["avg_latency_ms"] = 0.0
        summary["avg_prompt_tokens"] = 0.0
        summary["avg_completion_tokens"] = 0.0
        return summary

    summary["avg_latency_ms"] = round(sum(float(trace.metrics.get("latency_ms", 0.0)) for trace in traces) / len(traces), 2)
    summary["avg_prompt_tokens"] = round(sum(float(trace.metrics.get("prompt_tokens", 0.0)) for trace in traces) / len(traces), 2)
    summary["avg_completion_tokens"] = round(sum(float(trace.metrics.get("completion_tokens", 0.0)) for trace in traces) / len(traces), 2)
    return summary


def _count_planned_runs(tasks: list[Task], variants: list[Variant]) -> int:
    variant_map: dict[str, int] = {}
    for variant in variants:
        variant_map[variant.base_task_id] = variant_map.get(variant.base_task_id, 0) + 1
    return sum(variant_map.get(task.task_id, 1) for task in tasks)


if __name__ == "__main__":
    main()
