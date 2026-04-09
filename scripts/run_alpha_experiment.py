from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gemma4_capability_map.benchmark import (
    ROOT,
    build_runtime_bundle,
    load_tasks,
    load_variants,
    plan_runs,
    runtime_bundle_snapshot,
    run_benchmark,
    warm_runtime_bundle,
)
from gemma4_capability_map.io import load_yaml
from gemma4_capability_map.schemas import RunTrace, Task, Variant
from gemma4_capability_map.traces.exporters import export_leaderboard_csv
from gemma4_capability_map.traces.recorder import TraceRecorder
from gemma4_capability_map.traces.replay import load_traces
from gemma4_capability_map.traces.replay import summarize_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "alpha_matrix.yaml"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-group-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    matrix = config["matrix"]
    experiment = next(item for item in matrix["experiments"] if item["id"] == args.experiment_id)
    summary, _ = run_experiment(
        run_group_id=args.run_group_id,
        matrix_name=matrix["name"],
        output_dir=Path(args.output_dir),
        experiment=experiment,
    )
    print(json.dumps(summary, indent=2))


def run_experiment(
    run_group_id: str,
    matrix_name: str,
    output_dir: Path,
    experiment: dict[str, Any],
) -> tuple[dict[str, Any], list[RunTrace]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(track=experiment.get("track"), task_ids=experiment.get("task_ids"))
    variants = load_variants(tasks, include_generated=bool(experiment.get("variants", False)))
    planned_runs = plan_runs(tasks, variants)
    total_runs = min(int(experiment.get("limit", len(planned_runs))), len(planned_runs))
    selected_pairs = planned_runs[:total_runs]
    traces_path = output_dir / "traces.jsonl"
    traces_so_far: list[RunTrace] = load_traces(traces_path) if traces_path.exists() else []
    completed_variant_ids = {trace.variant_id for trace in traces_so_far}
    remaining_pairs = [pair for pair in selected_pairs if pair[1].variant_id not in completed_variant_ids]
    warmup_details: dict[str, Any] = {}
    bundle_runtime: dict[str, Any] = {}

    def persist(status: str, error: str | None = None) -> dict[str, Any]:
        if traces_so_far:
            recorder = TraceRecorder()
            for trace in traces_so_far:
                recorder.add(trace)
            recorder.write(output_dir / "traces.jsonl")
            export_leaderboard_csv(traces_so_far, output_dir / "leaderboard.csv")
        summary = _build_summary(traces_so_far)
        summary.update(
            {
                "run_group_id": run_group_id,
                "matrix_name": matrix_name,
                "experiment_id": experiment["id"],
                "execution_mode": experiment.get("execution_mode", "subprocess"),
                "pipeline": experiment["pipeline"],
                "track": experiment.get("track"),
                "backend": experiment["backend"],
                "reasoner_backend": experiment.get("reasoner_backend") or experiment["backend"],
                "router_backend": experiment.get("router_backend"),
                "retriever_backend": experiment.get("retriever_backend"),
                "reasoner": experiment["reasoner"],
                "router": experiment.get("router"),
                "retriever": experiment.get("retriever"),
                "max_new_tokens": experiment.get("max_new_tokens"),
                "planning_max_new_tokens": experiment.get("planning_max_new_tokens"),
                "final_max_new_tokens": experiment.get("final_max_new_tokens"),
                "thinking_enabled": bool(experiment.get("thinking", False)),
                "variants": bool(experiment.get("variants", False)),
                "status": status,
                "completed_runs": len(traces_so_far),
                "total_runs": total_runs,
                "remaining_runs": max(total_runs - len(traces_so_far), 0),
                "notes": experiment.get("notes", []),
            }
        )
        if warmup_details:
            summary["warmup"] = warmup_details
        if bundle_runtime:
            summary["runtime_bundle"] = bundle_runtime
        if traces_so_far:
            summary["last_run_id"] = traces_so_far[-1].run_id
            summary["last_task_id"] = traces_so_far[-1].task_id
            summary["last_variant_id"] = traces_so_far[-1].variant_id
        if error:
            summary["error"] = error
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "progress.json", summary)
        return summary

    def on_trace(trace: RunTrace, completed_runs: int, planned_runs: int) -> None:
        traces_so_far.append(trace)
        summary = persist("running")
        print(
            f"[{experiment['id']}] completed {len(traces_so_far)}/{total_runs} "
            f"task={trace.task_id} success={trace.metrics.get('success', 0.0)} "
            f"latency_ms={trace.metrics.get('latency_ms', 0)}",
            flush=True,
        )
        print(f"[{experiment['id']}] progress summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)

    persist("starting")
    if not remaining_pairs:
        summary = persist("completed")
        return summary, traces_so_far

    remaining_tasks = list({task.task_id: task for task, _ in remaining_pairs}.values())
    remaining_variants = [variant for _, variant in remaining_pairs]
    bundle = build_runtime_bundle(
        tasks=remaining_tasks,
        pipeline_name=experiment["pipeline"],
        backend=experiment["backend"],
        reasoner_backend=experiment.get("reasoner_backend"),
        router_backend=experiment.get("router_backend"),
        retriever_backend=experiment.get("retriever_backend"),
        reasoner_id=experiment["reasoner"],
        router_id=experiment.get("router", "google/functiongemma-270m-it"),
        retriever_id=experiment.get("retriever", "google/embeddinggemma-300m"),
        reasoner_device=experiment.get("reasoner_device", "auto"),
        reasoner_max_new_tokens=int(experiment.get("max_new_tokens", 64)),
    )
    bundle_runtime = runtime_bundle_snapshot(bundle)

    try:
        persist("warming_runtime")
        warmup_details = warm_runtime_bundle(bundle, remaining_tasks)
        bundle_runtime = runtime_bundle_snapshot(bundle)
        persist("running")
        traces = run_benchmark(
            tasks=remaining_tasks,
            variants=remaining_variants,
            pipeline_name=experiment["pipeline"],
            backend=experiment["backend"],
            reasoner_backend=experiment.get("reasoner_backend"),
            router_backend=experiment.get("router_backend"),
            retriever_backend=experiment.get("retriever_backend"),
            reasoner_id=experiment["reasoner"],
            router_id=experiment.get("router", "google/functiongemma-270m-it"),
            retriever_id=experiment.get("retriever", "google/embeddinggemma-300m"),
            reasoner_device=experiment.get("reasoner_device", "auto"),
            reasoner_max_new_tokens=int(experiment.get("max_new_tokens", 64)),
            planning_max_new_tokens=experiment.get("planning_max_new_tokens"),
            final_max_new_tokens=experiment.get("final_max_new_tokens"),
            limit=len(remaining_pairs),
            thinking_enabled=bool(experiment.get("thinking", False)),
            on_trace=on_trace,
            bundle=bundle,
        )
    except Exception as exc:
        summary = persist("failed", error=f"{type(exc).__name__}: {exc}")
        return summary, traces_so_far

    summary = persist("completed")
    return summary, traces


def _build_summary(traces: list[RunTrace]) -> dict[str, Any]:
    summary: dict[str, Any] = summarize_traces(traces)
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
    return len(plan_runs(tasks, variants))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
