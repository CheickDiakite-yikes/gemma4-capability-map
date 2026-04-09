from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.metrics.failure_taxonomy import failure_tags, summarize_failure_tags
from gemma4_capability_map.schemas import RunTrace


def load_traces(path: str | Path) -> list[RunTrace]:
    return load_jsonl(path, RunTrace)


def summarize_traces(traces: list[RunTrace]) -> dict[str, float | dict[str, int] | dict[str, float] | list[dict[str, object]]]:
    if not traces:
        return {"runs": 0.0, "metric_averages": {}, "failing_variants": []}
    successes = [float(trace.metrics.get("success", 0.0)) for trace in traces]
    summary: dict[str, float | dict[str, int] | dict[str, float] | list[dict[str, object]]] = {
        "runs": float(len(traces)),
        "success_rate": sum(successes) / len(successes),
        "strict_interface_rate": _average(_strict_interface_outcome(trace) for trace in traces),
        "recovered_execution_rate": _average(_recovered_execution_outcome(trace) for trace in traces),
    }
    readiness_values = [float(trace.metrics.get("real_world_readiness_score", 0.0)) for trace in traces if "real_world_readiness_score" in trace.metrics]
    if readiness_values:
        summary["real_world_readiness_avg"] = _average(readiness_values)
    summary["failure_breakdown"] = summarize_failure_tags(traces)
    summary["metric_averages"] = _metric_averages(traces)
    summary["failing_variants"] = _failing_variants(traces)
    return summary


def _metric_averages(traces: list[RunTrace]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for trace in traces:
        for key, value in trace.metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {
        key: round(totals[key] / counts[key], 4)
        for key in sorted(totals)
        if counts.get(key, 0) > 0
    }


def _failing_variants(traces: list[RunTrace], limit: int = 10) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for trace in traces:
        if float(trace.metrics.get("success", 0.0)) >= 1.0:
            continue
        failures.append(
            {
                "task_id": trace.task_id,
                "variant_id": trace.variant_id,
                "failure_tags": failure_tags(trace),
                "latency_ms": float(trace.metrics.get("latency_ms", 0.0)),
                "interface_reliability_score": float(trace.metrics.get("interface_reliability_score", 0.0)),
                "controller_repair_count": int(trace.metrics.get("controller_repair_count", 0)),
            }
        )
    return failures[:limit]


def _average(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _strict_interface_outcome(trace: RunTrace) -> float:
    metrics = trace.metrics
    if "tool_exact" in metrics or "arg_exact" in metrics:
        return float(float(metrics.get("tool_exact", 1.0)) >= 1.0 and float(metrics.get("arg_exact", 1.0)) >= 1.0)
    if "recall_at_k" in metrics or "evidence_hit_rate" in metrics:
        return float(float(metrics.get("recall_at_k", 1.0)) >= 1.0 and float(metrics.get("evidence_hit_rate", 1.0)) >= 1.0)
    if "answer_match" in metrics:
        return float(float(metrics.get("answer_match", 0.0)) >= 1.0)
    return float(float(metrics.get("success", 0.0)) >= 1.0)


def _recovered_execution_outcome(trace: RunTrace) -> float:
    metrics = trace.metrics
    if "final_state_match" in metrics:
        recovery = float(metrics.get("recovery_correct", 1.0))
        return float(float(metrics.get("final_state_match", 0.0)) >= 1.0 and recovery >= 1.0)
    if "success" in metrics:
        return float(float(metrics.get("success", 0.0)) >= 1.0)
    if "answer_match" in metrics:
        return float(float(metrics.get("answer_match", 0.0)) >= 1.0)
    return 0.0
