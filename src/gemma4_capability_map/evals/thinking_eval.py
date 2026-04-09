from __future__ import annotations

from gemma4_capability_map.metrics.answer_match import answer_contains_all
from gemma4_capability_map.metrics.real_world_metrics import derive_real_world_metrics
from gemma4_capability_map.metrics.trace_metrics import derive_trace_metrics
from gemma4_capability_map.schemas import RunTrace, Task


def score_thinking_trace(task: Task, trace: RunTrace) -> dict[str, float | int | bool]:
    answer_match = answer_contains_all(task.expected_answer_contains, trace.final_answer)
    metrics = {
        "success": float(answer_match),
        "exact_match": float(answer_match),
        "answer_match": float(answer_match),
        "latency_ms": 0,
    }
    metrics.update(derive_trace_metrics(task, trace))
    metrics.update(derive_real_world_metrics(task, trace, metrics))
    return metrics
