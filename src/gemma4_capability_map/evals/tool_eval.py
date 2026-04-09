from __future__ import annotations

from gemma4_capability_map.metrics.answer_match import answer_contains_all
from gemma4_capability_map.metrics.interface_reliability import interface_reliability_score
from gemma4_capability_map.metrics.real_world_metrics import derive_real_world_metrics
from gemma4_capability_map.metrics.trace_metrics import derive_trace_metrics
from gemma4_capability_map.schemas import RunTrace, Task


def score_tool_trace(task: Task, trace: RunTrace) -> dict[str, float | int | bool]:
    expected = [event for event in task.expected_events if event.event_type == "tool_call"]
    actual = trace.tool_steps
    tool_exact = float(len(expected) == len(actual) and all(e.tool_name == a.selected_tool for e, a in zip(expected, actual, strict=False)))
    arg_exact = float(len(expected) == len(actual) and all(e.arguments == a.arguments for e, a in zip(expected, actual, strict=False)))
    if not expected and not actual:
        recovery_correct = 1.0
    else:
        recovery_correct = float(all(step.validator_result == "pass" for step in actual[-len(expected):])) if actual else 0.0
    answer_match = (
        float(answer_contains_all(task.expected_answer_contains, trace.final_answer))
        if task.scoring_profile.answer_match and task.expected_answer_contains
        else 1.0
    )
    metrics = {
        "success": float(tool_exact and arg_exact and recovery_correct and answer_match),
        "tool_exact": tool_exact,
        "arg_exact": arg_exact,
        "recovery_correct": recovery_correct,
        "answer_match": answer_match,
    }
    metrics["interface_reliability_score"] = interface_reliability_score(metrics)
    metrics.update(derive_trace_metrics(task, trace))
    metrics.update(derive_real_world_metrics(task, trace, metrics))
    return metrics
