from __future__ import annotations

from gemma4_capability_map.metrics.answer_match import answer_contains_all
from gemma4_capability_map.metrics.final_state import final_state_match
from gemma4_capability_map.metrics.interface_reliability import interface_reliability_score
from gemma4_capability_map.metrics.real_world_metrics import derive_real_world_metrics
from gemma4_capability_map.metrics.trace_metrics import derive_trace_metrics
from gemma4_capability_map.schemas import RunTrace, Task


def score_full_stack_trace(task: Task, trace: RunTrace) -> dict[str, float | int | bool]:
    actual_state = trace.state_transitions[-1].after if trace.state_transitions else task.initial_state
    final_state_score = final_state_match(task.expected_final_state, actual_state)
    answer_match = answer_contains_all(task.expected_answer_contains, trace.final_answer)
    tool_expected = [event for event in task.expected_events if event.event_type == "tool_call"]
    tool_exact = float(len(tool_expected) == len(trace.tool_steps) and all(event.tool_name == step.selected_tool for event, step in zip(tool_expected, trace.tool_steps, strict=False)))
    arg_exact = float(len(tool_expected) == len(trace.tool_steps) and all(event.arguments == step.arguments for event, step in zip(tool_expected, trace.tool_steps, strict=False)))
    if not tool_expected and not trace.tool_steps:
        recovery_correct = 1.0
    else:
        recovery_correct = float(all(step.validator_result == "pass" for step in trace.tool_steps)) if trace.tool_steps else 0.0
    metrics = {
        "success": float(final_state_score and answer_match),
        "tool_exact": tool_exact,
        "arg_exact": arg_exact,
        "recovery_correct": recovery_correct,
        "final_state_match": final_state_score,
        "answer_match": float(answer_match),
    }
    metrics["interface_reliability_score"] = interface_reliability_score(metrics)
    metrics.update(derive_trace_metrics(task, trace))
    metrics.update(derive_real_world_metrics(task, trace, metrics))
    return metrics
