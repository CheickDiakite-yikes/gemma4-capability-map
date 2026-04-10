from __future__ import annotations

from gemma4_capability_map.metrics.answer_match import answer_matches_task, judgment_answer_matches
from gemma4_capability_map.metrics.interface_reliability import interface_reliability_score
from gemma4_capability_map.metrics.real_world_metrics import derive_real_world_metrics
from gemma4_capability_map.metrics.trace_metrics import derive_trace_metrics
from gemma4_capability_map.schemas import RunTrace, Task


def score_visual_trace(task: Task, trace: RunTrace) -> dict[str, float | int | bool]:
    expected_calls = [event for event in task.expected_events if event.event_type == "tool_call"]
    actual_calls = trace.tool_steps
    tool_sequence_exactness = float(
        len(expected_calls) == len(actual_calls)
        and all(expected.tool_name == actual.selected_tool for expected, actual in zip(expected_calls, actual_calls, strict=False))
    )
    argument_exactness = float(_visual_argument_exactness(expected_calls, actual_calls))
    expected_visual = task.expected_final_state.get("visual_selection", {})
    actual_visual = _actual_visual_selection(trace)
    selection_accuracy = _selection_accuracy(expected_visual, actual_visual)
    count_accuracy = _count_accuracy(expected_visual, actual_visual, trace.final_answer)
    refinement_success = _refinement_success(expected_calls, actual_calls, expected_visual, actual_visual)
    referent_retention = _referent_retention(expected_calls, actual_calls)
    stale_selection_recovery = _stale_selection_recovery(expected_calls, actual_calls, referent_retention)
    unnecessary_tool_rate = _unnecessary_tool_rate(expected_calls, actual_calls)
    final_answer_accuracy = float(answer_matches_task(task, trace.final_answer))
    recovery_correct = 1.0 if not actual_calls else float(all(step.validator_result == "pass" for step in actual_calls))

    metrics = {
        "success": float(
            tool_sequence_exactness >= 1.0
            and argument_exactness >= 1.0
            and selection_accuracy >= 0.999
            and count_accuracy >= 1.0
            and referent_retention >= 1.0
            and stale_selection_recovery >= 1.0
            and final_answer_accuracy >= 1.0
            and recovery_correct >= 1.0
        ),
        "tool_exact": tool_sequence_exactness,
        "arg_exact": argument_exactness,
        "recovery_correct": recovery_correct,
        "tool_sequence_exactness": tool_sequence_exactness,
        "selection_accuracy": selection_accuracy,
        "count_accuracy": count_accuracy,
        "refinement_success": refinement_success,
        "referent_retention": referent_retention,
        "stale_selection_recovery": stale_selection_recovery,
        "unnecessary_tool_rate": unnecessary_tool_rate,
        "answer_match": final_answer_accuracy,
        "final_answer_accuracy": final_answer_accuracy,
    }
    if task.judgment_mode and task.judgment_mode.enabled:
        metrics["escalation_correctness"] = float(judgment_answer_matches(task, trace.final_answer))
    metrics["interface_reliability_score"] = interface_reliability_score(metrics)
    metrics.update(derive_trace_metrics(task, trace))
    metrics.update(derive_real_world_metrics(task, trace, metrics))
    return metrics


def _actual_visual_selection(trace: RunTrace) -> dict[str, object]:
    for step in reversed(trace.tool_steps):
        output = step.output or {}
        if "selection_id" in output or "entity_ids" in output or "region_ids" in output:
            return output
    if trace.tool_steps:
        return trace.tool_steps[-1].output or {}
    return {}


def _selection_accuracy(expected_visual: dict[str, object], actual_visual: dict[str, object]) -> float:
    expected_ids = _expected_ids(expected_visual)
    if not expected_ids:
        return 1.0
    actual_ids = _actual_ids(actual_visual)
    if not actual_ids:
        return 0.0
    overlap = len(expected_ids & actual_ids)
    union = len(expected_ids | actual_ids)
    return overlap / union if union else 1.0


def _count_accuracy(expected_visual: dict[str, object], actual_visual: dict[str, object], final_answer: str) -> float:
    expected_count = expected_visual.get("count")
    if expected_count is None:
        return 1.0
    expected_int = int(expected_count)
    actual_count = actual_visual.get("count")
    if actual_count is not None and int(actual_count) == expected_int:
        return 1.0
    return float(str(expected_int) in final_answer)


def _refinement_success(
    expected_calls,
    actual_calls,
    expected_visual: dict[str, object],
    actual_visual: dict[str, object],
) -> float:
    if not any(event.tool_name == "refine_selection" for event in expected_calls):
        return 1.0
    if not any(step.selected_tool == "refine_selection" for step in actual_calls):
        return 0.0
    return float(_selection_accuracy(expected_visual, actual_visual) >= 0.999)


def _stale_selection_recovery(expected_calls, actual_calls, referent_retention: float) -> float:
    if not any(event.tool_name == "refine_selection" for event in expected_calls):
        return 1.0
    selection_progress = [
        int(output.get("count", 0))
        for step in actual_calls
        for output in [step.output or {}]
        if "selection_id" in output and step.selected_tool in {"segment_entities", "extract_layout", "refine_selection"}
    ]
    if len(selection_progress) < 2:
        return 0.0
    return float(selection_progress[-1] < selection_progress[0] and referent_retention >= 1.0)


def _referent_retention(expected_calls, actual_calls) -> float:
    if not any(event.tool_name == "refine_selection" for event in expected_calls):
        return 1.0
    prior_selection_id: str | None = None
    refine_steps = 0
    retained = 0
    for step in actual_calls:
        output = step.output or {}
        if step.selected_tool == "refine_selection":
            refine_steps += 1
            selection_id = str(step.arguments.get("selection_id", ""))
            if selection_id and prior_selection_id and selection_id == prior_selection_id:
                retained += 1
        if step.selected_tool == "refine_selection":
            prior_selection_id = str(output.get("selection_id", prior_selection_id or ""))
        elif step.selected_tool in {"segment_entities", "extract_layout"} and "selection_id" in output:
            prior_selection_id = str(output["selection_id"])
    if refine_steps == 0:
        return 0.0
    return retained / refine_steps


def _unnecessary_tool_rate(expected_calls, actual_calls) -> float:
    if not actual_calls:
        return 0.0
    extra_calls = max(0, len(actual_calls) - len(expected_calls))
    return extra_calls / len(actual_calls)


def _visual_arguments_match(expected_arguments: dict[str, object], actual_arguments: dict[str, object]) -> bool:
    return _visual_arguments_match_with_context(expected_arguments, actual_arguments, [])


def _visual_argument_exactness(expected_calls, actual_calls) -> bool:
    if len(expected_calls) != len(actual_calls):
        return False
    prior_outputs: list[dict[str, object]] = []
    for expected, actual in zip(expected_calls, actual_calls, strict=False):
        if not _visual_arguments_match_with_context(expected.arguments, actual.arguments, prior_outputs):
            return False
        prior_outputs.append(actual.output or {})
    return True


def _visual_arguments_match_with_context(
    expected_arguments: dict[str, object],
    actual_arguments: dict[str, object],
    prior_outputs: list[dict[str, object]],
) -> bool:
    for key, expected_value in expected_arguments.items():
        actual_value = actual_arguments.get(key)
        if expected_value == "$selection":
            if actual_value in (None, ""):
                return False
            if str(actual_value) != _latest_selection_id(prior_outputs):
                return False
            continue
        if expected_value == "$region":
            if actual_value in (None, ""):
                return False
            if str(actual_value) != _latest_region_id(prior_outputs):
                return False
            continue
        if isinstance(expected_value, str) and expected_value.startswith("$"):
            if actual_value in (None, ""):
                return False
            continue
        if actual_value != expected_value:
            return False
    return True


def _expected_ids(expected_visual: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    for key in ("entity_ids", "region_ids"):
        values = expected_visual.get(key, [])
        if isinstance(values, list):
            ids.update(str(value) for value in values)
    return ids


def _actual_ids(actual_visual: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    for key in ("entity_ids", "region_ids"):
        values = actual_visual.get(key, [])
        if isinstance(values, list):
            ids.update(str(value) for value in values)
    return ids


def _latest_selection_id(prior_outputs: list[dict[str, object]]) -> str:
    for output in reversed(prior_outputs):
        selection_id = output.get("selection_id")
        if selection_id not in (None, ""):
            return str(selection_id)
    return ""


def _latest_region_id(prior_outputs: list[dict[str, object]]) -> str:
    for output in reversed(prior_outputs):
        region_id = output.get("region_id")
        if region_id not in (None, ""):
            return str(region_id)
        region_ids = output.get("region_ids")
        if isinstance(region_ids, list) and region_ids:
            return str(region_ids[0])
    return ""
