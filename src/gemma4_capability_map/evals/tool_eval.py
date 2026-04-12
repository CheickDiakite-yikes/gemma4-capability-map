from __future__ import annotations

from gemma4_capability_map.metrics.answer_match import answer_matches_task, judgment_answer_matches
from gemma4_capability_map.metrics.interface_reliability import interface_reliability_score
from gemma4_capability_map.metrics.real_world_metrics import derive_real_world_metrics
from gemma4_capability_map.metrics.trace_metrics import derive_trace_metrics
from gemma4_capability_map.schemas import RunTrace, Task, Track


_TOOL_FAMILIES = {"function_call", "cli", "api"}
_TOOL_INTENTS = {"inspect", "read", "write", "patch", "search", "execute", "approve", "revise"}


def score_tool_trace(task: Task, trace: RunTrace) -> dict[str, float | int | bool | str]:
    expected = [event for event in task.expected_events if event.event_type == "tool_call"]
    actual = trace.tool_steps
    tool_exact = float(len(expected) == len(actual) and all(e.tool_name == a.selected_tool for e, a in zip(expected, actual, strict=False)))
    arg_exact = float(len(expected) == len(actual) and all(e.arguments == a.arguments for e, a in zip(expected, actual, strict=False)))
    if not expected and not actual:
        recovery_correct = 1.0
    else:
        recovery_correct = float(all(step.validator_result == "pass" for step in actual[-len(expected):])) if actual else 0.0
    answer_match = (
        float(answer_matches_task(task, trace.final_answer))
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
    metrics.update(_tool_taxonomy_metrics(task))
    if task.judgment_mode and task.judgment_mode.enabled:
        metrics["escalation_correctness"] = float(judgment_answer_matches(task, trace.final_answer))
    metrics["interface_reliability_score"] = interface_reliability_score(metrics)
    metrics.update(derive_trace_metrics(task, trace))
    metrics.update(derive_real_world_metrics(task, trace, metrics))
    return metrics


def _tool_taxonomy_metrics(task: Task) -> dict[str, str]:
    explicit_tags = [tag.strip().lower() for tag in task.benchmark_tags if tag and str(tag).strip()]
    family = _extract_tag_value(explicit_tags, "tool_family:") or next((tag for tag in explicit_tags if tag in _TOOL_FAMILIES), "")
    intent = _extract_tag_value(explicit_tags, "tool_intent:") or next((tag for tag in explicit_tags if tag in _TOOL_INTENTS), "")
    source = "explicit" if family or intent else "inferred"
    if not family:
        family = _infer_tool_family(task, explicit_tags)
    if not intent:
        intent = _infer_tool_intent(task, explicit_tags)
    taxonomy = f"{family or 'unknown'}:{intent or 'unknown'}"
    return {
        "tool_family": family or "unknown",
        "tool_intent": intent or "unknown",
        "tool_taxonomy": taxonomy,
        "tool_taxonomy_source": source if source == "explicit" else "inferred",
    }


def _extract_tag_value(tags: list[str], prefix: str) -> str:
    for tag in tags:
        if tag.startswith(prefix):
            return tag.split(":", 1)[1].strip()
    return ""


def _infer_tool_family(task: Task, tags: list[str]) -> str:
    lowered_goal = task.user_goal.lower()
    if any(tag in {"cli", "command_line", "terminal"} for tag in tags) or any(keyword in lowered_goal for keyword in {"cli", "command line", "terminal", "shell", "run the command"}):
        return "cli"
    if any(tag == "api" for tag in tags) or any(keyword in lowered_goal for keyword in {"api", "endpoint", "request", "http"}):
        return "api"
    if task.track in {Track.TOOL_ROUTING, Track.VISUAL_TOOL_ORCHESTRATION, Track.FULL_STACK}:
        return "function_call"
    if task.tool_specs:
        return "function_call"
    return "unknown"


def _infer_tool_intent(task: Task, tags: list[str]) -> str:
    lowered_goal = task.user_goal.lower()
    if any(tag in _TOOL_INTENTS for tag in tags):
        return next(tag for tag in tags if tag in _TOOL_INTENTS)
    if any(keyword in lowered_goal for keyword in {"revise", "refine", "update", "rewrite", "patch", "edit"}):
        return "revise"
    if any(keyword in lowered_goal for keyword in {"approve", "deny", "hold", "review"}):
        return "approve"
    if any(keyword in lowered_goal for keyword in {"execute", "run", "launch", "install", "build"}):
        return "execute"
    if any(keyword in lowered_goal for keyword in {"search", "find", "locate"}):
        return "search"
    if any(keyword in lowered_goal for keyword in {"write", "draft", "compose", "create"}):
        return "write"
    if any(keyword in lowered_goal for keyword in {"read", "inspect", "look at", "show", "summarize", "list"}):
        return "inspect"
    return "read"
