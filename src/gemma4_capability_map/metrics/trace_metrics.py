from __future__ import annotations

from gemma4_capability_map.schemas import RunTrace, Task


def derive_trace_metrics(task: Task, trace: RunTrace) -> dict[str, float | int | bool]:
    malformed_calls = sum(1 for step in trace.tool_steps if step.validator_result == "fail")
    hallucinated_tools = sum(1 for step in trace.tool_steps if step.selected_tool not in {tool.name for tool in task.tool_specs})
    avoidable_retries = sum(1 for step in trace.tool_steps if step.validator_result == "fail")
    repair_notes = trace.prompt_artifacts.get("planning_repair_notes", [])
    normalized_batches: list[list[str]] = []
    for note_batch in repair_notes:
        if isinstance(note_batch, list):
            normalized_batches.append([str(note) for note in note_batch if str(note).strip()])
        elif note_batch:
            normalized_batches.append([str(note_batch)])
        else:
            normalized_batches.append([])
    control_marker_notes = {"controller_repair_disabled", "controller_fallback_disabled"}
    flattened_notes = [note for batch in normalized_batches for note in batch]
    effective_notes = [note for note in flattened_notes if note not in control_marker_notes]
    controller_repairs = len(effective_notes)
    planning_latencies = [int(value) for value in trace.prompt_artifacts.get("planning_latency_ms", [])]
    planning_prompt_tokens = [int(value) for value in trace.prompt_artifacts.get("planning_prompt_tokens", [])]
    planning_completion_tokens = [int(value) for value in trace.prompt_artifacts.get("planning_completion_tokens", [])]
    final_latency_ms = int(trace.prompt_artifacts.get("final_latency_ms", 0))
    final_prompt_tokens = int(trace.prompt_artifacts.get("final_prompt_tokens", 0))
    final_completion_tokens = int(trace.prompt_artifacts.get("final_completion_tokens", 0))
    planning_turn_count = len(normalized_batches)
    planning_turns_with_repairs = sum(1 for batch in normalized_batches if any(note not in control_marker_notes for note in batch))
    controller_fallback_count = sum(1 for note in effective_notes if note == "controller_fallback_planner")
    intent_override_count = sum(1 for note in effective_notes if note.startswith("intent_prior:"))
    argument_repair_count = sum(1 for note in effective_notes if note.startswith("repaired_arguments:"))
    canonicalized_tool_count = sum(1 for note in effective_notes if note.startswith("canonicalized_tool:"))
    return {
        "steps_taken": len(trace.tool_steps),
        "latency_ms": sum(planning_latencies) + final_latency_ms,
        "prompt_tokens": sum(planning_prompt_tokens) + final_prompt_tokens,
        "completion_tokens": sum(planning_completion_tokens) + final_completion_tokens,
        "malformed_call_rate": malformed_calls / len(trace.tool_steps) if trace.tool_steps else 0.0,
        "hallucinated_tool_rate": hallucinated_tools / len(trace.tool_steps) if trace.tool_steps else 0.0,
        "avoidable_retries": avoidable_retries,
        "controller_repair_count": controller_repairs,
        "argument_repair_count": argument_repair_count,
        "canonicalized_tool_count": canonicalized_tool_count,
        "controller_fallback_count": controller_fallback_count,
        "intent_override_count": intent_override_count,
        "planning_turn_count": planning_turn_count,
        "planning_turns_with_repairs": planning_turns_with_repairs,
        "raw_planning_clean_rate": (
            (planning_turn_count - planning_turns_with_repairs) / planning_turn_count
            if planning_turn_count
            else 1.0
        ),
    }
