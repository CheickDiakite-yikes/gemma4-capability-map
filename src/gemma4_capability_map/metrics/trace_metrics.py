from __future__ import annotations

from gemma4_capability_map.schemas import RunTrace, Task


def derive_trace_metrics(task: Task, trace: RunTrace) -> dict[str, float | int | bool]:
    malformed_calls = sum(1 for step in trace.tool_steps if step.validator_result == "fail")
    hallucinated_tools = sum(1 for step in trace.tool_steps if step.selected_tool not in {tool.name for tool in task.tool_specs})
    avoidable_retries = sum(1 for step in trace.tool_steps if step.validator_result == "fail")
    repair_notes = trace.prompt_artifacts.get("planning_repair_notes", [])
    controller_repairs = 0
    for note_batch in repair_notes:
        if isinstance(note_batch, list):
            controller_repairs += len(note_batch)
        elif note_batch:
            controller_repairs += 1
    planning_latencies = [int(value) for value in trace.prompt_artifacts.get("planning_latency_ms", [])]
    planning_prompt_tokens = [int(value) for value in trace.prompt_artifacts.get("planning_prompt_tokens", [])]
    planning_completion_tokens = [int(value) for value in trace.prompt_artifacts.get("planning_completion_tokens", [])]
    final_latency_ms = int(trace.prompt_artifacts.get("final_latency_ms", 0))
    final_prompt_tokens = int(trace.prompt_artifacts.get("final_prompt_tokens", 0))
    final_completion_tokens = int(trace.prompt_artifacts.get("final_completion_tokens", 0))
    return {
        "steps_taken": len(trace.tool_steps),
        "latency_ms": sum(planning_latencies) + final_latency_ms,
        "prompt_tokens": sum(planning_prompt_tokens) + final_prompt_tokens,
        "completion_tokens": sum(planning_completion_tokens) + final_completion_tokens,
        "malformed_call_rate": malformed_calls / len(trace.tool_steps) if trace.tool_steps else 0.0,
        "hallucinated_tool_rate": hallucinated_tools / len(trace.tool_steps) if trace.tool_steps else 0.0,
        "avoidable_retries": avoidable_retries,
        "controller_repair_count": controller_repairs,
    }
