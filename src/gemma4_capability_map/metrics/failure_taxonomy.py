from __future__ import annotations

from collections import Counter

from gemma4_capability_map.schemas import RunTrace


def failure_tags(trace: RunTrace) -> list[str]:
    tags: list[str] = []
    success = float(trace.metrics.get("success", 0.0)) >= 1.0
    final_answer = trace.final_answer.strip()
    final_raw_output = str(trace.prompt_artifacts.get("final_raw_output", ""))
    final_thinking_text = str(trace.prompt_artifacts.get("final_thinking_text", ""))
    final_completion_tokens = int(trace.prompt_artifacts.get("final_completion_tokens", 0) or 0)
    final_max_new_tokens = trace.prompt_artifacts.get("final_max_new_tokens")
    answer_match = float(trace.metrics.get("answer_match", 1.0))

    if not success:
        tags.append("failed")
    if float(trace.metrics.get("malformed_call_rate", 0.0)) > 0.0:
        tags.append("malformed_call")
    if float(trace.metrics.get("hallucinated_tool_rate", 0.0)) > 0.0:
        tags.append("hallucinated_tool")
    if "final_state_match" in trace.metrics and float(trace.metrics.get("final_state_match", 1.0)) < 1.0:
        tags.append("wrong_final_state")
    if "tool_exact" in trace.metrics and float(trace.metrics.get("tool_exact", 1.0)) < 1.0:
        tags.append("wrong_tool")
    if "arg_exact" in trace.metrics and float(trace.metrics.get("arg_exact", 1.0)) < 1.0:
        tags.append("arg_mismatch")
    if "recall_at_k" in trace.metrics and float(trace.metrics.get("recall_at_k", 1.0)) < 1.0:
        tags.append("retrieval_miss")

    if not final_answer:
        tags.append("answer_missing")
    elif answer_match < 1.0:
        tags.append("answer_mismatch")

    if (
        final_max_new_tokens not in {None, 0}
        and final_completion_tokens >= int(final_max_new_tokens)
        and not final_answer
    ):
        tags.append("generation_truncated")

    if trace.thinking_enabled and not final_answer and (final_thinking_text or "<|channel>thought" in final_raw_output):
        tags.append("thinking_overflow")

    if trace.image_refs and answer_match < 1.0 and final_answer and not trace.tool_steps:
        tags.append("image_grounding_miss")

    if not success and tags == ["failed"]:
        tags.append("uncategorized_failure")

    return tags


def summarize_failure_tags(traces: list[RunTrace]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for trace in traces:
        counter.update(failure_tags(trace))
    return dict(sorted(counter.items()))
