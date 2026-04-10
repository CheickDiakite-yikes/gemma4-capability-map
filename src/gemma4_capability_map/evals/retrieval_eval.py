from __future__ import annotations

from gemma4_capability_map.metrics.answer_match import answer_matches_task, judgment_answer_matches
from gemma4_capability_map.metrics.real_world_metrics import derive_real_world_metrics
from gemma4_capability_map.metrics.retrieval_metrics import ndcg, recall_at_k, reciprocal_rank
from gemma4_capability_map.metrics.trace_metrics import derive_trace_metrics
from gemma4_capability_map.schemas import RunTrace, Task


def score_retrieval_trace(task: Task, trace: RunTrace) -> dict[str, float | int | bool]:
    expected_events = [event for event in task.expected_events if event.event_type == "retrieval"]
    expected_doc_ids = expected_events[0].expected_doc_ids if expected_events else []
    actual_doc_ids = [hit.doc_id for hit in trace.retrieval_hits]
    answer_match = answer_matches_task(task, trace.final_answer)
    metrics = {
        "success": float(answer_match),
        "recall_at_k": recall_at_k(expected_doc_ids, actual_doc_ids),
        "mrr": reciprocal_rank(expected_doc_ids, actual_doc_ids),
        "ndcg": ndcg(expected_doc_ids, actual_doc_ids),
        "evidence_hit_rate": recall_at_k(expected_doc_ids, actual_doc_ids),
        "answer_match": float(answer_match),
        "token_budget_used": len(trace.prompt_artifacts.get("retrieved_doc_ids", [])),
    }
    if task.judgment_mode and task.judgment_mode.enabled:
        metrics["escalation_correctness"] = float(judgment_answer_matches(task, trace.final_answer))
    metrics.update(derive_trace_metrics(task, trace))
    metrics.update(derive_real_world_metrics(task, trace, metrics))
    return metrics
