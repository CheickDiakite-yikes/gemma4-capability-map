from __future__ import annotations

from typing import Any

from gemma4_capability_map.schemas import RunTrace, Task


def derive_real_world_metrics(
    task: Task,
    trace: RunTrace,
    base_metrics: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    profile = task.real_world_profile
    if profile is None:
        return {}

    success = float(base_metrics.get("success", 0.0))
    answer_match = float(base_metrics.get("answer_match", success))
    recovery_correct = float(base_metrics.get("recovery_correct", 1.0 if not trace.tool_steps else 0.0))
    final_state_metric = base_metrics.get("final_state_match")
    if final_state_metric is not None:
        state_integrity = float(final_state_metric)
    elif profile.requires_multistep_state or trace.tool_steps:
        state_integrity = recovery_correct
    else:
        state_integrity = answer_match

    malformed_rate = float(base_metrics.get("malformed_call_rate", 0.0))
    hallucinated_rate = float(base_metrics.get("hallucinated_tool_rate", 0.0))
    avoidable_retries = int(base_metrics.get("avoidable_retries", 0))

    collateral_damage_free = float(
        malformed_rate == 0.0 and hallucinated_rate == 0.0 and state_integrity >= 1.0
    )
    intervention_free_success = float(success >= 1.0 and avoidable_retries == 0)
    recovery_resilience = recovery_correct if profile.requires_recovery else 1.0
    escalation_readiness = answer_match if profile.requires_escalation_judgment else 1.0

    weighted_components: list[tuple[float, float]] = [
        (0.35, success),
        (0.25, state_integrity),
        (0.20, collateral_damage_free),
        (0.10, intervention_free_success),
        (0.10, recovery_resilience if profile.requires_recovery else escalation_readiness),
    ]
    if profile.requires_recovery and profile.requires_escalation_judgment:
        weighted_components[-1] = (0.05, recovery_resilience)
        weighted_components.append((0.05, escalation_readiness))

    total_weight = sum(weight for weight, _ in weighted_components)
    readiness_score = sum(weight * value for weight, value in weighted_components) / total_weight

    metrics: dict[str, float | int | bool] = {
        "state_integrity_score": round(state_integrity, 4),
        "collateral_damage_free": round(collateral_damage_free, 4),
        "intervention_free_success": round(intervention_free_success, 4),
        "recovery_resilience": round(recovery_resilience, 4),
        "escalation_readiness": round(escalation_readiness, 4),
        "real_world_readiness_score": round(readiness_score, 4),
    }
    if profile.human_equivalent_minutes:
        human_ms = profile.human_equivalent_minutes * 60_000
        latency_ms = float(base_metrics.get("latency_ms", 0.0))
        ratio = latency_ms / human_ms if human_ms else 0.0
        metrics["human_time_ratio"] = round(ratio, 4)
        metrics["faster_than_human"] = float(ratio <= 1.0)
    return metrics
