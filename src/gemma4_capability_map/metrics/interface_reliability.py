from __future__ import annotations


DEFAULT_WEIGHTS = {
    "tool_exact": 0.30,
    "arg_exact": 0.30,
    "recovery_correct": 0.20,
    "final_state_match": 0.20,
}


def interface_reliability_score(metrics: dict[str, float | int | bool]) -> float:
    present = {name: weight for name, weight in DEFAULT_WEIGHTS.items() if name in metrics}
    if not present:
        return 0.0
    total_weight = sum(present.values())
    score = 0.0
    for name, weight in present.items():
        score += float(metrics[name]) * (weight / total_weight)
    return score
