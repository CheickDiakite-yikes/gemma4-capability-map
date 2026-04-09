from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.knowledge_work.schemas import EpisodeTrace


def load_episode_traces(path: str | Path) -> list[EpisodeTrace]:
    return load_jsonl(path, EpisodeTrace)


def summarize_episode_traces(traces: list[EpisodeTrace]) -> dict[str, float]:
    if not traces:
        return {"runs": 0.0}
    return {
        "runs": float(len(traces)),
        "artifact_quality_avg": _average(trace.scorecard.artifact_quality_score for trace in traces),
        "browser_workflow_avg": _average(trace.scorecard.browser_workflow_score for trace in traces),
        "strict_interface_avg": _average(trace.scorecard.strict_interface_score for trace in traces),
        "recovered_execution_avg": _average(trace.scorecard.recovered_execution_score for trace in traces),
        "real_world_readiness_avg": _average(trace.scorecard.role_readiness_score for trace in traces),
        "escalation_correctness_avg": _average(trace.scorecard.escalation_correctness for trace in traces),
    }


def _average(values) -> float:  # noqa: ANN001
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0
