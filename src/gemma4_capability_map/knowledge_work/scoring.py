from __future__ import annotations

from collections import defaultdict

from gemma4_capability_map.knowledge_work.artifacts import grade_artifact
from gemma4_capability_map.knowledge_work.schemas import ArtifactSpec, ArtifactVersion, Episode, EpisodeScorecard, EpisodeTrace


def score_episode(episode: Episode, trace: EpisodeTrace) -> EpisodeScorecard:
    latest_artifacts = _latest_artifacts(trace.artifact_versions)
    artifact_quality_score = _average(
        grade_artifact(latest_artifacts.get(spec.artifact_id), spec)
        for spec in episode.artifacts
    )
    browser_workflow_score = _browser_workflow_score(trace)
    stage_task_traces = [task_trace for stage in trace.stage_traces for task_trace in stage.task_traces]
    strict_interface_score = _average(_strict_interface_score(task_trace) for task_trace in stage_task_traces)
    recovered_execution_score = _average(
        _recovered_execution(task_trace) for task_trace in stage_task_traces
    )
    escalation_correctness = _average(
        task_trace.metrics.get("escalation_correctness", task_trace.metrics.get("answer_match", 1.0))
        for task_trace in stage_task_traces
        if task_trace.real_world_profile and task_trace.real_world_profile.requires_escalation_judgment
    )
    if escalation_correctness == 0.0 and not any(
        task_trace.real_world_profile and task_trace.real_world_profile.requires_escalation_judgment for task_trace in stage_task_traces
    ):
        escalation_correctness = 1.0
    collateral_damage_free = _average(task_trace.metrics.get("collateral_damage_free", 1.0) for task_trace in stage_task_traces)
    revision_responsiveness = _revision_responsiveness(trace, episode.artifacts)
    memory_retention_score = _memory_retention_score(trace, latest_artifacts)
    human_time_ratio = _human_time_ratio(episode, trace)
    role_readiness_score = round(
        _bounded_weighted_average(
            [
                (artifact_quality_score, 0.22),
                (browser_workflow_score, 0.12),
                (strict_interface_score, 0.18),
                (recovered_execution_score, 0.18),
                (revision_responsiveness, 0.1),
                (memory_retention_score, 0.08),
                (escalation_correctness, 0.1),
                (collateral_damage_free, 0.1),
            ]
        ),
        4,
    )
    return EpisodeScorecard(
        artifact_quality_score=round(artifact_quality_score, 4),
        browser_workflow_score=round(browser_workflow_score, 4),
        strict_interface_score=round(strict_interface_score, 4),
        recovered_execution_score=round(recovered_execution_score, 4),
        revision_responsiveness=round(revision_responsiveness, 4),
        memory_retention_score=round(memory_retention_score, 4),
        escalation_correctness=round(escalation_correctness, 4),
        collateral_damage_free=round(collateral_damage_free, 4),
        human_time_ratio=round(human_time_ratio, 4),
        role_readiness_score=role_readiness_score,
    )


def _latest_artifacts(versions: list[ArtifactVersion]) -> dict[str, ArtifactVersion]:
    latest: dict[str, ArtifactVersion] = {}
    for version in versions:
        current = latest.get(version.artifact_id)
        if current is None or version.revision >= current.revision:
            latest[version.artifact_id] = version
    return latest


def _recovered_execution(task_trace) -> float:  # noqa: ANN001
    if "recovered_execution_score" in task_trace.metrics:
        return float(task_trace.metrics["recovered_execution_score"])
    if "final_state_match" in task_trace.metrics:
        return float(task_trace.metrics.get("final_state_match", 0.0)) * float(task_trace.metrics.get("recovery_correct", 1.0))
    return float(task_trace.metrics.get("success", 0.0))


def _strict_interface_score(task_trace) -> float:  # noqa: ANN001
    metrics = task_trace.metrics
    if "interface_reliability_score" in metrics:
        return float(metrics["interface_reliability_score"])
    if "tool_exact" in metrics or "arg_exact" in metrics:
        return float(
            float(metrics.get("tool_exact", 1.0)) >= 1.0
            and float(metrics.get("arg_exact", 1.0)) >= 1.0
        )
    if "recall_at_k" in metrics or "evidence_hit_rate" in metrics:
        return float(
            float(metrics.get("recall_at_k", 1.0)) >= 1.0
            and float(metrics.get("evidence_hit_rate", 1.0)) >= 1.0
        )
    if "answer_match" in metrics:
        return float(float(metrics.get("answer_match", 0.0)) >= 1.0)
    return float(float(metrics.get("success", 0.0)) >= 1.0)


def _revision_responsiveness(trace: EpisodeTrace, artifact_specs: list[ArtifactSpec]) -> float:
    versions_by_artifact: dict[str, list[ArtifactVersion]] = defaultdict(list)
    for version in trace.artifact_versions:
        versions_by_artifact[version.artifact_id].append(version)
    improvements: list[float] = []
    for spec in artifact_specs:
        ordered = sorted(versions_by_artifact.get(spec.artifact_id, []), key=lambda item: item.revision)
        if len(ordered) < 2:
            continue
        first = grade_artifact(ordered[0], spec)
        last = grade_artifact(ordered[-1], spec)
        improvements.append(max(0.0, min(1.0, last - first)) if last >= first else 0.0)
    if not improvements:
        return 1.0
    return sum(improvements) / len(improvements)


def _memory_retention_score(trace: EpisodeTrace, latest_artifacts: dict[str, ArtifactVersion]) -> float:
    if not trace.memory_updates:
        return 1.0
    combined = "\n".join(version.content.lower() for version in latest_artifacts.values())
    checks = [float(update.value.lower() in combined or update.key.lower() in combined) for update in trace.memory_updates]
    return sum(checks) / len(checks)


def _human_time_ratio(episode: Episode, trace: EpisodeTrace) -> float:
    if episode.human_baseline_minutes <= 0:
        return 0.0
    stage_count = max(1, len(trace.stage_traces))
    estimated_agent_minutes = stage_count * 2
    return min(1.5, estimated_agent_minutes / float(episode.human_baseline_minutes))


def _browser_workflow_score(trace: EpisodeTrace) -> float:
    if not trace.browser_actions:
        return 1.0
    checks: list[float] = []
    for action in trace.browser_actions:
        checks.append(float(bool(action.purpose)))
        checks.append(float(bool(action.expected_signal)))
        checks.append(float(bool(action.evidence)))
        checks.append(float(action.verification_result == "pass"))
        if action.validation_rules:
            checks.append(1.0)
        if action.state_updates:
            checks.append(1.0)
        if action.state_machine_id:
            checks.append(float(bool(action.transition_id)))
            checks.append(float(bool(action.from_state)))
            checks.append(float(bool(action.to_state)))
        if action.submission_gate in {"blocked", "approval_required"}:
            checks.append(float(action.gate_result in {"blocked", "approval_required"}))
            checks.append(float(bool(action.blocked_reason)))
        if action.submission_gate == "sandbox_only":
            checks.append(float(action.gate_result == "passed"))
        if action.status == "dry_run" and _requires_sandbox_endpoint(action.action):
            checks.append(float(action.sandbox_endpoint is not None or action.surface == "public_web"))
        if action.captured_fields:
            checks.append(1.0)
    return sum(checks) / len(checks) if checks else 1.0


def _requires_sandbox_endpoint(action_name: str) -> bool:
    normalized = action_name.lower()
    return any(token in normalized for token in ("submit", "submission", "apply", "post", "send", "release"))


def _bounded_weighted_average(weighted_values: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(value * weight for value, weight in weighted_values)
    return max(0.0, min(1.0, weighted_sum / total_weight))


def _average(values) -> float:  # noqa: ANN001
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0
