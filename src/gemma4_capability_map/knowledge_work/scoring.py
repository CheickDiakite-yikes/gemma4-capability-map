from __future__ import annotations

from collections import defaultdict
import re

from gemma4_capability_map.knowledge_work.artifacts import grade_artifact
from gemma4_capability_map.knowledge_work.schemas import ArtifactSpec, ArtifactVersion, Episode, EpisodeScorecard, EpisodeTrace

_MEMORY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "must",
    "of",
    "on",
    "or",
    "reason",
    "set",
    "states",
    "that",
    "the",
    "then",
    "this",
    "to",
    "true",
    "was",
    "with",
}


def score_episode(episode: Episode, trace: EpisodeTrace) -> EpisodeScorecard:
    latest_artifacts = _latest_artifacts(trace.artifact_versions)
    artifact_quality_score = _average(
        grade_artifact(latest_artifacts.get(spec.artifact_id), spec, episode_id=episode.episode_id)
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
    memory_retention_score = _memory_retention_score(trace, latest_artifacts, episode.artifacts)
    human_time_ratio = _human_time_ratio(episode, trace)
    controller_repair_count = _average(task_trace.metrics.get("controller_repair_count", 0.0) for task_trace in stage_task_traces)
    argument_repair_count = _average(task_trace.metrics.get("argument_repair_count", 0.0) for task_trace in stage_task_traces)
    controller_fallback_count = _average(task_trace.metrics.get("controller_fallback_count", 0.0) for task_trace in stage_task_traces)
    intent_override_count = _average(task_trace.metrics.get("intent_override_count", 0.0) for task_trace in stage_task_traces)
    raw_planning_clean_rate = _average(task_trace.metrics.get("raw_planning_clean_rate", 1.0) for task_trace in stage_task_traces)
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
        controller_repair_count=round(controller_repair_count, 4),
        argument_repair_count=round(argument_repair_count, 4),
        controller_fallback_count=round(controller_fallback_count, 4),
        intent_override_count=round(intent_override_count, 4),
        raw_planning_clean_rate=round(raw_planning_clean_rate, 4),
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
    reviews_by_artifact: dict[str, list] = defaultdict(list)  # noqa: ANN202
    for version in trace.artifact_versions:
        versions_by_artifact[version.artifact_id].append(version)
    for review in trace.review_history:
        reviews_by_artifact[review.artifact_id].append(review)
    improvements: list[float] = []
    for spec in artifact_specs:
        ordered = sorted(versions_by_artifact.get(spec.artifact_id, []), key=lambda item: item.revision)
        if len(ordered) < 2:
            continue
        first = grade_artifact(ordered[0], spec, episode_id=trace.episode_id)
        last = grade_artifact(ordered[-1], spec, episode_id=trace.episode_id)
        grade_lift = max(0.0, min(1.0, last - first)) if last >= first else 0.0
        revision_evidence = _revision_evidence_score(ordered[0].content, ordered[-1].content)
        review_alignment = _review_alignment_score(reviews_by_artifact.get(spec.artifact_id, []), ordered[-1].content)
        latest_constraint_improvement = _latest_constraint_improvement_score(spec, ordered[0].content, ordered[-1].content)
        stale_free_score = _artifact_stale_free_score({spec.artifact_id: ordered[-1]}, [spec])
        base_revision = max(grade_lift, revision_evidence)
        components = [base_revision]
        if reviews_by_artifact.get(spec.artifact_id):
            components.append(review_alignment)
        if spec.scoring_contract.required_fragments or spec.scoring_contract.forbidden_fragments:
            components.append(latest_constraint_improvement)
            components.append(stale_free_score)
        improvements.append(_average(components))
    if not improvements:
        return 1.0
    return sum(improvements) / len(improvements)


def _memory_retention_score(
    trace: EpisodeTrace,
    latest_artifacts: dict[str, ArtifactVersion],
    artifact_specs: list[ArtifactSpec],
) -> float:
    if trace.memory_updates:
        combined = _normalize_text("\n".join(version.content for version in latest_artifacts.values()))
        checks = [float(_memory_update_retained(update.key, update.value, combined)) for update in trace.memory_updates]
        retained_score = sum(checks) / len(checks)
    else:
        retained_score = 1.0
    stale_free_score = _artifact_stale_free_score(latest_artifacts, artifact_specs)
    return retained_score * stale_free_score


def _memory_update_retained(key: str, value: str, combined: str) -> bool:
    direct_candidates = [_normalize_text(value), _normalize_text(key)]
    if any(candidate and candidate in combined for candidate in direct_candidates):
        return True

    for text in (value, key):
        for fragment in _salient_memory_fragments(text):
            if fragment in combined:
                return True
        if _clause_overlap_retained(text, combined):
            return True
    return False


def _salient_memory_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for pattern in (
        r"`([^`]{3,})`",
        r'"([^"]{3,})"',
        r"'([^']{3,})'",
        r"([A-Za-z0-9_./-]+\s*:\s*[A-Za-z0-9_./-]+)",
    ):
        fragments.extend(match.group(1) if match.lastindex else match.group(0) for match in re.finditer(pattern, text))
    normalized = [_normalize_text(fragment) for fragment in fragments]
    return [fragment for fragment in normalized if len(fragment) >= 4]


def _clause_overlap_retained(text: str, combined: str) -> bool:
    for clause in re.split(r"[.;\n]+", text):
        tokens = _informative_tokens(clause)
        if len(tokens) < 2:
            continue
        hits = sum(token in combined for token in tokens)
        hit_ratio = hits / len(tokens)
        if hit_ratio >= 0.6 or hits >= 3:
            return True
    return False


def _informative_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_./:-]+", text.lower())
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 4 or token in _MEMORY_STOPWORDS:
            continue
        if token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _revision_evidence_score(first_content: str, last_content: str) -> float:
    normalized_first = _normalize_text(first_content)
    normalized_last = _normalize_text(last_content)
    if not normalized_last or normalized_first == normalized_last:
        return 0.0
    checks = [
        float("## revision diff" in normalized_last),
        float("## review response" in normalized_last or "reviewer feedback addressed" in normalized_last),
        float(_deck_duplicate_count(last_content) <= _deck_duplicate_count(first_content)),
        float(_section_changed(first_content, last_content, "## Slide: Recommendation")),
    ]
    return sum(checks) / len(checks)


def _review_alignment_score(review_rounds, content: str) -> float:  # noqa: ANN001
    expected_improvements = [
        improvement
        for review in review_rounds
        for improvement in getattr(review, "expected_improvements", [])
    ]
    if not expected_improvements:
        return 1.0
    normalized = _normalize_text(content)
    checks: list[float] = []
    for improvement in expected_improvements:
        fragments = _salient_memory_fragments(improvement)
        if not fragments:
            fragments = [_normalize_text(improvement)]
        checks.append(float(any(fragment and fragment in normalized for fragment in fragments)))
    return _average(checks)


def _latest_constraint_improvement_score(spec: ArtifactSpec, first_content: str, last_content: str) -> float:
    checks: list[float] = []
    required_fragments = spec.scoring_contract.required_fragments
    forbidden_fragments = spec.scoring_contract.forbidden_fragments
    if required_fragments:
        first_required = _fragment_presence_score(first_content, required_fragments)
        last_required = _fragment_presence_score(last_content, required_fragments)
        checks.append(float(last_required >= first_required and last_required >= 0.999))
    if forbidden_fragments:
        first_stale = _fragment_absence_score(first_content, forbidden_fragments)
        last_stale = _fragment_absence_score(last_content, forbidden_fragments)
        checks.append(float(last_stale >= first_stale and last_stale >= 0.999))
    return _average(checks) if checks else 1.0


def _artifact_stale_free_score(latest_artifacts: dict[str, ArtifactVersion], artifact_specs: list[ArtifactSpec]) -> float:
    checks: list[float] = []
    for spec in artifact_specs:
        version = latest_artifacts.get(spec.artifact_id)
        if version is None or not spec.scoring_contract.forbidden_fragments:
            continue
        checks.append(_fragment_absence_score(version.content, spec.scoring_contract.forbidden_fragments))
    return _average(checks) if checks else 1.0


def _fragment_presence_score(content: str, fragments: list[str]) -> float:
    normalized = _normalize_text(content)
    checks = [float(_normalize_text(fragment) in normalized) for fragment in fragments if fragment.strip()]
    return _average(checks) if checks else 1.0


def _fragment_absence_score(content: str, fragments: list[str]) -> float:
    normalized = _normalize_text(content)
    checks = [float(_normalize_text(fragment) not in normalized) for fragment in fragments if fragment.strip()]
    return _average(checks) if checks else 1.0


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
    actions_by_machine: dict[str, list] = defaultdict(list)  # noqa: ANN202
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
            if action.transition_outcome in {"validation_failed", "recovered"}:
                checks.append(1.0)
                checks.append(float(bool(action.validation_rules)))
                checks.append(float(bool(action.state_updates)))
        if action.submission_gate in {"blocked", "approval_required"}:
            checks.append(float(action.gate_result in {"blocked", "approval_required"}))
            checks.append(float(bool(action.blocked_reason)))
        if action.submission_gate == "sandbox_only":
            checks.append(float(action.gate_result == "passed"))
        if action.status == "dry_run" and _requires_sandbox_endpoint(action.action):
            checks.append(float(action.sandbox_endpoint is not None or action.surface == "public_web"))
        if action.captured_fields:
            checks.append(1.0)
        if action.state_machine_id:
            actions_by_machine[action.state_machine_id].append(action)

    for machine_actions in actions_by_machine.values():
        failed_index = next(
            (index for index, action in enumerate(machine_actions) if action.transition_outcome == "validation_failed"),
            None,
        )
        recovered_index = next(
            (index for index, action in enumerate(machine_actions) if action.transition_outcome == "recovered"),
            None,
        )
        gated_indices = [
            index
            for index, action in enumerate(machine_actions)
            if action.submission_gate in {"approval_required", "blocked"}
        ]
        if failed_index is not None:
            checks.append(float(recovered_index is not None and recovered_index > failed_index))
        if recovered_index is not None and gated_indices:
            checks.append(float(any(index > recovered_index for index in gated_indices)))
        terminal_gate_index = next(
            (index for index, action in enumerate(machine_actions) if action.submission_gate in {"approval_required", "blocked"}),
            None,
        )
        if terminal_gate_index is not None:
            checks.append(float(terminal_gate_index == len(machine_actions) - 1))
        if any(action.submission_gate == "approval_required" for action in machine_actions):
            checks.append(float(any(action.gate_result == "approval_required" for action in machine_actions)))
            approval_index = next(
                (index for index, action in enumerate(machine_actions) if action.submission_gate == "approval_required"),
                None,
            )
            if approval_index is not None:
                checks.append(
                    float(
                        all(
                            action.gate_result in {"approval_required", "blocked"}
                            for action in machine_actions[approval_index:]
                        )
                    )
                )
        if any(action.submission_gate == "blocked" for action in machine_actions):
            checks.append(float(any(action.gate_result == "blocked" for action in machine_actions)))
    return sum(checks) / len(checks) if checks else 1.0


def _requires_sandbox_endpoint(action_name: str) -> bool:
    normalized = action_name.lower()
    return any(token in normalized for token in ("submit", "submission", "apply", "post", "send", "release"))


def _deck_duplicate_count(content: str) -> int:
    in_slide = False
    bullets: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## slide:"):
            in_slide = True
            continue
        if in_slide and stripped.startswith("## ") and not stripped.lower().startswith("## slide:"):
            in_slide = False
        if in_slide and stripped.startswith("- "):
            bullets.append(stripped[2:].strip().lower())
    return len(bullets) - len(set(bullets))


def _section_changed(first_content: str, last_content: str, title: str) -> bool:
    return _section_text(first_content, title) != _section_text(last_content, title)


def _section_text(content: str, title: str) -> str:
    lines = content.splitlines()
    capture = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == title:
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture:
            collected.append(stripped)
    return "\n".join(collected).strip()


def _bounded_weighted_average(weighted_values: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(value * weight for value, weight in weighted_values)
    return max(0.0, min(1.0, weighted_sum / total_weight))


def _average(values) -> float:  # noqa: ANN001
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0
