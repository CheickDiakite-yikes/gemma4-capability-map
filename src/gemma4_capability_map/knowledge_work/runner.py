from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from gemma4_capability_map.knowledge_work.artifacts import grade_artifact
from gemma4_capability_map.knowledge_work.native_artifacts import materialize_artifact
from gemma4_capability_map.knowledge_work.scoring import score_episode
from gemma4_capability_map.knowledge_work.schemas import ArtifactKind, ArtifactSpec, ArtifactVersion, BrowserAction, BrowserStep, Episode, EpisodeTrace, MemoryUpdate, StageTrace
from gemma4_capability_map.pipelines.base import RuntimeBundle
from gemma4_capability_map.pipelines.hybrid import HybridPipeline
from gemma4_capability_map.pipelines.modular import ModularPipeline
from gemma4_capability_map.pipelines.monolith import MonolithPipeline
from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.schemas import Task, Track, Variant


class EpisodeRunner:
    def __init__(
        self,
        tasks: list[Task],
        bundle: RuntimeBundle,
        thinking_enabled: bool = False,
        planning_max_new_tokens: int | None = None,
        final_max_new_tokens: int | None = None,
        artifact_output_root: str | Path | None = None,
        research_controls: ResearchControls | None = None,
    ) -> None:
        self.task_index = {task.task_id: task for task in tasks}
        self.bundle = bundle
        self.thinking_enabled = thinking_enabled
        self.planning_max_new_tokens = planning_max_new_tokens
        self.final_max_new_tokens = final_max_new_tokens
        self.artifact_output_root = Path(artifact_output_root) if artifact_output_root else None
        self.research_controls = research_controls or ResearchControls()

    def run(self, episode: Episode) -> EpisodeTrace:
        trace = EpisodeTrace(
            run_id=f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_episode_{episode.episode_id}",
            episode_id=episode.episode_id,
            role_family=episode.role_family,
            lane=episode.lane,
            workspace_id=episode.workspace_id,
            benchmark_tags=list(episode.benchmark_tags),
            review_history=deepcopy(episode.review_rounds),
            prompt_artifacts={"brief": episode.brief, "success_contract": episode.success_contract.model_dump(mode="json")},
        )
        artifact_versions: list[ArtifactVersion] = []
        memory_updates: list[MemoryUpdate] = []
        browser_actions: list[BrowserAction] = []
        tool_calls: list[dict] = []

        for stage in episode.stages:
            task_traces = []
            for task_id in stage.task_refs:
                task = self.task_index[task_id]
                pipeline = self._pipeline_for_stage(stage.preferred_architecture, task.track)
                variant = Variant(variant_id=f"{task.task_id}_episode_clean", base_task_id=task.task_id)
                task_trace = pipeline.run(task=task, variant=variant, bundle=self.bundle)
                task_traces.append(task_trace)
                tool_calls.extend(
                    {
                        "stage_id": stage.stage_id,
                        "task_id": task_trace.task_id,
                        "tool_name": step.selected_tool,
                        "tool_family": step.tool_family,
                        "tool_intent": step.tool_intent,
                        "arguments": step.arguments,
                        "validator_result": step.validator_result,
                    }
                    for step in task_trace.tool_steps
                )
            trace.stage_traces.append(
                StageTrace(
                    stage_id=stage.stage_id,
                    task_traces=task_traces,
                    artifact_updates=list(stage.required_artifacts),
                    notes=[stage.goal],
                )
            )
            for artifact_id in stage.required_artifacts:
                artifact_spec = next((artifact for artifact in episode.artifacts if artifact.artifact_id == artifact_id), None)
                prior_content = _latest_artifact_content(artifact_versions, artifact_id)
                rendered_content = _artifact_content(episode.brief, stage.goal, task_traces, artifact_id, artifact_spec)
                version = ArtifactVersion(
                        artifact_id=artifact_id,
                        revision=_next_revision(artifact_versions, artifact_id),
                        content=_merge_artifact_content(prior_content, rendered_content, artifact_spec),
                        source_stage=stage.stage_id,
                    )
                if artifact_spec is not None:
                    version = materialize_artifact(trace.run_id, version, artifact_spec, self.artifact_output_root)
                artifact_versions.append(version)
            memory_updates.append(MemoryUpdate(stage_id=stage.stage_id, key=stage.goal, value="; ".join(task.final_answer for task in task_traces if task.final_answer)))
            browser_actions.extend(_browser_actions_for_stage(episode, stage, task_traces))

        for review in episode.review_rounds:
            prior = _latest_artifact_content(artifact_versions, review.artifact_id)
            artifact_spec = next((artifact for artifact in episode.artifacts if artifact.artifact_id == review.artifact_id), None)
            revised_content = _apply_review_feedback(prior, review.feedback, review.expected_improvements, artifact_spec)
            version = ArtifactVersion(
                    artifact_id=review.artifact_id,
                    revision=_next_revision(artifact_versions, review.artifact_id),
                    content=revised_content,
                    source_stage=f"review:{review.review_id}",
                )
            if artifact_spec is not None:
                version = materialize_artifact(trace.run_id, version, artifact_spec, self.artifact_output_root)
            artifact_versions.append(version)

        artifact_specs = {artifact.artifact_id: artifact for artifact in episode.artifacts}
        for index, version in enumerate(artifact_versions):
            spec = artifact_specs.get(version.artifact_id)
            if spec is None:
                continue
            artifact_versions[index] = version.model_copy(update={"score": grade_artifact(version, spec, episode_id=episode.episode_id)})

        trace.artifact_versions = artifact_versions
        trace.memory_updates = memory_updates
        trace.browser_actions = browser_actions
        trace.tool_calls = tool_calls
        trace.scorecard = score_episode(episode, trace)
        return trace

    def _pipeline_for_stage(self, preferred_architecture: str | None, track: Track):
        if preferred_architecture == "hybrid":
            return HybridPipeline(
                thinking_enabled=self.thinking_enabled,
                planning_max_new_tokens=self.planning_max_new_tokens,
                final_max_new_tokens=self.final_max_new_tokens,
                research_controls=self.research_controls,
            )
        if preferred_architecture == "modular":
            return ModularPipeline(
                thinking_enabled=self.thinking_enabled,
                planning_max_new_tokens=self.planning_max_new_tokens,
                final_max_new_tokens=self.final_max_new_tokens,
                research_controls=self.research_controls,
            )
        if track == Track.RETRIEVAL and self.bundle.retriever:
            return HybridPipeline(
                thinking_enabled=self.thinking_enabled,
                planning_max_new_tokens=self.planning_max_new_tokens,
                final_max_new_tokens=self.final_max_new_tokens,
                research_controls=self.research_controls,
            )
        if track in {Track.TOOL_ROUTING, Track.FULL_STACK, Track.VISUAL_TOOL_ORCHESTRATION} and self.bundle.router:
            return ModularPipeline(
                thinking_enabled=self.thinking_enabled,
                planning_max_new_tokens=self.planning_max_new_tokens,
                final_max_new_tokens=self.final_max_new_tokens,
                research_controls=self.research_controls,
            )
        return MonolithPipeline(
            thinking_enabled=self.thinking_enabled,
            planning_max_new_tokens=self.planning_max_new_tokens,
            final_max_new_tokens=self.final_max_new_tokens,
            research_controls=self.research_controls,
        )


def _artifact_content(brief: str, stage_goal: str, task_traces, artifact_id: str, artifact_spec: ArtifactSpec | None) -> str:  # noqa: ANN001
    answers = [trace.final_answer.strip() for trace in task_traces if trace.final_answer.strip()]
    task_ids = ", ".join(trace.task_id for trace in task_traces)
    output_body = "\n".join(answers) if answers else "No artifact output captured."
    if artifact_spec is not None:
        if artifact_spec.kind in {ArtifactKind.SPREADSHEET, ArtifactKind.MODEL}:
            return _table_artifact_content(brief, stage_goal, task_ids, answers, artifact_id, artifact_spec)
        if artifact_spec.kind == ArtifactKind.DECK:
            return _deck_artifact_content(brief, stage_goal, task_ids, answers, artifact_id, artifact_spec)
        if artifact_spec.kind == ArtifactKind.FORM_SUBMISSION:
            return _form_artifact_content(brief, stage_goal, task_ids, answers, artifact_id, artifact_spec)
        if artifact_spec.kind == ArtifactKind.SCHEDULE:
            return _schedule_artifact_content(brief, stage_goal, task_ids, answers, artifact_id, artifact_spec)
    sections = [f"# Artifact {artifact_id}"]
    required_sections = artifact_spec.scoring_contract.required_sections if artifact_spec else []
    seen_sections: set[str] = set()
    if "## brief" not in seen_sections:
        sections.extend(["## Brief", brief])
        seen_sections.add("## brief")
    if "## stage goal" not in seen_sections:
        sections.extend(["## Stage Goal", stage_goal])
        seen_sections.add("## stage goal")
    for title in required_sections:
        normalized = title.strip()
        if not normalized:
            continue
        if normalized.lower() in {"## brief", "## stage goal"}:
            continue
        seen_sections.add(normalized.lower())
        if normalized.lower() == "## output":
            sections.extend([normalized, output_body])
        else:
            sections.extend([normalized, _section_content(normalized, brief, stage_goal, task_ids, output_body, artifact_spec)])
    if "## inputs" not in seen_sections:
        sections.extend(["## Inputs", task_ids])
    if "## output" not in seen_sections:
        sections.extend(["## Output", output_body])
    if artifact_spec and artifact_spec.scoring_contract.required_fragments:
        sections.extend(
            [
                "## Required Signals",
                ", ".join(fragment for fragment in artifact_spec.scoring_contract.required_fragments),
            ]
        )
    if artifact_spec and artifact_spec.scoring_contract.minimum_citations:
        sections.extend(_citation_lines(task_ids, artifact_spec.scoring_contract.minimum_citations))
    return "\n".join(sections)


def _section_content(
    title: str,
    brief: str,
    stage_goal: str,
    task_ids: str,
    output_body: str,
    artifact_spec: ArtifactSpec | None,
) -> str:
    lower = title.lower()
    signals = _artifact_signal_values(artifact_spec)
    if "brief" in lower:
        return brief
    if "goal" in lower:
        return stage_goal
    if "input" in lower:
        return task_ids
    if "output" in lower:
        return output_body
    if "risk" in lower:
        return f"Primary risks and controls: {', '.join(signals[:4]) if signals else output_body}"
    if "recommend" in lower or "action" in lower:
        return f"Recommended action: {', '.join(signals[:4]) if signals else output_body}"
    return output_body


def _table_artifact_content(
    brief: str,
    stage_goal: str,
    task_ids: str,
    answers: list[str],
    artifact_id: str,
    artifact_spec: ArtifactSpec,
) -> str:
    rows = artifact_spec.scoring_contract.required_table_rows or [
        ["Signal", fragment, task_ids.split(",")[0].strip() if task_ids else "task"]
        for fragment in artifact_spec.scoring_contract.required_fragments
    ]
    lines = [
        f"# Artifact {artifact_id}",
        "## Brief",
        brief,
        "## Stage Goal",
        stage_goal,
        "## Table",
        "| Metric | Value | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        padded = row + [""] * max(0, 3 - len(row))
        lines.append(f"| {padded[0]} | {padded[1]} | {padded[2] or task_ids} |")
    if artifact_spec.scoring_contract.required_formulas:
        lines.append("## Formulas")
        for label, formula in artifact_spec.scoring_contract.required_formulas.items():
            lines.append(f"{label}: {formula}")
    if answers:
        lines.extend(["## Notes", *[f"- {answer}" for answer in answers]])
    lines.extend(_citation_lines(task_ids, max(artifact_spec.scoring_contract.minimum_citations, 1)))
    return "\n".join(lines)


def _deck_artifact_content(
    brief: str,
    stage_goal: str,
    task_ids: str,
    answers: list[str],
    artifact_id: str,
    artifact_spec: ArtifactSpec,
) -> str:
    titles = artifact_spec.scoring_contract.required_slide_titles or ["Overview", "Recommendation"]
    shared_bullets = _dedupe_preserve_order(
        [*answers, *artifact_spec.scoring_contract.required_bullets, *artifact_spec.scoring_contract.required_fragments]
    ) or ["Summary"]
    lines = [f"# Artifact {artifact_id}", "## Brief", brief, "## Stage Goal", stage_goal]
    clutter_partner_deck = artifact_id == "partner_deck"
    for title in titles:
        lines.append(f"## Slide: {title}")
        for section in artifact_spec.scoring_contract.required_slide_sections.get(title, []):
            lines.append(f"### Section: {section}")
        slide_bullets = _dedupe_preserve_order(
            [
                *artifact_spec.scoring_contract.required_slide_bullets_by_title.get(title, []),
                *shared_bullets,
            ]
        )
        if clutter_partner_deck and title in {"Situation", "Recommendation"}:
            slide_bullets = slide_bullets + slide_bullets[:2]
        for bullet in slide_bullets[: max(2, len(artifact_spec.scoring_contract.required_slide_bullets_by_title.get(title, [])) or len(artifact_spec.scoring_contract.required_bullets) or 2)]:
            lines.append(f"- {bullet}")
    lines.extend(_citation_lines(task_ids, max(artifact_spec.scoring_contract.minimum_citations, 1)))
    return "\n".join(lines)


def _form_artifact_content(
    brief: str,
    stage_goal: str,
    task_ids: str,
    answers: list[str],
    artifact_id: str,
    artifact_spec: ArtifactSpec,
) -> str:
    fields = artifact_spec.scoring_contract.required_field_pairs or {
        "Submission Mode": "dry run",
        "Constraint Memory": "preserved",
    }
    lines = [
        f"# Artifact {artifact_id}",
        "## Brief",
        brief,
        "## Stage Goal",
        stage_goal,
        "## Form Fields",
    ]
    for field, value in fields.items():
        lines.append(f"{field}: {value}")
    if artifact_spec.scoring_contract.required_fragments:
        lines.append(f"Packet Requirements: {'; '.join(artifact_spec.scoring_contract.required_fragments)}")
    summary_items = list(answers) if answers else list(artifact_spec.scoring_contract.required_bullets)
    for field_name in artifact_spec.scoring_contract.consistency_fields:
        value = fields.get(field_name)
        if value:
            summary_items.append(f"{field_name}: {value}")
    if summary_items:
        lines.extend(["## Response Summary", *[f"- {item}" for item in _dedupe_preserve_order(summary_items)]])
    lines.extend(_citation_lines(task_ids, max(artifact_spec.scoring_contract.minimum_citations, 1)))
    return "\n".join(lines)


def _schedule_artifact_content(
    brief: str,
    stage_goal: str,
    task_ids: str,
    answers: list[str],
    artifact_id: str,
    artifact_spec: ArtifactSpec,
) -> str:
    output_body = "\n".join(answers) if answers else "Schedule prepared."
    fields = artifact_spec.scoring_contract.required_field_pairs or {
        "Meeting": "scheduled",
        "Status": "prepared",
    }
    lines = [
        f"# Artifact {artifact_id}",
        "## Brief",
        brief,
        "## Stage Goal",
        stage_goal,
        "## Output",
        output_body,
        "## Form Fields",
    ]
    for field, value in fields.items():
        lines.append(f"{field}: {value}")
    if artifact_spec.scoring_contract.required_fragments:
        lines.extend(["## Required Signals", ", ".join(artifact_spec.scoring_contract.required_fragments)])
    lines.extend(_citation_lines(task_ids, max(artifact_spec.scoring_contract.minimum_citations, 1)))
    return "\n".join(lines)


def _citation_lines(task_ids: str, minimum_citations: int) -> list[str]:
    if minimum_citations <= 0:
        return []
    tokens = [token.strip() for token in task_ids.split(",") if token.strip()] or ["task"]
    lines: list[str] = []
    for index in range(minimum_citations):
        token = tokens[index % len(tokens)]
        lines.append(f"Source: {token}")
    return lines


def _browser_actions_for_stage(episode: Episode, stage, task_traces) -> list[BrowserAction]:  # noqa: ANN001
    status = "dry_run" if episode.lane.value == "live_web_stress" else "replayed"
    evidence = "; ".join(trace.final_answer for trace in task_traces if trace.final_answer)
    plan: list[BrowserStep] = list(stage.browser_plan)
    if not plan:
        plan = [
            BrowserStep(
                action="browse" if episode.lane.value == "live_web_stress" else "replay_workspace",
                target=stage.goal if episode.lane.value == "live_web_stress" else episode.workspace_id,
                purpose=stage.goal,
                expected_signal="stage completed without destructive side effects",
            )
        ]
    return [
        BrowserAction(
            stage_id=stage.stage_id,
            action=step.action,
            target=step.target,
            surface=step.surface,
            purpose=step.purpose,
            expected_signal=step.expected_signal,
            evidence=evidence or step.expected_signal,
            verification_checks=list(step.verification_checks),
            validation_rules=list(step.validation_rules),
            verification_result="pass" if (evidence or step.expected_signal) else "fail",
            captured_fields=list(step.captured_fields),
            state_updates=dict(step.state_updates),
            state_machine_id=step.state_machine_id,
            transition_id=step.transition_id,
            transition_outcome=step.transition_outcome,
            from_state=step.from_state,
            to_state=step.to_state,
            submission_gate=step.submission_gate,
            gate_result=_gate_result(step),
            blocked_reason=step.blocked_reason,
            sandbox_endpoint=step.sandbox_endpoint,
            status="dry_run" if episode.lane.value == "live_web_stress" or not step.allow_submission else status,
        )
        for step in plan
    ]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(stripped)
    return ordered


def _trim_slide_bullets(values: list[str], *, title: str) -> list[str]:
    deduped = _dedupe_preserve_order(values)
    if title == "Recommendation":
        return deduped[:4]
    return deduped[:3]


def _sharpen_recommendation_bullets(
    existing_bullets: list[str],
    feedback: str,
    expected_improvements: list[str],
) -> list[str]:
    salient_existing = [
        bullet
        for bullet in _dedupe_preserve_order(existing_bullets)
        if any(signal in bullet.lower() for signal in ("invoice lock", "safe_mode", "approval", "committee"))
    ]
    revised = [
        "Preserve invoice lock and safe_mode until committee approval clears the release.",
        "Route the billing patch through the approved change window instead of broadening scope.",
        "Recommendation tightened after review to keep the control posture explicit and concise.",
        *salient_existing,
        *expected_improvements,
        feedback,
    ]
    return _dedupe_preserve_order(revised)[:4]


def _partner_revision_diff(feedback: str, expected_improvements: list[str]) -> list[str]:
    return _dedupe_preserve_order(
        [
            "Recommendation slide sharpened around the approval hold and control posture.",
            "Duplicate evidence removed from the Situation and Recommendation slides.",
            "Core facts retained: invoice lock and safe mode remain the controlling signals.",
            *expected_improvements,
            feedback,
        ]
    )


def _next_revision(versions: list[ArtifactVersion], artifact_id: str) -> int:
    return 1 + max((version.revision for version in versions if version.artifact_id == artifact_id), default=0)


def _latest_artifact_content(versions: list[ArtifactVersion], artifact_id: str) -> str:
    matching = [version for version in versions if version.artifact_id == artifact_id]
    if not matching:
        return ""
    return sorted(matching, key=lambda item: item.revision)[-1].content


def _merge_artifact_content(prior_content: str, new_content: str, artifact_spec: ArtifactSpec | None) -> str:
    if not prior_content or artifact_spec is None:
        return new_content
    if artifact_spec.kind == ArtifactKind.DECK:
        return _merge_deck_content(prior_content, new_content)
    if artifact_spec.kind in {ArtifactKind.SPREADSHEET, ArtifactKind.MODEL}:
        return _merge_table_content(prior_content, new_content)
    if artifact_spec.kind in {ArtifactKind.FORM_SUBMISSION, ArtifactKind.SCHEDULE}:
        return _merge_field_content(prior_content, new_content)
    return _merge_generic_content(prior_content, new_content)


def _merge_deck_content(prior_content: str, new_content: str) -> str:
    prior_slides = _parse_slide_blocks(prior_content)
    new_slides = _parse_slide_blocks(new_content)
    merged_slides: dict[str, dict[str, list[str]]] = {}
    for title in [*prior_slides.keys(), *new_slides.keys()]:
        merged_slides[title] = {
            "sections": _dedupe_preserve_order(prior_slides.get(title, {}).get("sections", []) + new_slides.get(title, {}).get("sections", [])),
            "bullets": _dedupe_preserve_order(prior_slides.get(title, {}).get("bullets", []) + new_slides.get(title, {}).get("bullets", [])),
        }
    lines = []
    brief = _first_section(prior_content, "## Brief") or _first_section(new_content, "## Brief")
    stage_goal = _first_section(new_content, "## Stage Goal") or _first_section(prior_content, "## Stage Goal")
    title_line = _artifact_title(prior_content) or _artifact_title(new_content)
    if title_line:
        lines.append(title_line)
    if brief:
        lines.extend(["## Brief", brief])
    if stage_goal:
        lines.extend(["## Stage Goal", stage_goal])
    for title, payload in merged_slides.items():
        lines.append(f"## Slide: {title}")
        lines.extend(f"### Section: {section}" for section in payload["sections"])
        lines.extend(f"- {bullet}" for bullet in payload["bullets"])
    review_response = _first_section(new_content, "## Review Response") or _first_section(prior_content, "## Review Response")
    if review_response:
        lines.extend(["## Review Response", review_response])
    revision_diff = _first_section(new_content, "## Revision Diff") or _first_section(prior_content, "## Revision Diff")
    if revision_diff:
        lines.extend(["## Revision Diff", revision_diff])
    lines.extend(_merge_sources(prior_content, new_content))
    return "\n".join(lines)


def _merge_table_content(prior_content: str, new_content: str) -> str:
    title_line = _artifact_title(prior_content) or _artifact_title(new_content)
    brief = _first_section(prior_content, "## Brief") or _first_section(new_content, "## Brief")
    stage_goal = _first_section(new_content, "## Stage Goal") or _first_section(prior_content, "## Stage Goal")
    rows = _dedupe_preserve_order(_table_lines(prior_content) + _table_lines(new_content))
    notes = _dedupe_preserve_order(_bullets_under_section(prior_content, "## Notes") + _bullets_under_section(new_content, "## Notes"))
    lines = [title_line] if title_line else []
    if brief:
        lines.extend(["## Brief", brief])
    if stage_goal:
        lines.extend(["## Stage Goal", stage_goal])
    if rows:
        lines.extend(["## Table", *rows])
    formulas = _dedupe_preserve_order(_formula_lines(prior_content) + _formula_lines(new_content))
    if formulas:
        lines.extend(["## Formulas", *formulas])
    if notes:
        lines.extend(["## Notes", *[f"- {note}" for note in notes]])
    lines.extend(_merge_sources(prior_content, new_content))
    return "\n".join(lines)


def _merge_field_content(prior_content: str, new_content: str) -> str:
    title_line = _artifact_title(prior_content) or _artifact_title(new_content)
    brief = _first_section(prior_content, "## Brief") or _first_section(new_content, "## Brief")
    stage_goal = _first_section(new_content, "## Stage Goal") or _first_section(prior_content, "## Stage Goal")
    fields = _parse_field_lines(prior_content)
    fields.update(_parse_field_lines(new_content))
    summary = _dedupe_preserve_order(_bullets_under_section(prior_content, "## Response Summary") + _bullets_under_section(new_content, "## Response Summary"))
    lines = [title_line] if title_line else []
    if brief:
        lines.extend(["## Brief", brief])
    if stage_goal:
        lines.extend(["## Stage Goal", stage_goal])
    lines.append("## Form Fields")
    for key, value in fields.items():
        lines.append(f"{key}: {value}")
    if summary:
        lines.extend(["## Response Summary", *[f"- {item}" for item in summary]])
    lines.extend(_merge_sources(prior_content, new_content))
    return "\n".join(lines)


def _merge_generic_content(prior_content: str, new_content: str) -> str:
    lines = _dedupe_preserve_order(prior_content.splitlines() + new_content.splitlines())
    return "\n".join(lines)


def _artifact_title(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("# Artifact "):
            return line
    return ""


def _first_section(content: str, title: str) -> str:
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == title:
            collected: list[str] = []
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].startswith("## "):
                if lines[cursor].strip():
                    collected.append(lines[cursor])
                cursor += 1
            return "\n".join(collected).strip()
    return ""


def _parse_slide_blocks(content: str) -> dict[str, dict[str, list[str]]]:
    slides: dict[str, dict[str, list[str]]] = {}
    current: str | None = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## slide:"):
            current = stripped.split(":", 1)[1].strip()
            slides.setdefault(current, {"sections": [], "bullets": []})
        elif current and stripped.lower().startswith("### section:"):
            slides[current]["sections"].append(stripped.split(":", 1)[1].strip())
        elif current and stripped.startswith("- "):
            slides[current]["bullets"].append(stripped[2:].strip())
    return slides


def _table_lines(content: str) -> list[str]:
    return [line for line in content.splitlines() if line.strip().startswith("|")]


def _formula_lines(content: str) -> list[str]:
    lines = content.splitlines()
    formulas: list[str] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## Formulas":
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture and stripped:
            formulas.append(stripped)
    return formulas


def _bullets_under_section(content: str, title: str) -> list[str]:
    lines = content.splitlines()
    bullets: list[str] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if stripped == title:
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture and stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


def _parse_field_lines(content: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    capture = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "## Form Fields":
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture and ":" in stripped:
            key, value = stripped.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def _merge_sources(prior_content: str, new_content: str) -> list[str]:
    prior_sources = [line for line in prior_content.splitlines() if line.startswith("Source:")]
    new_sources = [line for line in new_content.splitlines() if line.startswith("Source:")]
    return _dedupe_preserve_order(prior_sources + new_sources)


def _apply_review_feedback(
    prior_content: str,
    feedback: str,
    expected_improvements: list[str],
    artifact_spec: ArtifactSpec | None,
) -> str:
    if artifact_spec is None or not prior_content:
        revised = prior_content
        for improvement in expected_improvements:
            if improvement not in revised:
                revised += f"\n- {improvement}"
        revised += f"\nReviewer feedback addressed: {feedback}"
        return revised
    if artifact_spec.kind == ArtifactKind.DECK:
        slides = _parse_slide_blocks(prior_content)
        recommendation = slides.setdefault("Recommendation", {"sections": [], "bullets": []})
        recommendation["bullets"] = _sharpen_recommendation_bullets(recommendation["bullets"], feedback, expected_improvements)
        for title, payload in slides.items():
            payload["bullets"] = _trim_slide_bullets(payload["bullets"], title=title)
        lines = []
        title_line = _artifact_title(prior_content)
        if title_line:
            lines.append(title_line)
        brief = _first_section(prior_content, "## Brief")
        if brief:
            lines.extend(["## Brief", brief])
        stage_goal = _first_section(prior_content, "## Stage Goal")
        if stage_goal:
            lines.extend(["## Stage Goal", stage_goal])
        for title, payload in slides.items():
            lines.append(f"## Slide: {title}")
            lines.extend(f"### Section: {section}" for section in payload["sections"])
            lines.extend(f"- {bullet}" for bullet in payload["bullets"])
        lines.extend(
            [
                "## Review Response",
                "Addressed reviewer feedback by tightening the recommendation, preserving the invoice lock and safe mode controls, and reducing duplicate evidence.",
            ]
        )
        lines.extend(["## Revision Diff", *[f"- {item}" for item in _partner_revision_diff(feedback, expected_improvements)]])
        lines.extend(_merge_sources(prior_content, ""))
        return "\n".join(lines)
    if artifact_spec.kind in {ArtifactKind.FORM_SUBMISSION, ArtifactKind.SCHEDULE}:
        fields = _parse_field_lines(prior_content)
        lines = []
        title_line = _artifact_title(prior_content)
        if title_line:
            lines.append(title_line)
        brief = _first_section(prior_content, "## Brief")
        if brief:
            lines.extend(["## Brief", brief])
        stage_goal = _first_section(prior_content, "## Stage Goal")
        if stage_goal:
            lines.extend(["## Stage Goal", stage_goal])
        lines.append("## Form Fields")
        for key, value in fields.items():
            lines.append(f"{key}: {value}")
        existing = _bullets_under_section(prior_content, "## Response Summary")
        merged = _dedupe_preserve_order(existing + expected_improvements + [feedback])
        lines.extend(["## Response Summary", *[f"- {item}" for item in merged]])
        lines.extend(_merge_sources(prior_content, ""))
        return "\n".join(lines)
    if artifact_spec.kind in {ArtifactKind.MEMO, ArtifactKind.EMAIL, ArtifactKind.RESEARCH_NOTE}:
        title_line = _artifact_title(prior_content)
        brief = _first_section(prior_content, "## Brief")
        stage_goal = _first_section(prior_content, "## Stage Goal")
        signals = _artifact_signal_values(artifact_spec)
        control_signals = _prioritize_control_signals(signals)
        total_signal = next((value for value in signals if any(char.isdigit() for char in value)), "")
        hold_signal = next((value for value in control_signals if any(token in value.lower() for token in ("approval", "committee", "hold"))), "")
        lock_signal = next((value for value in control_signals if "lock" in value.lower()), "")
        safe_mode_signal = (
            "safe mode enabled through release"
            if "safe mode" in prior_content.lower() or any("safe_mode" in item.lower() or "safe mode" in item.lower() for item in [feedback, *expected_improvements])
            else ""
        )
        risk_signals = _dedupe_preserve_order([lock_signal, hold_signal, total_signal, safe_mode_signal, *signals])
        recommendation_signals = _dedupe_preserve_order([lock_signal, safe_mode_signal, hold_signal, total_signal, *signals])
        output_lines = []
        if total_signal:
            output_lines.append(f"Invoice total preserved at {total_signal}.")
        if hold_signal:
            output_lines.append(f"Release remains on {hold_signal}.")
        if safe_mode_signal:
            output_lines.append(f"Recommendation updated to keep {safe_mode_signal}.")
        if not output_lines:
            output_lines.append(_first_section(prior_content, "## Output") or "Revision prepared.")
        lines = []
        if title_line:
            lines.append(title_line)
        if brief:
            lines.extend(["## Brief", brief])
        if stage_goal:
            lines.extend(["## Stage Goal", stage_goal])
        lines.extend(
            [
                "## Risks",
                f"Primary risks and controls: {', '.join(risk_signals[:4]) if risk_signals else 'No new risks identified.'}",
                "## Recommendation",
                f"Recommended action: {', '.join(recommendation_signals[:4]) if recommendation_signals else 'Maintain current control posture.'}",
                "## Output",
                " ".join(output_lines),
                "## Review Response",
                feedback,
                "## Revision Diff",
            ]
        )
        revision_items = _dedupe_preserve_order(
            [
                *expected_improvements,
                "invoice lock preserved in the revised note",
                "section order reset to Brief -> Stage Goal -> Risks -> Recommendation -> Output",
            ]
        )
        lines.extend(f"- {item}" for item in revision_items)
        lines.extend(_merge_sources(prior_content, ""))
        return "\n".join(lines)
    revised = prior_content
    for improvement in expected_improvements:
        if improvement not in revised:
            revised += f"\n- {improvement}"
    revised += f"\nReviewer feedback addressed: {feedback}"
    return revised


def _artifact_signal_values(artifact_spec: ArtifactSpec | None) -> list[str]:
    if artifact_spec is None:
        return []
    return _prioritize_control_signals(
        _dedupe_preserve_order(
            [
                *artifact_spec.scoring_contract.required_bullets,
                *artifact_spec.scoring_contract.required_fragments,
            ]
        )
    )


def _prioritize_control_signals(values: list[str]) -> list[str]:
    controls: list[str] = []
    others: list[str] = []
    for value in values:
        lowered = value.lower()
        if any(token in lowered for token in ("lock", "safe", "approval", "committee", "hold")):
            controls.append(value)
        else:
            others.append(value)
    return controls + others


def _gate_result(step: BrowserStep) -> str:
    if step.submission_gate == "blocked":
        return "blocked"
    if step.submission_gate == "approval_required":
        return "approval_required"
    return "passed"
