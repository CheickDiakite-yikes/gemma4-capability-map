from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.knowledge_work.exporters import export_episode_leaderboard_csv
from gemma4_capability_map.knowledge_work.loader import load_episodes
from gemma4_capability_map.knowledge_work.native_artifacts import inspect_artifact
from gemma4_capability_map.knowledge_work.replay import load_episode_traces, summarize_episode_traces
from gemma4_capability_map.knowledge_work.runner import EpisodeRunner
from gemma4_capability_map.knowledge_work.scoring import score_episode
from gemma4_capability_map.knowledge_work.schemas import (
    ArtifactKind,
    ArtifactScoringContract,
    ArtifactSpec,
    ArtifactVersion,
    BenchmarkLane,
    Episode,
    EpisodeTrace,
    MemoryUpdate,
    RoleFamily,
)
from gemma4_capability_map.models.embeddinggemma_runner import EmbeddingGemmaRetriever
from gemma4_capability_map.models.functiongemma_runner import FunctionGemmaRunner
from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.pipelines.base import RuntimeBundle
from gemma4_capability_map.schemas import Task
from gemma4_capability_map.tools.executor import DeterministicExecutor
from gemma4_capability_map.tools.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[1]
MAKE_KWA_PATH = ROOT / "scripts" / "make_knowledge_work_arena.py"
SPEC = importlib.util.spec_from_file_location("make_knowledge_work_arena", MAKE_KWA_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
build_replayable_episodes = MODULE.build_replayable_episodes
build_live_web_episodes = MODULE.build_live_web_episodes

MAKE_GOLD_PATH = ROOT / "scripts" / "make_gold.py"
GOLD_SPEC = importlib.util.spec_from_file_location("make_gold_for_kwa_tests", MAKE_GOLD_PATH)
GOLD_MODULE = importlib.util.module_from_spec(GOLD_SPEC)
assert GOLD_SPEC and GOLD_SPEC.loader
GOLD_SPEC.loader.exec_module(GOLD_MODULE)
build_visual_tool_tasks = GOLD_MODULE.build_visual_tool_tasks

BUILD_HISTORY_PATH = ROOT / "scripts" / "build_knowledge_work_history.py"
HISTORY_SPEC = importlib.util.spec_from_file_location("build_knowledge_work_history", BUILD_HISTORY_PATH)
HISTORY_MODULE = importlib.util.module_from_spec(HISTORY_SPEC)
assert HISTORY_SPEC and HISTORY_SPEC.loader
HISTORY_SPEC.loader.exec_module(HISTORY_MODULE)

RUN_KWA_PATH = ROOT / "scripts" / "run_knowledge_work_arena.py"
RUN_KWA_SPEC = importlib.util.spec_from_file_location("run_knowledge_work_arena", RUN_KWA_PATH)
RUN_KWA_MODULE = importlib.util.module_from_spec(RUN_KWA_SPEC)
assert RUN_KWA_SPEC and RUN_KWA_SPEC.loader
RUN_KWA_SPEC.loader.exec_module(RUN_KWA_MODULE)


def _load_tasks() -> list[Task]:
    from gemma4_capability_map.io import load_jsonl

    tasks_by_id: dict[str, Task] = {}
    for path in sorted((ROOT / "data" / "gold").glob("*.jsonl")):
        for task in load_jsonl(path, Task):
            tasks_by_id[task.task_id] = task
    for task in build_visual_tool_tasks():
        tasks_by_id[task.task_id] = task
    return list(tasks_by_id.values())


def test_episode_specs_validate_and_cover_both_lanes() -> None:
    replayable = build_replayable_episodes()
    live = build_live_web_episodes()
    assert len(replayable) == 29
    assert len(live) == 23
    assert {episode.role_family.value for episode in replayable} == {
        "executive_assistant",
        "job_application_ops",
        "finance",
    }
    assert any(artifact.scoring_contract.required_table_rows for episode in replayable for artifact in episode.artifacts)
    assert any(artifact.scoring_contract.required_field_pairs for episode in replayable for artifact in episode.artifacts)
    assert any(artifact.scoring_contract.required_slide_titles for episode in replayable for artifact in episode.artifacts)
    assert any(artifact.scoring_contract.required_formula_cells for episode in replayable + live for artifact in episode.artifacts)
    assert any(artifact.scoring_contract.required_heading_order for episode in replayable + live for artifact in episode.artifacts)
    assert any(artifact.scoring_contract.required_slide_bullets_by_title for episode in replayable + live for artifact in episode.artifacts)
    assert any("visual_kwa" in episode.benchmark_tags for episode in replayable + live)
    assert all(stage.browser_plan for episode in replayable + live for stage in episode.stages)


def test_episode_runner_produces_scorecard_and_revision_history() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    trace = runner.run(build_replayable_episodes()[0])

    assert trace.scorecard.role_readiness_score > 0.0
    assert trace.stage_traces
    assert trace.artifact_versions
    assert trace.review_history
    assert trace.memory_updates
    assert all(action.purpose for action in trace.browser_actions)
    assert all(action.expected_signal for action in trace.browser_actions)
    assert trace.scorecard.browser_workflow_score > 0.0
    assert 0.0 <= trace.scorecard.role_readiness_score <= 1.0


def test_episode_runner_memory_is_isolated_between_runs() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    first = runner.run(build_replayable_episodes()[0])
    second = runner.run(build_replayable_episodes()[1])

    assert first.episode_id != second.episode_id
    assert all(update.stage_id.startswith(first.episode_id) for update in first.memory_updates)
    assert all(update.stage_id.startswith(second.episode_id) for update in second.memory_updates)


def test_memory_retention_scores_preserved_patch_facts_without_verbatim_reason_block() -> None:
    episode = Episode(
        episode_id="kwa_memory_retention_semantic",
        role_family=RoleFamily.FINANCE,
        lane=BenchmarkLane.REPLAYABLE_CORE,
        workspace_id="finance-memory",
        brief="Retain the approved billing patch facts across revisions.",
        artifacts=[
            ArtifactSpec(
                artifact_id="hold_note",
                kind=ArtifactKind.MEMO,
                path_or_target="workspace://finance-memory/hold_note.docx",
                scoring_contract=ArtifactScoringContract(required_fragments=["invoice_lock: true"]),
            )
        ],
    )
    trace = EpisodeTrace(
        run_id="memory-retention-semantic",
        episode_id=episode.episode_id,
        role_family=episode.role_family,
        lane=episode.lane,
        workspace_id=episode.workspace_id,
        artifact_versions=[
            ArtifactVersion(
                artifact_id="hold_note",
                revision=1,
                content="Invoice Lock patch: `invoice_lock: true` in `config/billing.yaml`.",
                source_stage="stage_1",
            )
        ],
        memory_updates=[
            MemoryUpdate(
                stage_id="stage_1",
                key="Finance hold stage",
                value=(
                    "Invoice Lock patch: `invoice_lock: true` in `config/billing.yaml`.\n\n"
                    'Reason: The billing runbook states that "Invoice Lock" must remain enabled to prevent invoice edits.'
                ),
            )
        ],
    )

    scorecard = score_episode(episode, trace)

    assert scorecard.memory_retention_score == 1.0


def test_memory_retention_still_fails_when_salient_fact_is_missing() -> None:
    episode = Episode(
        episode_id="kwa_memory_retention_missing",
        role_family=RoleFamily.FINANCE,
        lane=BenchmarkLane.REPLAYABLE_CORE,
        workspace_id="finance-memory-missing",
        brief="Do not over-credit unrelated retained content.",
        artifacts=[
            ArtifactSpec(
                artifact_id="hold_note",
                kind=ArtifactKind.MEMO,
                path_or_target="workspace://finance-memory-missing/hold_note.docx",
                scoring_contract=ArtifactScoringContract(required_fragments=["summary"]),
            )
        ],
    )
    trace = EpisodeTrace(
        run_id="memory-retention-missing",
        episode_id=episode.episode_id,
        role_family=episode.role_family,
        lane=episode.lane,
        workspace_id=episode.workspace_id,
        artifact_versions=[
            ArtifactVersion(
                artifact_id="hold_note",
                revision=1,
                content="General hold summary with no billing control details.",
                source_stage="stage_1",
            )
        ],
        memory_updates=[
            MemoryUpdate(
                stage_id="stage_1",
                key="Finance hold stage",
                value="Invoice Lock patch: `invoice_lock: true` in `config/billing.yaml`.",
            )
        ],
    )

    scorecard = score_episode(episode, trace)

    assert scorecard.memory_retention_score == 0.0


def test_episode_export_and_summary_round_trip(tmp_path: Path) -> None:
    episodes_path = tmp_path / "episodes.jsonl"
    traces_path = tmp_path / "episode_traces.jsonl"
    leaderboard_path = tmp_path / "episode_leaderboard.csv"
    replayable = build_replayable_episodes()[:2]
    dump_jsonl(episodes_path, replayable)
    loaded = load_episodes(episodes_path)
    assert len(loaded) == 2

    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    traces = [runner.run(episode) for episode in loaded]
    dump_jsonl(traces_path, traces)
    export_episode_leaderboard_csv(traces, leaderboard_path)
    reloaded = load_episode_traces(traces_path)
    summary = summarize_episode_traces(reloaded)

    assert summary["runs"] == 2.0
    assert summary["strict_interface_avg"] >= 0.0
    assert leaderboard_path.exists()


def test_episode_runner_emits_structured_model_and_deck_artifacts() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    replayable = build_replayable_episodes()

    model_trace = runner.run(next(episode for episode in replayable if episode.episode_id == "kwa_finance_three_statement_model"))
    model_artifact = next(version for version in model_trace.artifact_versions if version.artifact_id == "financial_model" and version.revision == 1)
    assert "| Metric | Value | Evidence |" in model_artifact.content
    assert "## Formulas" in model_artifact.content
    assert "Revenue Forecast: =BASE_REVENUE+DELTA" in model_artifact.content
    assert "Source:" in model_artifact.content
    assert model_artifact.file_format == "xlsx"
    assert model_artifact.file_path and model_artifact.file_path.endswith(".xlsx")

    deck_trace = runner.run(next(episode for episode in replayable if episode.episode_id == "kwa_finance_partner_deck_revision"))
    deck_artifact = [version for version in deck_trace.artifact_versions if version.artifact_id == "partner_deck"][-1]
    assert "## Slide: Situation" in deck_artifact.content
    assert "## Slide: Recommendation" in deck_artifact.content
    assert "### Section: Context" in deck_artifact.content
    assert "## Revision Diff" in deck_artifact.content
    assert deck_artifact.file_format == "pptx"
    assert deck_artifact.file_path and deck_artifact.file_path.endswith(".pptx")


def test_visual_kwa_episodes_exist_for_each_role_and_lane() -> None:
    replayable = build_replayable_episodes()
    live = build_live_web_episodes()
    replayable_visual = [episode for episode in replayable if "visual_kwa" in episode.benchmark_tags]
    live_visual = [episode for episode in live if "visual_kwa" in episode.benchmark_tags]

    assert {episode.role_family.value for episode in replayable_visual} == {
        "executive_assistant",
        "job_application_ops",
        "finance",
    }
    assert {episode.role_family.value for episode in live_visual} == {
        "executive_assistant",
        "job_application_ops",
        "finance",
    }
    assert all(any(task_ref.startswith("visual_") for stage in episode.stages for task_ref in stage.task_refs) for episode in replayable_visual + live_visual)
    assert all(episode.review_rounds for episode in replayable_visual + live_visual)
    assert any("visual_013_dashboard_stale_selection_recovery" in task_ref for episode in replayable_visual for stage in episode.stages for task_ref in stage.task_refs)
    assert any("visual_014_form_phone_refinement" in task_ref for episode in replayable_visual for stage in episode.stages for task_ref in stage.task_refs)
    assert any("visual_015_slide_policy_revision_pressure" in task_ref for episode in replayable_visual for stage in episode.stages for task_ref in stage.task_refs)


def test_visual_kwa_v2_realism_episodes_exist_in_both_lanes() -> None:
    replayable = {episode.episode_id: episode for episode in build_replayable_episodes()}
    live = {episode.episode_id: episode for episode in build_live_web_episodes()}

    assert "kwa_exec_visual_dashboard_revision_hold_v2" in replayable
    assert "kwa_exec_visual_dashboard_referent_hold_v3" in replayable
    assert "kwa_jobs_visual_constraint_override_hold_v2" in replayable
    assert "kwa_jobs_visual_latest_issue_hold_v3" in replayable
    assert "kwa_finance_visual_invoice_revision_hold_v2" in replayable
    assert "kwa_exec_live_visual_dashboard_revision_hold_v2" in live
    assert "kwa_exec_live_visual_dashboard_referent_hold_v3" in live
    assert "kwa_jobs_live_visual_constraint_override_hold_v2" in live
    assert "kwa_jobs_live_visual_latest_issue_hold_v3" in live
    assert "kwa_finance_live_visual_invoice_revision_hold_v2" in live


def test_harnessability_resume_episodes_exist_in_both_lanes() -> None:
    replayable = {episode.episode_id: episode for episode in build_replayable_episodes()}
    live = {episode.episode_id: episode for episode in build_live_web_episodes()}

    assert "kwa_exec_latest_action_resume_hold_v4" in replayable
    assert "kwa_jobs_phone_patch_resume_hold_v4" in replayable
    assert "kwa_finance_invoice_lock_direction_hold_v4" in replayable
    assert "kwa_exec_live_latest_action_resume_hold_v4" in live
    assert "kwa_jobs_live_phone_patch_resume_hold_v4" in live
    assert "kwa_finance_live_invoice_lock_direction_hold_v4" in live

    expected_tags = {
        "harnessability_resume",
        "harnessability_project_memory",
        "direction_following_latest_instruction",
    }
    assert expected_tags.issubset(set(replayable["kwa_exec_latest_action_resume_hold_v4"].benchmark_tags))
    assert "tool_api" in replayable["kwa_exec_latest_action_resume_hold_v4"].benchmark_tags
    assert "tool_cli" in replayable["kwa_jobs_phone_patch_resume_hold_v4"].benchmark_tags
    assert "harnessability_approval_resume" in replayable["kwa_finance_invoice_lock_direction_hold_v4"].benchmark_tags
    assert "direction_following_stale_override" in replayable["kwa_finance_invoice_lock_direction_hold_v4"].benchmark_tags

    assert replayable["kwa_exec_visual_dashboard_revision_hold_v2"].review_rounds
    assert replayable["kwa_exec_visual_dashboard_referent_hold_v3"].review_rounds
    assert replayable["kwa_jobs_visual_constraint_override_hold_v2"].review_rounds
    assert replayable["kwa_jobs_visual_latest_issue_hold_v3"].review_rounds
    assert replayable["kwa_finance_visual_invoice_revision_hold_v2"].review_rounds
    assert any(step.transition_outcome == "validation_failed" for step in replayable["kwa_exec_visual_dashboard_revision_hold_v2"].stages[0].browser_plan)
    assert any(step.transition_outcome == "validation_failed" for step in replayable["kwa_exec_visual_dashboard_referent_hold_v3"].stages[0].browser_plan)
    assert any(step.submission_gate == "approval_required" for step in replayable["kwa_jobs_visual_constraint_override_hold_v2"].stages[1].browser_plan)
    assert any(step.submission_gate == "approval_required" for step in replayable["kwa_jobs_visual_latest_issue_hold_v3"].stages[1].browser_plan)
    assert any(step.submission_gate == "approval_required" for step in live["kwa_finance_live_visual_invoice_revision_hold_v2"].stages[1].browser_plan)
    assert any(step.submission_gate == "approval_required" for step in replayable["kwa_exec_visual_dashboard_referent_hold_v3"].stages[1].browser_plan)
    assert any(step.submission_gate == "approval_required" for step in live["kwa_exec_live_visual_dashboard_referent_hold_v3"].stages[1].browser_plan)
    assert any(step.submission_gate == "approval_required" for step in live["kwa_jobs_live_visual_latest_issue_hold_v3"].stages[1].browser_plan)


def test_visual_kwa_v3_referent_hold_episodes_require_recovery_then_approval_gate() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    replayable = {episode.episode_id: episode for episode in build_replayable_episodes()}
    live = {episode.episode_id: episode for episode in build_live_web_episodes()}

    exec_trace = runner.run(replayable["kwa_exec_visual_dashboard_referent_hold_v3"])
    assert any(action.transition_outcome == "validation_failed" for action in exec_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in exec_trace.browser_actions)
    assert any(action.submission_gate == "approval_required" for action in exec_trace.browser_actions)
    assert exec_trace.scorecard.browser_workflow_score > 0.95

    dashboard_brief = next(version for version in exec_trace.artifact_versions if version.artifact_id == "dashboard_referent_hold_brief")
    assert "latest filter" in dashboard_brief.content
    assert "approval hold" in dashboard_brief.content

    live_trace = runner.run(live["kwa_exec_live_visual_dashboard_referent_hold_v3"])
    assert any(action.transition_outcome == "validation_failed" for action in live_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in live_trace.browser_actions)
    assert any(action.submission_gate == "approval_required" for action in live_trace.browser_actions)
    assert live_trace.scorecard.browser_workflow_score > 0.95

    live_brief = next(version for version in live_trace.artifact_versions if version.artifact_id == "live_dashboard_referent_hold_brief_v3")
    assert "customer ops" in live_brief.content
    assert "stale finance panel" not in live_brief.content.lower()

    jobs_trace = runner.run(replayable["kwa_jobs_visual_latest_issue_hold_v3"])
    assert any(action.transition_outcome == "validation_failed" for action in jobs_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in jobs_trace.browser_actions)
    assert any(action.submission_gate == "approval_required" for action in jobs_trace.browser_actions)
    jobs_packet = next(version for version in jobs_trace.artifact_versions if version.artifact_id == "jobs_latest_issue_packet_v3")
    assert "latest issue" in jobs_packet.content.lower()
    assert "phone" in jobs_packet.content.lower()

    live_jobs_trace = runner.run(live["kwa_jobs_live_visual_latest_issue_hold_v3"])
    assert any(action.transition_outcome == "validation_failed" for action in live_jobs_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in live_jobs_trace.browser_actions)
    assert any(action.submission_gate == "approval_required" for action in live_jobs_trace.browser_actions)
    live_jobs_packet = next(version for version in live_jobs_trace.artifact_versions if version.artifact_id == "live_jobs_latest_issue_packet_v3")
    assert "phone" in live_jobs_packet.content.lower()
    assert "work authorization first" not in live_jobs_packet.content.lower()


def test_partner_deck_revision_responsiveness_is_material() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    episode = next(episode for episode in build_replayable_episodes() if episode.episode_id == "kwa_finance_partner_deck_revision")
    trace = runner.run(episode)

    assert trace.scorecard.strict_interface_score == 1.0
    assert trace.scorecard.recovered_execution_score == 1.0
    assert trace.scorecard.memory_retention_score == 1.0
    assert trace.scorecard.revision_responsiveness >= 0.5


def test_finance_visual_invoice_note_revision_preserves_invoice_lock_and_heading_order() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    episode = next(episode for episode in build_replayable_episodes() if episode.episode_id == "kwa_finance_visual_invoice_hold")
    trace = runner.run(episode)

    note = next(version for version in trace.artifact_versions if version.artifact_id == "finance_visual_note" and version.revision == 3)
    native = inspect_artifact(Path(note.file_path))

    assert note.score == 1.0
    assert "invoice lock" in note.content.lower()
    assert "## Review Response" in note.content
    assert "## Revision Diff" in note.content
    headings = native["headings"]
    assert headings.index("Brief") < headings.index("Stage Goal") < headings.index("Risks") < headings.index("Recommendation") < headings.index("Output")


def test_revision_responsiveness_rewards_latest_constraint_improvement() -> None:
    episode = Episode(
        episode_id="kwa_revision_alignment_latest_constraint",
        role_family=RoleFamily.FINANCE,
        lane=BenchmarkLane.REPLAYABLE_CORE,
        workspace_id="finance-revision-alignment",
        brief="Revise the note to keep only the latest approval-safe action.",
        artifacts=[
            ArtifactSpec(
                artifact_id="revision_note",
                kind=ArtifactKind.MEMO,
                path_or_target="workspace://finance-revision-alignment/revision_note.docx",
                scoring_contract=ArtifactScoringContract(
                    required_fragments=["invoice lock", "committee approval"],
                    forbidden_fragments=["hold publication until numbers reconcile"],
                ),
            )
        ],
        review_rounds=[
            MODULE._review(  # type: ignore[attr-defined]
                "revision_alignment_review",
                "revision_note",
                "Remove the stale publication note and keep the approval-safe action.",
                ["stale publication note removed", "committee approval explicit"],
            )
        ],
    )
    trace = EpisodeTrace(
        run_id="revision-alignment-latest-constraint",
        episode_id=episode.episode_id,
        role_family=episode.role_family,
        lane=episode.lane,
        workspace_id=episode.workspace_id,
        artifact_versions=[
            ArtifactVersion(
                artifact_id="revision_note",
                revision=1,
                content=(
                    "## Recommendation\n"
                    "- Hold publication until numbers reconcile.\n"
                    "- Invoice total remains $51,840.\n"
                ),
                source_stage="stage_1",
            ),
            ArtifactVersion(
                artifact_id="revision_note",
                revision=2,
                content=(
                    "## Recommendation\n"
                    "- Keep invoice lock enabled.\n"
                    "- Route for committee approval.\n"
                    "## Review Response\n"
                    "- Stale publication note removed.\n"
                    "## Revision Diff\n"
                    "- Replaced the earlier publication note with the latest approval-safe action.\n"
                ),
                source_stage="stage_2",
            ),
        ],
        review_history=episode.review_rounds,
    )

    scorecard = score_episode(episode, trace)

    assert scorecard.revision_responsiveness >= 0.75


def test_memory_retention_penalizes_stale_text_leak() -> None:
    episode = Episode(
        episode_id="kwa_memory_retention_stale_leak",
        role_family=RoleFamily.JOB_APPLICATION_OPS,
        lane=BenchmarkLane.REPLAYABLE_CORE,
        workspace_id="jobs-memory-stale-leak",
        brief="Keep the latest phone-first constraint and remove stale work-authorization-first framing.",
        artifacts=[
            ArtifactSpec(
                artifact_id="constraint_note",
                kind=ArtifactKind.MEMO,
                path_or_target="workspace://jobs-memory-stale-leak/constraint_note.docx",
                scoring_contract=ArtifactScoringContract(
                    required_fragments=["phone"],
                    forbidden_fragments=["work authorization first"],
                ),
            )
        ],
    )
    trace = EpisodeTrace(
        run_id="memory-retention-stale-leak",
        episode_id=episode.episode_id,
        role_family=episode.role_family,
        lane=episode.lane,
        workspace_id=episode.workspace_id,
        artifact_versions=[
            ArtifactVersion(
                artifact_id="constraint_note",
                revision=1,
                content="Priority fix is the phone field, but the draft still says work authorization first.",
                source_stage="stage_1",
            )
        ],
        memory_updates=[
            MemoryUpdate(
                stage_id="stage_1",
                key="Latest recruiter instruction",
                value="Phone issue comes first; remove any work authorization first framing.",
            )
        ],
    )

    scorecard = score_episode(episode, trace)

    assert scorecard.memory_retention_score == 0.0


def test_live_web_episode_actions_are_dry_run_browser_steps() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    trace = runner.run(build_live_web_episodes()[0])

    assert trace.browser_actions
    assert all(action.status == "dry_run" for action in trace.browser_actions)
    assert all(action.purpose for action in trace.browser_actions)
    assert all(action.expected_signal for action in trace.browser_actions)
    assert any(action.sandbox_endpoint for action in trace.browser_actions)
    assert all(action.verification_checks for action in trace.browser_actions)


def test_live_hold_episode_records_approval_gate_without_public_submission() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    episode = next(episode for episode in build_live_web_episodes() if episode.episode_id == "kwa_jobs_live_submission_hold")
    trace = runner.run(episode)

    assert any(action.submission_gate == "approval_required" for action in trace.browser_actions)
    assert any(action.gate_result == "approval_required" for action in trace.browser_actions)
    assert any(
        action.sandbox_endpoint
        for action in trace.browser_actions
        if action.submission_gate == "approval_required"
    )

    packet = next(version for version in trace.artifact_versions if version.artifact_id == "live_validated_packet")
    assert "Send Status: blocked_pending_approval" in packet.content
    assert "Candidate Role: Research Associate" in packet.content
    assert "Validation: consistent" in packet.content


def test_replayable_schedule_artifact_includes_required_fields() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    episode = next(episode for episode in build_replayable_episodes() if episode.episode_id == "kwa_exec_board_prep_pack")
    trace = runner.run(episode)

    schedule_artifact = next(version for version in trace.artifact_versions if version.artifact_id == "board_schedule")
    assert "## Output" in schedule_artifact.content
    assert "Meeting: board" in schedule_artifact.content
    assert "Status: prepared" in schedule_artifact.content


def test_partial_progress_hold_episode_records_approval_gate() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    episode = next(episode for episode in build_replayable_episodes() if episode.episode_id == "kwa_jobs_submission_hold")
    trace = runner.run(episode)

    assert any(action.submission_gate == "approval_required" for action in trace.browser_actions)
    assert any(action.gate_result == "approval_required" for action in trace.browser_actions)
    assert any(action.blocked_reason for action in trace.browser_actions if action.submission_gate == "approval_required")

    packet = next(version for version in trace.artifact_versions if version.artifact_id == "validated_packet")
    assert "Send Status: blocked_pending_approval" in packet.content
    assert "Candidate Role: Research Associate" in packet.content
    assert "Target Company: Northwind Capital" in packet.content
    assert packet.file_format == "docx"
    assert packet.file_path and packet.file_path.endswith(".docx")
    assert any(action.transition_outcome == "validation_failed" for action in trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in trace.browser_actions)


def test_new_replayable_policy_hold_episodes_require_recovery_then_gate() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    replayable = {episode.episode_id: episode for episode in build_replayable_episodes()}

    exec_trace = runner.run(replayable["kwa_exec_vendor_access_hold"])
    assert any(action.transition_outcome == "validation_failed" for action in exec_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in exec_trace.browser_actions)
    assert any(action.submission_gate == "approval_required" for action in exec_trace.browser_actions)

    finance_trace = runner.run(replayable["kwa_finance_billing_patch_hold"])
    assert any(action.transition_outcome == "validation_failed" for action in finance_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in finance_trace.browser_actions)
    assert any(action.submission_gate == "blocked" for action in finance_trace.browser_actions)


def test_episode_runner_emits_all_declared_artifacts_even_when_stage_count_is_smaller() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    episode = next(episode for episode in build_live_web_episodes() if episode.episode_id == "kwa_finance_live_comps_revision")
    trace = runner.run(episode)

    declared = {artifact.artifact_id for artifact in episode.artifacts}
    emitted = {version.artifact_id for version in trace.artifact_versions}
    assert declared == emitted


def test_multistage_deck_revisions_preserve_prior_bullets() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    episode = next(episode for episode in build_replayable_episodes() if episode.episode_id == "kwa_finance_partner_deck_revision")
    trace = runner.run(episode)

    latest = [version for version in trace.artifact_versions if version.artifact_id == "partner_deck"][-1]
    assert "safe_mode" in latest.content
    assert "invoice lock" in latest.content.lower()


def test_workspace_seed_generation_writes_browser_manifest(tmp_path: Path) -> None:
    import importlib.util

    script_path = ROOT / "scripts" / "make_knowledge_work_arena.py"
    spec = importlib.util.spec_from_file_location("make_knowledge_work_arena_for_workspace_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    original_data_root = module.DATA_ROOT
    try:
        module.DATA_ROOT = tmp_path / "knowledge_work"
        episodes = module.build_replayable_episodes()[:1]
        module._write_workspace_seeds(episodes)
        workspace_dir = module.DATA_ROOT / "workspaces" / episodes[0].workspace_id
        assert (workspace_dir / "browser_manifest.json").exists()
        manifest = (workspace_dir / "browser_manifest.json").read_text(encoding="utf-8")
        assert "browser_surfaces" in manifest
        assert "state_machines" in manifest
    finally:
        module.DATA_ROOT = original_data_root


def test_browser_actions_emit_state_machine_transitions() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    episode = next(episode for episode in build_live_web_episodes() if episode.episode_id == "kwa_finance_live_committee_hold")
    trace = runner.run(episode)

    assert any(action.state_machine_id for action in trace.browser_actions)
    assert any(action.transition_id for action in trace.browser_actions)
    assert any(action.from_state and action.to_state for action in trace.browser_actions)
    assert any(action.transition_outcome == "validation_failed" for action in trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in trace.browser_actions)


def test_new_live_policy_hold_episodes_require_recovery_then_gate() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    live = {episode.episode_id: episode for episode in build_live_web_episodes()}

    jobs_trace = runner.run(live["kwa_jobs_live_screening_hold"])
    assert any(action.transition_outcome == "validation_failed" for action in jobs_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in jobs_trace.browser_actions)
    assert any(action.submission_gate == "approval_required" for action in jobs_trace.browser_actions)

    finance_trace = runner.run(live["kwa_finance_live_billing_patch_hold"])
    assert any(action.transition_outcome == "validation_failed" for action in finance_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in finance_trace.browser_actions)
    assert any(action.submission_gate == "blocked" for action in finance_trace.browser_actions)


def test_new_replayable_harder_human_episodes_preserve_latest_or_original_constraints() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    replayable = {episode.episode_id: episode for episode in build_replayable_episodes()}

    exec_trace = runner.run(replayable["kwa_exec_stale_brief_hold"])
    exec_artifact = next(version for version in exec_trace.artifact_versions if version.artifact_id == "stale_brief_packet")
    assert "latest version" in exec_artifact.content.lower()
    assert "stale v2" not in exec_artifact.content.lower()
    assert exec_trace.scorecard.browser_workflow_score > 0.95

    jobs_trace = runner.run(replayable["kwa_jobs_constraint_preservation_hold"])
    jobs_artifact = next(version for version in jobs_trace.artifact_versions if version.artifact_id == "constraint_preservation_packet")
    assert "Constraint Status: preserved" in jobs_artifact.content
    assert "Candidate Preference: remote_only" in jobs_artifact.content
    assert "onsite_only" not in jobs_artifact.content.lower()
    assert any(action.transition_outcome == "validation_failed" for action in jobs_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in jobs_trace.browser_actions)
    assert any(action.submission_gate == "approval_required" for action in jobs_trace.browser_actions)

    finance_trace = runner.run(replayable["kwa_finance_stale_assumption_hold"])
    finance_model = next(version for version in finance_trace.artifact_versions if version.artifact_id == "stale_assumption_model")
    assert "8.5m final" not in finance_model.content.lower()
    assert finance_model.file_path and finance_model.file_path.endswith(".xlsx")
    assert finance_trace.scorecard.browser_workflow_score > 0.95


def test_new_live_harder_human_episodes_record_recovery_then_safe_stop() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    live = {episode.episode_id: episode for episode in build_live_web_episodes()}

    exec_trace = runner.run(live["kwa_exec_live_stale_brief_hold"])
    assert any(action.transition_outcome == "validation_failed" for action in exec_trace.browser_actions)
    assert any(action.transition_outcome == "recovered" for action in exec_trace.browser_actions)
    assert any(action.submission_gate == "approval_required" for action in exec_trace.browser_actions)

    jobs_trace = runner.run(live["kwa_jobs_live_constraint_hold"])
    jobs_artifact = next(version for version in jobs_trace.artifact_versions if version.artifact_id == "live_constraint_packet")
    assert "Constraint Status: preserved" in jobs_artifact.content
    assert "onsite_only" not in jobs_artifact.content.lower()
    assert jobs_trace.scorecard.browser_workflow_score > 0.95

    finance_trace = runner.run(live["kwa_finance_live_stale_assumption_hold"])
    finance_note = next(version for version in finance_trace.artifact_versions if version.artifact_id == "live_stale_assumption_note")
    assert "8.5m final" not in finance_note.content.lower()
    assert any(action.submission_gate == "approval_required" for action in finance_trace.browser_actions)
    assert finance_trace.scorecard.browser_workflow_score > 0.95


def test_native_artifact_outputs_grade_against_real_files() -> None:
    tasks = _load_tasks()
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle)
    replayable = build_replayable_episodes()

    model_episode = next(episode for episode in replayable if episode.episode_id == "kwa_finance_three_statement_model")
    model_trace = runner.run(model_episode)
    model_artifact = next(version for version in model_trace.artifact_versions if version.artifact_id == "financial_model" and version.revision == 1)
    assert model_artifact.file_path and model_artifact.file_path.endswith(".xlsx")
    assert model_artifact.score >= 1.0

    jobs_episode = next(episode for episode in replayable if episode.episode_id == "kwa_jobs_submission_hold")
    jobs_trace = runner.run(jobs_episode)
    jobs_artifact = next(version for version in jobs_trace.artifact_versions if version.artifact_id == "validated_packet")
    assert jobs_artifact.file_path and jobs_artifact.file_path.endswith(".docx")
    assert jobs_artifact.score >= 1.0

    deck_episode = next(episode for episode in replayable if episode.episode_id == "kwa_finance_partner_deck_revision")
    deck_trace = runner.run(deck_episode)
    deck_artifact = [version for version in deck_trace.artifact_versions if version.artifact_id == "partner_deck"][-1]
    assert deck_artifact.file_path and deck_artifact.file_path.endswith(".pptx")
    assert deck_artifact.score >= 1.0

    model_native = inspect_artifact(model_artifact.file_path)
    assert model_native["formula_cells"]["E2"] == "=BASE_REVENUE+DELTA"
    assert model_native["formula_cells"]["E3"] == "=BASE_EXPENSE+MARKETING_INCREASE"

    deck_native = inspect_artifact(deck_artifact.file_path)
    assert any("invoice lock" in bullet.lower() for bullet in deck_native["slide_bullets"]["Situation"])
    assert any("safe_mode" in bullet.lower() for bullet in deck_native["slide_bullets"]["Recommendation"])


def test_history_intent_inference_distinguishes_canonical_and_exploratory() -> None:
    canonical = HISTORY_MODULE._infer_run_intent(Path("/tmp/replayable_core"), {"lane": "replayable_core"})
    exploratory = HISTORY_MODULE._infer_run_intent(
        Path("/tmp/model_backed_hf_specialists_policy_replayable_v6"),
        {"lane": "replayable_core"},
    )
    explicit = HISTORY_MODULE._infer_run_intent(
        Path("/tmp/custom"),
        {"lane": "replayable_core", "run_intent": "canonical"},
    )

    assert canonical == "canonical"
    assert exploratory == "exploratory"
    assert explicit == "canonical"


def test_history_markdown_report_separates_canonical_and_exploratory_sections() -> None:
    report = {
        "generated_at": "2026-04-10T00:00:00+00:00",
        "total_runs": 2,
        "snapshot_count": 2,
        "latest_canonical_by_lane": [
            {
                "lane": "replayable_core",
                "real_world_readiness_avg": 0.9,
                "browser_workflow_avg": 1.0,
                "strict_interface_avg": 1.0,
                "recovered_execution_avg": 1.0,
                "output_dir": "/tmp/replayable_core",
            }
        ],
        "latest_exploratory_by_lane": [
            {
                "lane": "replayable_core",
                "real_world_readiness_avg": 0.8,
                "browser_workflow_avg": 0.9,
                "strict_interface_avg": 1.0,
                "recovered_execution_avg": 0.9,
                "output_dir": "/tmp/replayable_core_exploratory",
            }
        ],
        "best_by_lane": [],
    }

    markdown = HISTORY_MODULE._markdown_report(report)

    assert "## Latest Canonical by Lane" in markdown
    assert "## Latest Exploratory by Lane" in markdown
    assert "/tmp/replayable_core_exploratory" in markdown


def test_run_knowledge_work_arena_defaults_to_full_lane() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["run_knowledge_work_arena.py"]
        args = RUN_KWA_MODULE.parse_args()
    finally:
        sys.argv = original_argv

    assert args.limit is None


def test_run_knowledge_work_arena_accepts_system_id() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["run_knowledge_work_arena.py", "--system-id", "hf_service_gemma4_specialists_cpu"]
        args = RUN_KWA_MODULE.parse_args()
    finally:
        sys.argv = original_argv

    assert args.system_id == "hf_service_gemma4_specialists_cpu"


def test_run_knowledge_work_arena_infers_direct_hf_system_ids() -> None:
    reasoner_args = argparse.Namespace(
        system_id=None,
        backend="hf",
        reasoner_backend="hf",
        router_backend="heuristic",
        retriever_backend="heuristic",
        reasoner="google/gemma-4-E2B-it",
        router="google/functiongemma-270m-it",
        retriever="google/embeddinggemma-300m",
    )
    specialists_args = argparse.Namespace(
        system_id=None,
        backend="hf",
        reasoner_backend="hf",
        router_backend="hf",
        retriever_backend="hf",
        reasoner="google/gemma-4-E2B-it",
        router="google/functiongemma-270m-it",
        retriever="google/embeddinggemma-300m",
    )

    assert RUN_KWA_MODULE._infer_system_id(reasoner_args) == "hf_gemma4_e2b_reasoner_only"
    assert RUN_KWA_MODULE._infer_system_id(specialists_args) == "hf_gemma4_e2b_specialists_cpu"
