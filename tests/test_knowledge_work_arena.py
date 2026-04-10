from __future__ import annotations

import importlib.util
from pathlib import Path

from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.knowledge_work.exporters import export_episode_leaderboard_csv
from gemma4_capability_map.knowledge_work.loader import load_episodes
from gemma4_capability_map.knowledge_work.replay import load_episode_traces, summarize_episode_traces
from gemma4_capability_map.knowledge_work.runner import EpisodeRunner
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


def _load_tasks() -> list[Task]:
    from gemma4_capability_map.io import load_jsonl

    tasks: list[Task] = []
    for path in sorted((ROOT / "data" / "gold").glob("*.jsonl")):
        tasks.extend(load_jsonl(path, Task))
    return tasks


def test_episode_specs_validate_and_cover_both_lanes() -> None:
    replayable = build_replayable_episodes()
    live = build_live_web_episodes()
    assert len(replayable) == 15
    assert len(live) == 9
    assert {episode.role_family.value for episode in replayable} == {
        "executive_assistant",
        "job_application_ops",
        "finance",
    }
    assert any(artifact.scoring_contract.required_table_rows for episode in replayable for artifact in episode.artifacts)
    assert any(artifact.scoring_contract.required_field_pairs for episode in replayable for artifact in episode.artifacts)
    assert any(artifact.scoring_contract.required_slide_titles for episode in replayable for artifact in episode.artifacts)
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
    assert all(action.action in {"open_public_page", "capture_notes", "prepare_sandbox_submission"} for action in trace.browser_actions)
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
