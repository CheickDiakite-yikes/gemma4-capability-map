from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.knowledge_work.schemas import Episode
from gemma4_capability_map.schemas import Task, Variant


ROOT = Path(__file__).resolve().parents[1]


def test_gold_tasks_validate() -> None:
    tasks: list[Task] = []
    for path in sorted((ROOT / "data" / "gold").glob("*.jsonl")):
        tasks.extend(load_jsonl(path, Task))
    assert len(tasks) == 91
    counts = {
        track: sum(task.track.value == track for task in tasks)
        for track in {"thinking", "tool_routing", "retrieval", "full_stack", "visual_tool_orchestration"}
    }
    assert counts == {
        "thinking": 13,
        "tool_routing": 22,
        "retrieval": 13,
        "full_stack": 13,
        "visual_tool_orchestration": 30,
    }
    assert any(task.real_world_profile is not None for task in tasks)
    assert any("real_world" in task.benchmark_tags for task in tasks)
    assert any("visual_tool_orchestration" in task.benchmark_tags for task in tasks)


def test_generated_variants_validate() -> None:
    variants = load_jsonl(ROOT / "data" / "variants" / "generated_variants.jsonl", Variant)
    assert len(variants) == 396
    assert any(variant.secondary_stressor is not None for variant in variants)
    assert any(variant.base_task_id.startswith("visual_") for variant in variants)


def test_tool_specs_serialize_with_schema_alias() -> None:
    first_line = (ROOT / "data" / "gold" / "tools.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"schema"' in first_line
    assert '"json_schema"' not in first_line


def test_knowledge_work_episode_specs_validate() -> None:
    replayable = load_jsonl(ROOT / "data" / "knowledge_work" / "replayable_core" / "episodes.jsonl", Episode)
    live = load_jsonl(ROOT / "data" / "knowledge_work" / "live_web_stress" / "episodes.jsonl", Episode)
    assert len(replayable) == 32
    assert len(live) == 26
    assert all(stage.browser_plan for episode in replayable + live for stage in episode.stages)
    assert any(episode.browser_state_machines for episode in replayable + live)
    assert any("visual_kwa" in episode.benchmark_tags for episode in replayable + live)
    assert any(
        artifact.scoring_contract.forbidden_fragments
        for episode in replayable + live
        for artifact in episode.artifacts
    )
    assert any(
        artifact.scoring_contract.required_formula_cells
        for episode in replayable + live
        for artifact in episode.artifacts
    )
    assert any(
        artifact.scoring_contract.required_heading_order
        for episode in replayable + live
        for artifact in episode.artifacts
    )
    assert any(
        artifact.scoring_contract.required_slide_bullets_by_title
        for episode in replayable + live
        for artifact in episode.artifacts
    )
    assert any(
        step.submission_gate == "approval_required"
        for episode in live
        for stage in episode.stages
        for step in stage.browser_plan
    )
    assert any(
        step.submission_gate == "blocked"
        for episode in replayable + live
        for stage in episode.stages
        for step in stage.browser_plan
    )
    assert any(
        step.transition_outcome == "validation_failed"
        for episode in replayable + live
        for stage in episode.stages
        for step in stage.browser_plan
    )
    assert any(
        step.transition_outcome == "recovered"
        for episode in replayable + live
        for stage in episode.stages
        for step in stage.browser_plan
    )
