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
    assert len(tasks) == 52
    counts = {track: sum(task.track.value == track for task in tasks) for track in {"thinking", "tool_routing", "retrieval", "full_stack"}}
    assert counts == {"thinking": 13, "tool_routing": 13, "retrieval": 13, "full_stack": 13}
    assert any(task.real_world_profile is not None for task in tasks)
    assert any("real_world" in task.benchmark_tags for task in tasks)


def test_generated_variants_validate() -> None:
    variants = load_jsonl(ROOT / "data" / "variants" / "generated_variants.jsonl", Variant)
    assert len(variants) == 232
    assert any(variant.secondary_stressor is not None for variant in variants)


def test_tool_specs_serialize_with_schema_alias() -> None:
    first_line = (ROOT / "data" / "gold" / "tools.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"schema"' in first_line
    assert '"json_schema"' not in first_line


def test_knowledge_work_episode_specs_validate() -> None:
    replayable = load_jsonl(ROOT / "data" / "knowledge_work" / "replayable_core" / "episodes.jsonl", Episode)
    live = load_jsonl(ROOT / "data" / "knowledge_work" / "live_web_stress" / "episodes.jsonl", Episode)
    assert len(replayable) == 15
    assert len(live) == 9
    assert all(stage.browser_plan for episode in replayable + live for stage in episode.stages)
    assert any(episode.browser_state_machines for episode in replayable + live)
    assert any(
        step.submission_gate == "approval_required"
        for episode in live
        for stage in episode.stages
        for step in stage.browser_plan
    )
