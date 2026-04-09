from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.schemas import Task
from gemma4_capability_map.stressors.language import apply_language_variant
from gemma4_capability_map.stressors.schema import apply_schema_variant


ROOT = Path(__file__).resolve().parents[1]


def test_language_variant_rewrites_user_message() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "thinking.jsonl", Task) if task.task_id == "think_006_screenshot_security"][0]
    variant = apply_language_variant(task, "fr")
    assert variant.overrides.messages is not None
    assert variant.overrides.messages[0].content.startswith("Regarde la capture d'écran")


def test_schema_variant_renames_expected_arguments() -> None:
    task = load_jsonl(ROOT / "data" / "gold" / "tools.jsonl", Task)[0]
    variant = apply_schema_variant(task, "renamed_fields")
    assert variant.overrides.tool_specs is not None
    assert "start_date_renamed" in variant.overrides.tool_specs[0].json_schema["properties"]
    assert variant.overrides.expected_events is not None
    assert "start_date_renamed" in variant.overrides.expected_events[0].arguments
