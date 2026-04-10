from __future__ import annotations

import math
from pathlib import Path

from gemma4_capability_map.io import dump_jsonl, load_jsonl
from gemma4_capability_map.schemas import Task, Variant, VariantOverrides
from gemma4_capability_map.stressors.context import apply_context_variant
from gemma4_capability_map.stressors.efficiency import apply_efficiency_variant
from gemma4_capability_map.stressors.language import apply_language_variant
from gemma4_capability_map.stressors.schema import apply_schema_variant


ROOT = Path(__file__).resolve().parents[1]
VARIANT_PATH = ROOT / "data" / "variants" / "generated_variants.jsonl"

DEFAULT_STRESSOR_FLAVORS = {
    "thinking": {"language": "fr", "context": "stale_preference", "efficiency": "dim_256"},
    "tool_routing": {"language": "code_switch", "schema": "renamed_fields", "context": "irrelevant_tool_output"},
    "retrieval": {"language": "fr", "context": "long_history", "efficiency": "top_k_3"},
    "full_stack": {"language": "fr", "schema": "validator_feedback", "context": "changed_constraint", "efficiency": "truncation_compact"},
    "visual_tool_orchestration": {"language": "fr", "schema": "renamed_fields", "context": "changed_constraint"},
}


def main() -> None:
    tasks = load_gold_tasks()
    variants: list[Variant] = []
    crossed_budget = math.ceil(len(tasks) * 0.2)
    for index, task in enumerate(tasks):
        variants.append(Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id))
        flavors = DEFAULT_STRESSOR_FLAVORS[task.track.value]
        if "language" in flavors:
            variants.append(apply_language_variant(task, flavors["language"]))
        if "schema" in flavors and task.tool_specs:
            variants.append(apply_schema_variant(task, flavors["schema"]))
        if "context" in flavors:
            variants.append(apply_context_variant(task, flavors["context"]))
        if "efficiency" in flavors and task.track.value in {"thinking", "retrieval", "full_stack"}:
            variants.append(apply_efficiency_variant(task, flavors["efficiency"]))
        if index < crossed_budget:
            variants.append(build_crossed_variant(task, flavors))
    dump_jsonl(VARIANT_PATH, variants)
    print(f"Wrote {len(variants)} variants.")


def load_gold_tasks() -> list[Task]:
    tasks: list[Task] = []
    for path in sorted((ROOT / "data" / "gold").glob("*.jsonl")):
        tasks.extend(load_jsonl(path, Task))
    return tasks


def build_crossed_variant(task: Task, flavors: dict[str, str]) -> Variant:
    variant = Variant(
        variant_id=f"{task.task_id}_crossed",
        base_task_id=task.task_id,
        stressors={"language": None, "schema": None, "context": None, "efficiency": None},
    )
    if "language" in flavors:
        language_variant = apply_language_variant(task, flavors["language"])
        variant.overrides.messages = language_variant.overrides.messages
        variant.stressors["language"] = flavors["language"]
        variant.primary_stressor = language_variant.primary_stressor
    if "context" in flavors:
        context_variant = apply_context_variant(task, flavors["context"])
        variant.overrides.messages_prefix = context_variant.overrides.messages_prefix
        variant.stressors["context"] = flavors["context"]
        if variant.primary_stressor is None:
            variant.primary_stressor = context_variant.primary_stressor
        else:
            variant.secondary_stressor = context_variant.primary_stressor
    elif "schema" in flavors and task.tool_specs:
        schema_variant = apply_schema_variant(task, flavors["schema"])
        variant.overrides.tool_specs = schema_variant.overrides.tool_specs
        variant.overrides.expected_events = schema_variant.overrides.expected_events
        variant.stressors["schema"] = flavors["schema"]
        variant.secondary_stressor = schema_variant.primary_stressor
    if "efficiency" in flavors:
        efficiency_variant = apply_efficiency_variant(task, flavors["efficiency"])
        variant.overrides.efficiency = efficiency_variant.overrides.efficiency
        variant.stressors["efficiency"] = flavors["efficiency"]
        if variant.primary_stressor is not None and variant.secondary_stressor is None:
            variant.secondary_stressor = efficiency_variant.primary_stressor
    return variant


if __name__ == "__main__":
    main()
