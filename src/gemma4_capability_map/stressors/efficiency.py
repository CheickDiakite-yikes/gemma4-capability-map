from __future__ import annotations

from gemma4_capability_map.schemas import StressorKind, Task, Variant, VariantOverrides


EFFICIENCY_MAP = {
    "dim_128": {"embedding_dim": 128, "top_k": 5, "quantization": "int4"},
    "dim_256": {"embedding_dim": 256, "top_k": 5, "quantization": "int8"},
    "top_k_3": {"embedding_dim": 768, "top_k": 3, "quantization": "none"},
    "top_k_10": {"embedding_dim": 768, "top_k": 10, "quantization": "none"},
    "truncation_compact": {"embedding_dim": 512, "top_k": 5, "context_budget": 1024, "quantization": "none"},
}


def apply_efficiency_variant(task: Task, flavor: str) -> Variant:
    return Variant(
        variant_id=f"{task.task_id}_efficiency_{flavor}",
        base_task_id=task.task_id,
        primary_stressor=StressorKind.EFFICIENCY,
        stressors={"language": None, "schema": None, "context": None, "efficiency": flavor},
        overrides=VariantOverrides(efficiency=EFFICIENCY_MAP[flavor]),
    )

