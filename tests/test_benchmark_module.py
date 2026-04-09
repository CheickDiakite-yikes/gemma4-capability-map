from __future__ import annotations

import json

from gemma4_capability_map.benchmark import (
    load_tasks,
    load_variants,
    plan_runs,
    run_benchmark,
    runtime_bundle_snapshot,
)
from gemma4_capability_map.pipelines.base import RuntimeBundle


def test_load_tasks_can_filter_by_task_id() -> None:
    tasks = load_tasks(track="thinking", task_ids=["think_001_math"])
    assert [task.task_id for task in tasks] == ["think_001_math"]


def test_load_variants_returns_clean_variants_for_non_generated_runs() -> None:
    tasks = load_tasks(track="thinking", task_ids=["think_001_math"])
    variants = load_variants(tasks, include_generated=False)
    assert len(variants) == 1
    assert variants[0].variant_id == "think_001_math_clean"


def test_run_benchmark_emits_trace_callbacks() -> None:
    tasks = load_tasks(track="thinking", task_ids=["think_001_math"])
    variants = load_variants(tasks, include_generated=False)
    observed: list[tuple[str, int, int]] = []

    traces = run_benchmark(
        tasks=tasks,
        variants=variants,
        pipeline_name="monolith",
        backend="oracle",
        reasoner_backend=None,
        router_backend=None,
        retriever_backend=None,
        reasoner_id="google/gemma-4-E2B-it",
        router_id="google/functiongemma-270m-it",
        retriever_id="google/embeddinggemma-300m",
        reasoner_device="auto",
        reasoner_max_new_tokens=16,
        planning_max_new_tokens=8,
        final_max_new_tokens=32,
        limit=1,
        thinking_enabled=False,
        on_trace=lambda trace, completed, total: observed.append((trace.task_id, completed, total)),
    )

    assert len(traces) == 1
    assert observed == [("think_001_math", 1, 1)]
    assert traces[0].prompt_artifacts["planning_max_new_tokens"] == 8
    assert traces[0].prompt_artifacts["final_max_new_tokens"] == 32
    assert traces[0].prompt_artifacts["final_runtime_metadata"]["backend"] == "oracle"


def test_plan_runs_uses_clean_variant_when_generated_variants_are_disabled() -> None:
    tasks = load_tasks(track="thinking", task_ids=["think_001_math", "think_002_discount"])
    variants = load_variants(tasks, include_generated=False)
    planned = plan_runs(tasks, variants)

    assert [(task.task_id, variant.variant_id) for task, variant in planned] == [
        ("think_001_math", "think_001_math_clean"),
        ("think_002_discount", "think_002_discount_clean"),
    ]


class _FakeRuntimeComponent:
    def __init__(self, info):  # noqa: ANN001
        self._info = info

    def runtime_info(self):  # noqa: ANN201
        return self._info


def test_runtime_bundle_snapshot_includes_service_state(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"status": "loading", "phase": "hf_model_load_start"}), encoding="utf-8")
    bundle = RuntimeBundle(
        reasoner=_FakeRuntimeComponent(
            {
                "backend": "hf_service",
                "service": {
                    "state_path": str(state_path),
                    "socket_path": str(tmp_path / "service.sock"),
                },
            }
        ),
        router=None,
        retriever=None,
        executor=None,
    )

    snapshot = runtime_bundle_snapshot(bundle)

    assert snapshot["reasoner"]["backend"] == "hf_service"
    assert snapshot["reasoner"]["service_state"]["phase"] == "hf_model_load_start"
