from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.benchmark import load_tasks, load_variants, run_benchmark
from gemma4_capability_map.evals.visual_eval import score_visual_trace
from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.schemas import (
    Domain,
    ExpectedEvent,
    HardwareProfile,
    Message,
    ModelBundleSpec,
    RunTrace,
    Task,
    ToolResult,
    Track,
)
from gemma4_capability_map.tools.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[1]


def test_visual_gold_tasks_load_and_cover_both_lanes() -> None:
    tasks = load_jsonl(ROOT / "data" / "gold" / "visual_tools.jsonl", Task)
    assert len(tasks) == 18
    assert sum("replayable_core" in task.benchmark_tags for task in tasks) == 11
    assert sum("live_web_stress" in task.benchmark_tags for task in tasks) == 7
    assert any("visual_ui_ops" in task.benchmark_tags for task in tasks)
    assert any("visual_document" in task.benchmark_tags for task in tasks)
    assert any("visual_aerial" in task.benchmark_tags for task in tasks)


def test_seeded_visual_registry_can_segment_and_refine() -> None:
    registry = build_default_registry()
    state = {
        "visual_executor_mode": "seeded",
        "images": {
            "img-parking": {
                "entities": [
                    {"entity_id": "veh-001", "label": "vehicle", "attributes": {"color": "white"}},
                    {"entity_id": "veh-002", "label": "vehicle", "attributes": {"color": "green"}},
                ],
                "layouts": [],
            }
        },
    }

    state, segmented = registry.execute(state, "segment_entities", {"image_id": "img-parking", "entity_query": "vehicle"})
    assert segmented["count"] == 2
    assert segmented["selection_id"].startswith("sel-")

    state, refined = registry.execute(state, "refine_selection", {"selection_id": segmented["selection_id"], "filter_query": "white"})
    assert refined["count"] == 1
    assert refined["parent_selection_id"] == segmented["selection_id"]
    assert refined["entity_ids"] == ["veh-001"]


def test_local_visual_registry_can_extract_layout_and_read_region_text() -> None:
    registry = build_default_registry()
    state = {
        "visual_executor_mode": "local",
        "images": {
            "img-invoice-live": {
                "entities": [],
                "layouts": [],
                "local_layouts": [
                    {
                        "region_id": "invoice-table-001",
                        "label": "invoice totals table",
                        "text": "Total $51,840",
                        "attributes": {"section": "total", "kind": "table"},
                    }
                ],
            }
        },
    }

    state, extracted = registry.execute(
        state,
        "extract_layout",
        {"image_id": "img-invoice-live", "target_query": "invoice totals table"},
    )
    assert extracted["region_id"] == "invoice-table-001"
    _, read_back = registry.execute(
        state,
        "read_region_text",
        {"image_id": "img-invoice-live", "region_id": "invoice-table-001"},
    )
    assert "51,840" in read_back["text"]


def test_visual_oracle_run_emits_visual_metrics() -> None:
    tasks = load_tasks(track="visual_tool_orchestration", task_ids=["visual_006_parking_white_vehicles"])
    variants = load_variants(tasks, include_generated=False)
    traces = run_benchmark(
        tasks=tasks,
        variants=variants,
        pipeline_name="modular",
        backend="oracle",
        reasoner_backend=None,
        router_backend=None,
        retriever_backend=None,
        reasoner_id="google/gemma-4-E2B-it",
        router_id="google/functiongemma-270m-it",
        retriever_id="google/embeddinggemma-300m",
        reasoner_device="auto",
        reasoner_max_new_tokens=32,
        planning_max_new_tokens=24,
        final_max_new_tokens=48,
        limit=1,
        thinking_enabled=False,
    )

    assert len(traces) == 1
    metrics = traces[0].metrics
    assert metrics["tool_sequence_exactness"] == 1.0
    assert metrics["selection_accuracy"] == 1.0
    assert metrics["count_accuracy"] == 1.0
    assert metrics["referent_retention"] == 1.0
    assert metrics["unnecessary_tool_rate"] == 0.0
    assert metrics["success"] == 1.0


def test_visual_oracle_run_handles_stale_selection_recovery() -> None:
    tasks = load_tasks(track="visual_tool_orchestration", task_ids=["visual_013_dashboard_stale_selection_recovery"])
    variants = load_variants(tasks, include_generated=False)
    traces = run_benchmark(
        tasks=tasks,
        variants=variants,
        pipeline_name="modular",
        backend="oracle",
        reasoner_backend=None,
        router_backend=None,
        retriever_backend=None,
        reasoner_id="google/gemma-4-E2B-it",
        router_id="google/functiongemma-270m-it",
        retriever_id="google/embeddinggemma-300m",
        reasoner_device="auto",
        reasoner_max_new_tokens=32,
        planning_max_new_tokens=24,
        final_max_new_tokens=48,
        limit=1,
        thinking_enabled=False,
    )

    assert len(traces) == 1
    metrics = traces[0].metrics
    assert metrics["tool_sequence_exactness"] == 1.0
    assert metrics["selection_accuracy"] == 1.0
    assert metrics["count_accuracy"] == 1.0
    assert metrics["referent_retention"] == 1.0
    assert metrics["stale_selection_recovery"] == 1.0
    assert metrics["final_answer_accuracy"] == 1.0
    assert metrics["success"] == 1.0


def test_visual_eval_requires_latest_selection_for_refinement() -> None:
    task = Task(
        task_id="visual_test_latest_selection",
        track=Track.VISUAL_TOOL_ORCHESTRATION,
        domain=Domain.SCREENSHOT,
        input_modalities=["text", "image"],
        user_goal="Keep only the support backlog metric.",
        messages=[Message(role="user", content="Keep only the support backlog metric.")],
        expected_events=[
            ExpectedEvent(event_type="tool_call", tool_name="extract_layout", arguments={"image_id": "img-dashboard", "target_query": "dashboard metric"}),
            ExpectedEvent(event_type="tool_call", tool_name="refine_selection", arguments={"selection_id": "$selection", "filter_query": "below target"}),
            ExpectedEvent(event_type="tool_call", tool_name="refine_selection", arguments={"selection_id": "$selection", "filter_query": "support backlog"}),
        ],
        expected_final_state={"visual_selection": {"region_ids": ["metric-002"], "count": 1}},
        expected_answer_contains=["support backlog"],
    )
    trace = RunTrace(
        run_id="run-visual-selection",
        task_id=task.task_id,
        variant_id="base",
        track=task.track,
        architecture="modular",
        model_bundle=ModelBundleSpec(reasoner="google/gemma-4-E2B-it"),
        backend="oracle",
        hardware_profile=HardwareProfile(platform="macOS", platform_version="15", machine="arm64", cpu_count=8, memory_gb=32.0),
        tool_steps=[
            ToolResult(step=1, selected_tool="extract_layout", arguments={"image_id": "img-dashboard", "target_query": "dashboard metric"}, output={"selection_id": "sel-001", "region_ids": ["metric-001", "metric-002"], "count": 2}),
            ToolResult(step=2, selected_tool="refine_selection", arguments={"selection_id": "sel-001", "filter_query": "below target"}, output={"selection_id": "sel-002", "region_ids": ["metric-001", "metric-002"], "count": 2}),
            ToolResult(step=3, selected_tool="refine_selection", arguments={"selection_id": "sel-001", "filter_query": "support backlog"}, output={"selection_id": "sel-003", "region_ids": ["metric-002"], "count": 1}),
        ],
        final_answer="Support backlog remains selected.",
    )

    metrics = score_visual_trace(task, trace)
    assert metrics["tool_sequence_exactness"] == 1.0
    assert metrics["arg_exact"] == 0.0
    assert metrics["referent_retention"] == 0.5
    assert metrics["stale_selection_recovery"] == 0.0
    assert metrics["success"] == 0.0


def test_visual_eval_requires_latest_region_for_readback() -> None:
    task = Task(
        task_id="visual_test_latest_region",
        track=Track.VISUAL_TOOL_ORCHESTRATION,
        domain=Domain.SCREENSHOT,
        input_modalities=["text", "image"],
        user_goal="Read the phone validation error.",
        messages=[Message(role="user", content="Read the phone validation error.")],
        expected_events=[
            ExpectedEvent(event_type="tool_call", tool_name="extract_layout", arguments={"image_id": "img-form", "target_query": "validation error"}),
            ExpectedEvent(event_type="tool_call", tool_name="refine_selection", arguments={"selection_id": "$selection", "filter_query": "phone"}),
            ExpectedEvent(event_type="tool_call", tool_name="read_region_text", arguments={"image_id": "img-form", "region_id": "$region"}),
        ],
        expected_final_state={"visual_selection": {"region_ids": ["form-err-002"], "count": 1}},
        expected_answer_contains=["phone", "invalid"],
    )
    trace = RunTrace(
        run_id="run-visual-region",
        task_id=task.task_id,
        variant_id="base",
        track=task.track,
        architecture="modular",
        model_bundle=ModelBundleSpec(reasoner="google/gemma-4-E2B-it"),
        backend="oracle",
        hardware_profile=HardwareProfile(platform="macOS", platform_version="15", machine="arm64", cpu_count=8, memory_gb=32.0),
        tool_steps=[
            ToolResult(step=1, selected_tool="extract_layout", arguments={"image_id": "img-form", "target_query": "validation error"}, output={"selection_id": "sel-001", "region_id": "form-err-001", "region_ids": ["form-err-001", "form-err-002"], "count": 2}),
            ToolResult(step=2, selected_tool="refine_selection", arguments={"selection_id": "sel-001", "filter_query": "phone"}, output={"selection_id": "sel-002", "region_ids": ["form-err-002"], "count": 1}),
            ToolResult(step=3, selected_tool="read_region_text", arguments={"image_id": "img-form", "region_id": "form-err-001"}, output={"image_id": "img-form", "region_id": "form-err-001", "text": "Work authorization required"}),
        ],
        final_answer="Phone number format invalid.",
    )

    metrics = score_visual_trace(task, trace)
    assert metrics["tool_sequence_exactness"] == 1.0
    assert metrics["arg_exact"] == 0.0
    assert metrics["success"] == 0.0
