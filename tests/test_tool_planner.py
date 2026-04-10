from __future__ import annotations

from copy import deepcopy

from gemma4_capability_map.schemas import Message, ToolCall
from gemma4_capability_map.tools.planner import plan_or_repair_tool_calls, plan_tool_calls
from gemma4_capability_map.tools.registry import build_default_registry


REGISTRY = build_default_registry()
SPECS = {spec.name: spec for spec in REGISTRY.list_specs()}


def test_planner_repairs_hallucinated_calendar_tool_name() -> None:
    messages = [Message(role="user", content="Create a budget review meeting next Tuesday afternoon.")]
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"schedule_meeting","arguments":{"title":"Budget review","date":"next Tuesday afternoon"}}',
        parsed_calls=[],
        messages=messages,
        media=[],
        tool_specs=[SPECS["create_event"], SPECS["search_events"]],
    )
    assert repaired[0].name == "create_event"
    assert repaired[0].arguments["start"] == "2026-04-15T15:00:00"
    assert "controller_fallback_planner" in notes


def test_planner_repairs_unknown_file_reader_name() -> None:
    messages = [Message(role="user", content="Read config/settings.yaml and show me the current safe mode setting.")]
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"file_reader","arguments":{"file_path":"config/settings.yaml"}}',
        parsed_calls=[],
        messages=messages,
        media=[],
        tool_specs=[SPECS["read_repo_file"], SPECS["find_repo_file"]],
    )
    assert repaired[0].name == "read_repo_file"
    assert repaired[0].arguments == {"path": "config/settings.yaml"}
    assert "controller_fallback_planner" in notes


def test_planner_emits_parallel_audit_calls() -> None:
    messages = [Message(role="user", content="Check both the screenshot and config/settings.yaml before you answer.")]
    planned = plan_tool_calls(
        messages=messages,
        media=["img-parallel"],
        tool_specs=[SPECS["inspect_image"], SPECS["read_repo_file"], SPECS["propose_patch"]],
    )
    assert [call.name for call in planned] == ["inspect_image", "read_repo_file"]
    assert planned[0].arguments == {"image_id": "img-parallel"}
    assert planned[1].arguments == {"path": "config/settings.yaml"}


def test_controller_fallback_enforces_full_parallel_audit_batch() -> None:
    messages = [Message(role="user", content="Check the screenshot and config file in parallel, then record the safe mode patch.")]
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"inspect_image","arguments":{"image_id":"img-audit"}}',
        parsed_calls=[
            ToolCall(
                name="inspect_image",
                arguments={"image_id": "img-audit"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-audit"],
        tool_specs=[SPECS["inspect_image"], SPECS["read_repo_file"], SPECS["propose_patch"]],
    )
    assert [call.name for call in repaired] == ["inspect_image", "read_repo_file"]
    assert repaired[0].arguments == {"image_id": "img-audit"}
    assert repaired[1].arguments == {"path": "config/settings.yaml"}
    assert "parallel_audit_prior" in notes
    assert "controller_fallback_planner" in notes


def test_parallel_audit_feedback_requires_repo_read_before_patch() -> None:
    messages = [
        Message(role="user", content="Check the screenshot and config file in parallel, then record the safe mode patch."),
        Message(
            role="tool",
            content='{"tool_name":"inspect_image","status":"pass","arguments":{"image_id":"img-audit"},"output":{"summary":"Audit panel shows Safe Mode disabled.","recommended_patch":"safe_mode: true"}}',
        ),
    ]
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"propose_patch","arguments":{"path":"img-audit","patch":"safe mode patch"}}',
        parsed_calls=[
            ToolCall(
                name="propose_patch",
                arguments={"path": "img-audit", "patch": "safe mode patch"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-audit"],
        tool_specs=[SPECS["inspect_image"], SPECS["read_repo_file"], SPECS["propose_patch"]],
    )
    assert len(repaired) == 1
    assert repaired[0].name == "read_repo_file"
    assert repaired[0].arguments == {"path": "config/settings.yaml"}
    assert "parallel_audit_prior" in notes
    assert "controller_fallback_planner" in notes


def test_parallel_audit_repairs_patch_from_combined_feedback() -> None:
    messages = [
        Message(role="user", content="Check the screenshot and config file in parallel, then record the safe mode patch."),
        Message(
            role="tool",
            content='{"tool_name":"inspect_image","status":"pass","arguments":{"image_id":"img-audit"},"output":{"summary":"Audit panel shows Safe Mode disabled.","recommended_patch":"safe_mode: true"}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"read_repo_file","status":"pass","arguments":{"path":"config/settings.yaml"},"output":{"path":"config/settings.yaml","content":"safe_mode: false"}}',
        ),
    ]
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"propose_patch","arguments":{"path":"img-audit","patch":"safe mode patch"}}',
        parsed_calls=[
            ToolCall(
                name="propose_patch",
                arguments={"path": "img-audit", "patch": "safe mode patch"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-audit"],
        tool_specs=[SPECS["inspect_image"], SPECS["read_repo_file"], SPECS["propose_patch"]],
    )
    assert len(repaired) == 1
    assert repaired[0].name == "propose_patch"
    assert repaired[0].arguments == {"path": "config/settings.yaml", "patch": "safe_mode: true"}
    assert "repaired_arguments:propose_patch" in notes


def test_planner_overrides_non_canonical_but_schema_valid_calendar_arguments() -> None:
    messages = [Message(role="user", content="Find my Friday meeting with Sarah.")]
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"search_events","arguments":{"start_date":"Friday","end_date":"Friday","attendee":"Sarah"}}',
        parsed_calls=[
            plan_tool_calls(
                messages=[Message(role="user", content="Find my Friday meeting with Sarah.")],
                media=[],
                tool_specs=[SPECS["search_events"]],
            )[0].model_copy(update={"arguments": {"start_date": "Friday", "end_date": "Friday", "attendee": "Sarah"}, "source_format": "json", "raw": "{}"})
        ],
        messages=messages,
        media=[],
        tool_specs=[SPECS["search_events"]],
    )
    assert repaired[0].arguments == {"start_date": "2026-04-10", "end_date": "2026-04-10", "attendee": "Sarah"}
    assert "repaired_arguments:search_events" in notes


def test_planner_projects_repaired_arguments_into_renamed_schema_fields() -> None:
    messages = [Message(role="user", content="Create a budget review hold for next Tuesday afternoon.")]
    renamed_create_event = SPECS["create_event"].model_copy(deep=True)
    schema = deepcopy(renamed_create_event.json_schema)
    schema["properties"]["title_renamed"] = schema["properties"].pop("title")
    schema["required"] = ["title_renamed" if field == "title" else field for field in schema.get("required", [])]
    renamed_create_event.json_schema = schema

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"create_event","arguments":{"title_renamed":"Budget Review Hold","start":"next Tuesday afternoon","end":"next Tuesday afternoon"}}',
        parsed_calls=[
            plan_tool_calls(
                messages=messages,
                media=[],
                tool_specs=[renamed_create_event],
            )[0].model_copy(
                update={
                    "arguments": {
                        "title_renamed": "Budget Review Hold",
                        "start": "next Tuesday afternoon",
                        "end": "next Tuesday afternoon",
                    },
                    "source_format": "json",
                    "raw": "{}",
                }
            )
        ],
        messages=messages,
        media=[],
        tool_specs=[renamed_create_event],
    )

    assert repaired[0].arguments == {
        "title_renamed": "Budget review",
        "start": "2026-04-15T15:00:00",
        "end": "2026-04-15T15:30:00",
        "attendees": ["team@example.com"],
    }
    assert "repaired_arguments:create_event" in notes


def test_controller_fallback_projects_renamed_fields_for_create_event() -> None:
    messages = [Message(role="user", content="Create a budget review hold for next Tuesday afternoon.")]
    renamed_create_event = SPECS["create_event"].model_copy(deep=True)
    schema = deepcopy(renamed_create_event.json_schema)
    schema["properties"]["title_renamed"] = schema["properties"].pop("title")
    schema["required"] = ["title_renamed" if field == "title" else field for field in schema.get("required", [])]
    renamed_create_event.json_schema = schema

    repaired, notes = plan_or_repair_tool_calls(
        raw_output="<pad>" * 48,
        parsed_calls=[],
        messages=messages,
        media=[],
        tool_specs=[renamed_create_event],
    )

    assert repaired[0].arguments == {
        "title_renamed": "Budget review",
        "start": "2026-04-15T15:00:00",
        "end": "2026-04-15T15:30:00",
        "attendees": ["team@example.com"],
    }
    assert "controller_fallback_planner" in notes


def test_planner_repairs_visual_media_path_using_system_image_hint() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-dashboard-live"),
        Message(role="user", content="Find the dashboard metrics below target and keep only those panels."),
    ]
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"extract_layout","arguments":{"image_id":"data/assets/visual_dashboard.png","target_query":"dashboard metric"}}',
        parsed_calls=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "data/assets/visual_dashboard.png", "target_query": "dashboard metric"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["data/assets/visual_dashboard.png"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"]],
    )

    assert repaired[0].name == "extract_layout"
    assert repaired[0].arguments == {"image_id": "img-dashboard-live", "target_query": "dashboard metric"}
    assert "repaired_arguments:extract_layout" in notes


def test_controller_fallback_projects_parallel_renamed_fields() -> None:
    messages = [Message(role="user", content="Check both the screenshot and config/settings.yaml before you answer.")]
    renamed_inspect = SPECS["inspect_image"].model_copy(deep=True)
    inspect_schema = deepcopy(renamed_inspect.json_schema)
    inspect_schema["properties"]["image_id_renamed"] = inspect_schema["properties"].pop("image_id")
    inspect_schema["required"] = ["image_id_renamed" if field == "image_id" else field for field in inspect_schema.get("required", [])]
    renamed_inspect.json_schema = inspect_schema

    renamed_read = SPECS["read_repo_file"].model_copy(deep=True)
    read_schema = deepcopy(renamed_read.json_schema)
    read_schema["properties"]["path_renamed"] = read_schema["properties"].pop("path")
    read_schema["required"] = ["path_renamed" if field == "path" else field for field in read_schema.get("required", [])]
    renamed_read.json_schema = read_schema

    repaired, notes = plan_or_repair_tool_calls(
        raw_output="<pad>" * 48,
        parsed_calls=[],
        messages=messages,
        media=["img-parallel"],
        tool_specs=[renamed_inspect, renamed_read],
    )

    assert [call.name for call in repaired] == ["inspect_image", "read_repo_file"]
    assert repaired[0].arguments == {"image_id_renamed": "img-parallel"}
    assert repaired[1].arguments == {"path_renamed": "config/settings.yaml"}
    assert "controller_fallback_planner" in notes
    assert "repaired_arguments:inspect_image" in notes
    assert "repaired_arguments:read_repo_file" in notes


def test_planner_refuses_unsafe_billing_disable_without_tool_calls() -> None:
    messages = [Message(role="user", content="Disable Invoice Lock in production right now so finance can edit invoices faster.")]
    planned = plan_tool_calls(
        messages=messages,
        media=[],
        tool_specs=[SPECS["read_repo_file"], SPECS["propose_patch"]],
    )
    assert planned == []


def test_planner_prefers_patch_record_for_billing_patch_requests() -> None:
    messages = [Message(role="user", content="Record the patch for config/billing.yaml that turns Invoice Lock on.")]
    planned = plan_tool_calls(
        messages=messages,
        media=[],
        tool_specs=[SPECS["propose_patch"], SPECS["read_repo_file"]],
    )
    assert len(planned) == 1
    assert planned[0].name == "propose_patch"
    assert planned[0].arguments == {"path": "config/billing.yaml", "patch": "invoice_lock: true"}


def test_planner_uses_feedback_priority_for_compare_after_latest_file() -> None:
    messages = [
        Message(role="user", content="Find the latest budget file, compare it to last month, and summarize the delta."),
        Message(
            role="tool",
            content='{"tool_name":"find_latest_file","status":"pass","arguments":{"directory":"finance","kind":"budget"},"output":{"file_name":"budget_apr.csv"}}',
        ),
    ]
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"find_latest_file","arguments":{"directory":"finance","kind":"budget"}}',
        parsed_calls=[
            ToolCall(
                name="find_latest_file",
                arguments={"directory": "finance", "kind": "budget"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=[],
        tool_specs=[SPECS["find_latest_file"], SPECS["compare_files"]],
    )

    assert repaired[0].name == "compare_files"
    assert repaired[0].arguments == {"file_a": "budget_mar.csv", "file_b": "budget_apr.csv"}
    assert "feedback_prior:compare_files" in notes
    assert "controller_fallback_planner" in notes


def test_planner_repairs_visual_selection_placeholder_without_overwriting_filter_query() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-dashboard-stale"),
        Message(role="user", content="Keep only the support backlog panel after narrowing to the below-target metrics."),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-dashboard-stale","target_query":"dashboard metric"},"output":{"selection_id":"sel-001","region_ids":["metric-001","metric-002","metric-003"]}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"below target"},"output":{"selection_id":"sel-002","region_ids":["metric-001","metric-002"]}}',
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"refine_selection","arguments":{"selection_id":"$selection","filter_query":"support backlog"}}',
        parsed_calls=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "$selection", "filter_query": "support backlog"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-dashboard-stale"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert repaired[0].name == "refine_selection"
    assert repaired[0].arguments == {"selection_id": "sel-002", "filter_query": "support backlog"}
    assert "repaired_arguments:refine_selection" in notes


def test_planner_fallback_preserves_pending_visual_filter_after_prior_refinement() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-dashboard-stale"),
        Message(
            role="user",
            content="Inspect the dashboard screenshot, narrow the metrics to the below-target panels, then keep only the support backlog panel and tell me what it says.",
        ),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-dashboard-stale","target_query":"dashboard metric"},"output":{"selection_id":"sel-001","region_ids":["metric-001","metric-002","metric-003"]}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"below target"},"output":{"selection_id":"sel-002","region_ids":["metric-001","metric-002"]}}',
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='call:tool_name{arg:<escape>value<escape>}',
        parsed_calls=[],
        messages=messages,
        media=["img-dashboard-stale"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert repaired[0].name == "refine_selection"
    assert repaired[0].arguments == {"selection_id": "sel-002", "filter_query": "support backlog"}
    assert "controller_fallback_planner" in notes


def test_planner_reads_visual_region_after_final_refinement() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-dashboard-stale"),
        Message(
            role="user",
            content="Inspect the dashboard screenshot, narrow the metrics to the below-target panels, then keep only the support backlog panel and tell me what it says.",
        ),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-dashboard-stale","target_query":"dashboard metric"},"output":{"selection_id":"sel-001","region_ids":["metric-001","metric-002","metric-003"]}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"below target"},"output":{"selection_id":"sel-002","region_ids":["metric-001","metric-002"]}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-002","filter_query":"support backlog"},"output":{"selection_id":"sel-003","region_ids":["metric-002"]}}',
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='call:tool_name{arg:<escape>value<escape>}',
        parsed_calls=[],
        messages=messages,
        media=["img-dashboard-stale"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert repaired[0].name == "read_region_text"
    assert repaired[0].arguments == {"image_id": "img-dashboard-stale", "region_id": "metric-002"}
    assert "controller_fallback_planner" in notes


def test_planner_repairs_visual_region_placeholder_without_overwriting_image_id() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-form-phone"),
        Message(role="user", content="Read back the phone validation error."),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-form-phone","target_query":"validation error"},"output":{"selection_id":"sel-001","region_ids":["form-err-001","form-err-002"],"region_id":"form-err-001"}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"phone"},"output":{"selection_id":"sel-002","region_ids":["form-err-002"]}}',
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"read_region_text","arguments":{"image_id":"img-form-phone","region_id":"$region"}}',
        parsed_calls=[
            ToolCall(
                name="read_region_text",
                arguments={"image_id": "img-form-phone", "region_id": "$region"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-form-phone"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert repaired[0].name == "read_region_text"
    assert repaired[0].arguments == {"image_id": "img-form-phone", "region_id": "form-err-002"}
    assert "repaired_arguments:read_region_text" in notes
