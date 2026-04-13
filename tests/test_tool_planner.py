from __future__ import annotations

from copy import deepcopy

from gemma4_capability_map.research_controls import ResearchControls
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
    assert "controller_fallback_planner" not in notes


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
    assert "controller_fallback_planner" not in notes


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


def test_controller_repair_ablation_returns_raw_valid_call_without_argument_fixups() -> None:
    messages = [Message(role="user", content="Find my Friday meeting with Sarah.")]
    raw_call = plan_tool_calls(
        messages=messages,
        media=[],
        tool_specs=[SPECS["search_events"]],
    )[0].model_copy(
        update={
            "arguments": {"start_date": "Friday", "end_date": "Friday", "attendee": "Sarah"},
            "source_format": "json",
            "raw": "{}",
        }
    )

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"search_events","arguments":{"start_date":"Friday","end_date":"Friday","attendee":"Sarah"}}',
        parsed_calls=[raw_call],
        messages=messages,
        media=[],
        tool_specs=[SPECS["search_events"]],
        research_controls=ResearchControls(disable_controller_repair=True),
    )

    assert repaired[0].arguments == {"start_date": "Friday", "end_date": "Friday", "attendee": "Sarah"}
    assert notes == ["controller_repair_disabled"]


def test_controller_fallback_ablation_returns_no_calls_when_model_emits_none() -> None:
    messages = [Message(role="user", content="Create a budget review meeting next Tuesday afternoon.")]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output="<pad>" * 32,
        parsed_calls=[],
        messages=messages,
        media=[],
        tool_specs=[SPECS["create_event"], SPECS["search_events"]],
        research_controls=ResearchControls(disable_controller_fallback=True),
    )

    assert repaired == []
    assert "controller_fallback_disabled" in notes


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
    assert "controller_fallback_planner" not in notes


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


def test_planner_rejects_refine_selection_without_prior_visual_selection() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-form-live-phone"),
        Message(
            role="user",
            content="Using the local visual executor path, narrow the live application errors to the phone issue and read back that message.",
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"refine_selection","arguments":{"selection_id":"sel-001","filter_query":"phone"}}',
        parsed_calls=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-001", "filter_query": "phone"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-form-live-phone"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert repaired[0].name == "extract_layout"
    assert repaired[0].arguments == {"image_id": "img-form-live-phone", "target_query": "validation error"}
    assert "controller_fallback_planner" in notes


def test_planner_repairs_visual_region_from_latest_local_refinement_context() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-form-live-phone"),
        Message(
            role="user",
            content="Using the local visual executor path, narrow the live application errors to the phone issue and read back that message.",
        ),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-form-live-phone","target_query":"validation error"},"output":{"selection_id":"sel-001","region_ids":["form-err-001","form-err-002"],"region_id":"form-err-001"}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"phone"},"output":{"selection_id":"sel-002","image_id":"img-form-live-phone","region_ids":["form-err-002"]}}',
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='call:tool_name{arg:<escape>value<escape>}{"name":"read_region_text","arguments":{"image_id":"data/assets/visual_form.png","region_id":"sel-001"}}',
        parsed_calls=[
            ToolCall(
                name="read_region_text",
                arguments={"image_id": "data/assets/visual_form.png", "region_id": "sel-001"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-form-live-phone"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert repaired[0].name == "read_region_text"
    assert repaired[0].arguments == {"image_id": "img-form-live-phone", "region_id": "form-err-002"}
    assert "repaired_arguments:read_region_text" in notes


def test_planner_uses_slide_callout_target_for_policy_revision_tasks() -> None:
    calls = plan_tool_calls(
        messages=[
            Message(role="system", content="visual_image_ids: img-slide-policy"),
            Message(
                role="user",
                content="Inspect the slide callouts, keep only the action callout, and use it to revise the recommendation with the latest policy.",
            ),
        ],
        media=["img-slide-policy"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert calls[0].name == "extract_layout"
    assert calls[0].arguments == {"image_id": "img-slide-policy", "target_query": "slide callout"}


def test_planner_repairs_latest_approval_safe_action_followup_to_read_region_text() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-slide-policy"),
        Message(
            role="user",
            content="Inspect the slide callouts, ignore the earlier publication note, keep only the latest approval-safe action callout, and read it back.",
        ),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-slide-policy","target_query":"slide callout"},"output":{"selection_id":"sel-001","image_id":"img-slide-policy","selection_kind":"regions","count":3,"region_ids":["slide-action-101","slide-action-102","slide-risk-101"],"region_id":"slide-action-101"}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"latest action"},"output":{"selection_id":"sel-002","image_id":"img-slide-policy","selection_kind":"regions","count":1,"region_ids":["slide-action-102"],"region_id":"slide-action-102"}}',
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"refine_selection","arguments":{"selection_id":"sel-002","filter_query":"action"}}',
        parsed_calls=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-002", "filter_query": "action"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-slide-policy"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert len(repaired) == 1
    assert repaired[0].name == "read_region_text"
    assert repaired[0].arguments == {"image_id": "img-slide-policy", "region_id": "slide-action-102"}
    assert "feedback_prior:read_region_text" in notes
    assert "controller_fallback_planner" not in notes


def test_planner_repairs_wrong_visual_target_query_for_form_tasks() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-form-phone"),
        Message(role="user", content="Inspect the form screenshot, narrow to the phone validation error, then read that message."),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"extract_layout","arguments":{"image_id":"img-form-phone","target_query":"dashboard metric"}}',
        parsed_calls=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-form-phone", "target_query": "dashboard metric"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-form-phone"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert repaired[0].name == "extract_layout"
    assert repaired[0].arguments == {"image_id": "img-form-phone", "target_query": "validation error"}
    assert "repaired_arguments:extract_layout" in notes


def test_planner_forces_stepwise_visual_control_for_multi_call_batch() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-form-live-phone"),
        Message(
            role="user",
            content="Using the local visual executor path, narrow the live application errors to the phone issue and read back that message.",
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='[{"name":"refine_selection","arguments":{"selection_id":"","filter_query":"phone"}},{"name":"extract_layout","arguments":{"image_id":"img-form-live-phone","target_query":"dashboard metric"}},{"name":"refine_selection","arguments":{"selection_id":"sel-001","filter_query":"phone"}}]',
        parsed_calls=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "", "filter_query": "phone"},
                source_format="json",
                raw="{}",
            ),
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-form-live-phone", "target_query": "dashboard metric"},
                source_format="json",
                raw="{}",
            ),
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-001", "filter_query": "phone"},
                source_format="json",
                raw="{}",
            ),
        ],
        messages=messages,
        media=["img-form-live-phone"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert len(repaired) == 1
    assert repaired[0].name == "extract_layout"
    assert repaired[0].arguments == {"image_id": "img-form-live-phone", "target_query": "validation error"}
    assert "visual_stepwise_prior" in notes
    assert "controller_fallback_planner" not in notes


def test_planner_forces_read_region_after_final_visual_refinement() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-dashboard-followup"),
        Message(
            role="user",
            content="Inspect the dashboard metrics, keep only the needs review panels, then the customer ops panel, and tell me what it says.",
        ),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-dashboard-followup","target_query":"dashboard metric"},"output":{"selection_id":"sel-001","region_ids":["metric-101","metric-102","metric-103"]}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"needs review"},"output":{"selection_id":"sel-002","region_ids":["metric-101","metric-102"]}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-002","filter_query":"customer ops"},"output":{"selection_id":"sel-003","region_ids":["metric-102"]}}',
        ),
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='[{"name":"extract_layout","arguments":{"image_id":"img-dashboard-followup","target_query":"dashboard metric"}}]',
        parsed_calls=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-dashboard-followup", "target_query": "dashboard metric"},
                source_format="json",
                raw="{}",
            ),
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-001", "filter_query": "needs review"},
                source_format="json",
                raw="{}",
            ),
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-002", "filter_query": "customer ops"},
                source_format="json",
                raw="{}",
            ),
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-dashboard-followup", "target_query": "dashboard metric"},
                source_format="json",
                raw="{}",
            ),
        ],
        messages=messages,
        media=["img-dashboard-followup"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert len(repaired) == 1
    assert repaired[0].name == "read_region_text"
    assert repaired[0].arguments == {"image_id": "img-dashboard-followup", "region_id": "metric-102"}
    assert "visual_stepwise_prior" in notes
    assert "controller_fallback_planner" not in notes


def test_planner_requests_latest_filter_before_phone_followup() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-form-latest"),
        Message(
            role="user",
            content="Inspect the form errors, keep only the latest issue first, then narrow to the phone issue and read back the remaining message.",
        ),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-form-latest","target_query":"validation error"},"output":{"selection_id":"sel-001","image_id":"img-form-latest","selection_kind":"regions","count":2,"region_ids":["form-err-201","form-err-202"],"region_id":"form-err-201"}}',
        ),
    ]

    planned = plan_tool_calls(
        messages=messages,
        media=["img-form-latest"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert len(planned) == 1
    assert planned[0].name == "refine_selection"
    assert planned[0].arguments == {"selection_id": "sel-001", "filter_query": "latest"}


def test_planner_uses_extract_layout_for_live_phone_issue_cold_start() -> None:
    planned = plan_tool_calls(
        messages=[
            Message(role="system", content="visual_image_ids: img-form-live-phone"),
            Message(
                role="user",
                content="Using the local visual executor path, respect the latest recruiter note, isolate the phone issue first, and read back that message.",
            ),
        ],
        media=["img-form-live-phone"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert len(planned) == 1
    assert planned[0].name == "extract_layout"
    assert planned[0].arguments == {"image_id": "img-form-live-phone", "target_query": "validation error"}


def test_planner_stops_after_visual_answer_region_is_read() -> None:
    messages = [
        Message(role="system", content="visual_image_ids: img-form-latest"),
        Message(
            role="user",
            content="Inspect the form errors, keep only the latest issue first, then narrow to the phone issue and read back the remaining message.",
        ),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-form-latest","target_query":"validation error"},"output":{"selection_id":"sel-001","image_id":"img-form-latest","selection_kind":"regions","count":2,"region_ids":["form-err-201","form-err-202"],"region_id":"form-err-201"}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"latest"},"output":{"selection_id":"sel-002","image_id":"img-form-latest","selection_kind":"regions","count":1,"region_ids":["form-err-202"]}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-002","filter_query":"phone"},"output":{"selection_id":"sel-003","image_id":"img-form-latest","selection_kind":"regions","count":1,"region_ids":["form-err-202"]}}',
        ),
        Message(
            role="tool",
            content='{"tool_name":"read_region_text","status":"pass","arguments":{"image_id":"img-form-latest","region_id":"form-err-202"},"output":{"image_id":"img-form-latest","region_id":"form-err-202","text":"Phone number format invalid","label":"validation error"}}',
        ),
    ]

    planned = plan_tool_calls(
        messages=messages,
        media=["img-form-latest"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )
    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"extract_layout","arguments":{"image_id":"img-form-latest","target_query":"validation error"}}',
        parsed_calls=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-form-latest", "target_query": "validation error"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=["img-form-latest"],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert planned == []
    assert repaired == []
    assert notes == ["visual_complete"]


def test_planner_prefers_cli_patch_for_latest_phone_fix() -> None:
    messages = [
        Message(
            role="user",
            content="Ignore the earlier work-authorization edit. The newest recruiter instruction is to patch only the phone validation config in config/job_form.yaml.",
        )
    ]

    planned = plan_tool_calls(
        messages=messages,
        media=[],
        tool_specs=[SPECS["cli_apply_patch"], SPECS["read_repo_file"]],
    )

    assert len(planned) == 1
    assert planned[0].name == "cli_apply_patch"
    assert planned[0].arguments == {"path": "config/job_form.yaml", "patch": "phone_validation: strict"}


def test_planner_repairs_empty_api_fetch_record_arguments() -> None:
    messages = [
        Message(
            role="user",
            content="The earlier note said send the packet. Update: use the newest instruction and fetch BR-17 to confirm the latest approval-safe action.",
        )
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"api_fetch_record","arguments":{}}',
        parsed_calls=[
            ToolCall(
                name="api_fetch_record",
                arguments={},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=[],
        tool_specs=[SPECS["api_fetch_record"], SPECS["api_update_record"]],
    )

    assert repaired[0].name == "api_fetch_record"
    assert repaired[0].arguments == {"record_type": "briefing_record", "record_id": "BR-17"}
    assert "repaired_arguments:api_fetch_record" in notes


def test_planner_repairs_form_issue_api_fetch_record_arguments() -> None:
    messages = [
        Message(
            role="user",
            content="The first note mentioned work authorization, but the latest recruiter note says focus on the phone field. Fetch issue FORM-88 and identify the latest blocker.",
        )
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"api_fetch_record","arguments":{"record_type":"record","record_id":"FORM-88"}}',
        parsed_calls=[
            ToolCall(
                name="api_fetch_record",
                arguments={"record_type": "record", "record_id": "FORM-88"},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=[],
        tool_specs=[SPECS["api_fetch_record"]],
    )

    assert repaired[0].name == "api_fetch_record"
    assert repaired[0].arguments == {"record_type": "form_issue", "record_id": "FORM-88"}
    assert "repaired_arguments:api_fetch_record" in notes


def test_planner_repairs_empty_api_update_record_arguments() -> None:
    messages = [
        Message(
            role="user",
            content="Do not reopen publication. The latest finance direction is to update billing record INV-204 so invoice_lock stays on hold.",
        )
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"api_update_record","arguments":{}}',
        parsed_calls=[
            ToolCall(
                name="api_update_record",
                arguments={},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=[],
        tool_specs=[SPECS["api_update_record"], SPECS["api_fetch_record"]],
    )

    assert repaired[0].name == "api_update_record"
    assert repaired[0].arguments == {
        "record_type": "billing_record",
        "record_id": "INV-204",
        "field": "invoice_lock",
        "value": "hold",
    }
    assert "repaired_arguments:api_update_record" in notes


def test_planner_repairs_cli_search_logs_arguments() -> None:
    messages = [
        Message(
            role="user",
            content="Ignore the earlier publish plan. Search logs/billing.log for the latest invoice-lock failure and report it.",
        )
    ]

    repaired, notes = plan_or_repair_tool_calls(
        raw_output='{"name":"cli_search_logs","arguments":{}}',
        parsed_calls=[
            ToolCall(
                name="cli_search_logs",
                arguments={},
                source_format="json",
                raw="{}",
            )
        ],
        messages=messages,
        media=[],
        tool_specs=[SPECS["cli_search_logs"], SPECS["api_fetch_record"]],
    )

    assert repaired[0].name == "cli_search_logs"
    assert repaired[0].arguments == {"path": "logs/billing.log", "query": "invoice lock"}
    assert "repaired_arguments:cli_search_logs" in notes


def test_planner_prefers_cli_search_logs_for_latest_invoice_lock_failure() -> None:
    messages = [
        Message(
            role="user",
            content="Ignore the earlier publish plan. Search logs/billing.log for the latest invoice-lock failure and report it.",
        )
    ]

    planned = plan_tool_calls(
        messages=messages,
        media=[],
        tool_specs=[SPECS["cli_search_logs"], SPECS["api_fetch_record"]],
    )

    assert len(planned) == 1
    assert planned[0].name == "cli_search_logs"
    assert planned[0].arguments == {"path": "logs/billing.log", "query": "invoice lock"}


def test_planner_prefers_cli_inspect_diff_for_review_only_diff_instruction() -> None:
    messages = [
        Message(
            role="user",
            content="The earlier finance plan was to patch billing, but the latest approved direction is review-only. Inspect diff invoice_lock_resume_diff_v2 and confirm whether the approval banner keeps the committee hold explicit.",
        )
    ]

    planned = plan_tool_calls(
        messages=messages,
        media=[],
        tool_specs=[SPECS["cli_inspect_diff"], SPECS["cli_apply_patch"]],
    )

    assert len(planned) == 1
    assert planned[0].name == "cli_inspect_diff"
    assert planned[0].arguments == {"diff_id": "invoice_lock_resume_diff_v2"}


def test_planner_prefers_email_patch_when_latest_instruction_is_patch_only() -> None:
    messages = [
        Message(
            role="user",
            content="The earlier note focused on the phone field. Latest recruiter instruction: patch only config/job_form.yaml so email validation stays blocked until review. Do not reread the file or change work authorization.",
        )
    ]

    planned = plan_tool_calls(
        messages=messages,
        media=[],
        tool_specs=[SPECS["cli_apply_patch"], SPECS["read_repo_file"]],
    )

    assert len(planned) == 1
    assert planned[0].name == "cli_apply_patch"
    assert planned[0].arguments == {"path": "config/job_form.yaml", "patch": "email_validation: blocked"}


def test_planner_preserves_visual_filter_order_for_latest_blocked_email_chain() -> None:
    messages = [
        Message(
            role="user",
            content="Inspect the form errors, keep only the latest issues first, then keep only the blocked issues, then narrow to the email issue and read back the remaining message.",
        ),
        Message(
            role="tool",
            content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-form-blocked-email","target_query":"validation error"},"output":{"image_id":"img-form-blocked-email","selection_id":"sel-errors","region_ids":["form-err-401","form-err-402","form-err-403"],"count":3}}',
        ),
    ]

    latest_step = plan_tool_calls(
        messages=messages,
        media=[],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert latest_step[0].name == "refine_selection"
    assert latest_step[0].arguments == {"selection_id": "sel-errors", "filter_query": "latest"}

    blocked_messages = [
        *messages,
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-errors","filter_query":"latest"},"output":{"image_id":"img-form-blocked-email","selection_id":"sel-latest","region_ids":["form-err-402","form-err-403"],"count":2}}',
        ),
    ]
    blocked_step = plan_tool_calls(
        messages=blocked_messages,
        media=[],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert blocked_step[0].name == "refine_selection"
    assert blocked_step[0].arguments == {"selection_id": "sel-latest", "filter_query": "blocked"}

    email_messages = [
        *blocked_messages,
        Message(
            role="tool",
            content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-latest","filter_query":"blocked"},"output":{"image_id":"img-form-blocked-email","selection_id":"sel-blocked","region_ids":["form-err-403"],"count":1}}',
        ),
    ]
    email_step = plan_tool_calls(
        messages=email_messages,
        media=[],
        tool_specs=[SPECS["extract_layout"], SPECS["refine_selection"], SPECS["read_region_text"]],
    )

    assert email_step[0].name == "refine_selection"
    assert email_step[0].arguments == {"selection_id": "sel-blocked", "filter_query": "email"}
