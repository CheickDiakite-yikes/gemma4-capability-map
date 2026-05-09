from __future__ import annotations

from gemma4_capability_map.runtime.tool_directive_probe import _score_probe_case
from gemma4_capability_map.runtime.visual_hard_slice import (
    VISUAL_HARD_SLICE_DESIGNS,
    build_visual_hard_slice_cases,
)
from gemma4_capability_map.schemas import ModelTurn, ToolCall
from gemma4_capability_map.tools.planner import plan_tool_calls
from gemma4_capability_map.tools.registry import build_default_registry


EXPECTED_CALLS = {
    "visual_form_error_vs_message_author": [
        {"name": "extract_layout", "arguments": {"image_id": "img-hard-form-author", "target_query": "validation error"}}
    ],
    "visual_form_error_with_prior_selection_decoy": [
        {"name": "extract_layout", "arguments": {"image_id": "img-hard-form-decoy", "target_query": "validation error"}}
    ],
    "visual_latest_filter_existing_selection": [
        {"name": "refine_selection", "arguments": {"selection_id": "sel-hard-errors", "filter_query": "latest"}}
    ],
    "visual_remaining_filter_existing_selection": [
        {"name": "refine_selection", "arguments": {"selection_id": "sel-hard-remaining", "filter_query": "remaining"}}
    ],
    "visual_region_readback_after_layout_result": [
        {"name": "read_region_text", "arguments": {"image_id": "img-hard-callout-readback", "region_id": "hard-callout-901"}}
    ],
    "visual_metric_panel_vs_table_selector": [
        {"name": "extract_layout", "arguments": {"image_id": "img-hard-metric-table", "target_query": "dashboard metric"}}
    ],
    "visual_callout_warning_with_user_decoy": [
        {"name": "extract_layout", "arguments": {"image_id": "img-hard-callout-decoy", "target_query": "slide callout"}}
    ],
    "visual_selection_id_opaque_copy_with_filter": [
        {"name": "refine_selection", "arguments": {"selection_id": "sel-opaque-77", "filter_query": "blocked"}}
    ],
}


def test_visual_hard_slice_cases_match_design_catalog() -> None:
    cases = build_visual_hard_slice_cases()

    assert [case.case_id for case in cases] == [case.case_id for case in VISUAL_HARD_SLICE_DESIGNS]
    assert len(cases) == 8
    assert {case.case_id for case in cases} == set(EXPECTED_CALLS)


def test_visual_hard_slice_cases_have_expected_planner_calls() -> None:
    specs = build_default_registry().specs

    for case in build_visual_hard_slice_cases():
        tool_specs = [specs[name] for name in case.tool_names]
        planned = plan_tool_calls(case.messages, case.media, tool_specs)

        assert [{"name": call.name, "arguments": call.arguments} for call in planned] == EXPECTED_CALLS[case.case_id]


def test_visual_hard_slice_expected_calls_are_executable() -> None:
    specs = build_default_registry().specs

    for case in build_visual_hard_slice_cases():
        tool_specs = [specs[name] for name in case.tool_names]
        expected_calls = plan_tool_calls(case.messages, case.media, tool_specs)
        turn = ModelTurn(
            raw_model_output="",
            normalized_tool_call=[
                ToolCall(name=call.name, arguments=call.arguments, source_format="heuristic", raw="")
                for call in expected_calls
            ],
        )

        row = _score_probe_case(case, tool_specs, expected_calls, turn)

        assert row["exact_match"] is True
        assert row["executable_match"] is True
        assert row["executor_target_match"] is True
