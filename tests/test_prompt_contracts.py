from __future__ import annotations

import pytest

from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.schemas import Message
from gemma4_capability_map.tools.prompt_contracts import (
    get_tool_prompt_contract,
    known_tool_prompt_contract_ids,
    render_tool_prompt_contract,
)
from gemma4_capability_map.tools.registry import build_default_registry


def test_prompt_contract_registry_exposes_current_candidate_ids() -> None:
    ids = known_tool_prompt_contract_ids()

    assert ids == [
        "canonical_json_copy_v3",
        "literal_argument_guard_v1",
        "parallel_array_required_v2",
        "parallel_two_call_array_v3",
        "schema_anchor_v1",
        "schema_literal_tool_required_v2",
        "tool_required_parallel_v1",
        "visual_next_call_state_v2",
        "visual_state_tool_selection_v4",
        "visual_tool_initiation_v3",
    ]
    assert get_tool_prompt_contract("schema_anchor_v1") is not None


def test_prompt_contract_rendering_is_generic_and_does_not_leak_expected_call() -> None:
    specs = build_default_registry().specs
    rendered = render_tool_prompt_contract(
        "schema_anchor_v1",
        messages=[
            Message(
                role="user",
                content="Ignore the earlier publish plan. Search logs/billing.log for the latest invoice-lock failure and report it.",
            )
        ],
        media=[],
        tool_specs=[specs["cli_search_logs"], specs["api_fetch_record"]],
    )

    assert "Tool prompt contract candidate: schema_anchor_v1" in rendered
    assert "Allowed tool names for this turn: cli_search_logs, api_fetch_record." in rendered
    assert "does not reveal the expected tool call" in rendered
    assert '{"name":"cli_search_logs"' not in rendered
    assert "invoice lock" not in rendered


def test_prompt_contract_rendering_rejects_unknown_ids() -> None:
    specs = build_default_registry().specs

    with pytest.raises(ValueError, match="Unknown tool prompt contract"):
        render_tool_prompt_contract(
            "missing_contract",
            messages=[Message(role="user", content="Search logs.")],
            media=[],
            tool_specs=[specs["cli_search_logs"]],
        )


def test_research_controls_carry_prompt_contract_id_in_manifest() -> None:
    controls = ResearchControls.from_mapping(
        {
            "disable_tool_turn_directive": True,
            "tool_prompt_contract_id": "schema_anchor_v1",
        }
    )

    assert controls.disable_tool_turn_directive is True
    assert controls.tool_prompt_contract_id == "schema_anchor_v1"
    assert controls.manifest_payload() == {
        "disable_tool_turn_directive": True,
        "tool_prompt_contract_id": "schema_anchor_v1",
    }


def test_wave_three_contracts_target_live_replay_mechanisms_without_oracle_calls() -> None:
    specs = build_default_registry().specs
    rendered = render_tool_prompt_contract(
        "parallel_two_call_array_v3",
        messages=[Message(role="user", content="Cross-check the screenshot and config/settings.yaml before answering.")],
        media=["img-parallel"],
        tool_specs=[specs["inspect_image"], specs["read_repo_file"]],
    )

    assert "Tool prompt contract candidate: parallel_two_call_array_v3" in rendered
    assert "one tool-call object per source" in rendered
    assert "Allowed tool names for this turn: inspect_image, read_repo_file." in rendered
    assert '{"name":"inspect_image"' not in rendered
    assert "config/settings.yaml" not in rendered


def test_wave_four_visual_state_contract_is_tool_selection_specific_and_generic() -> None:
    specs = build_default_registry().specs
    rendered = render_tool_prompt_contract(
        "visual_state_tool_selection_v4",
        messages=[
            Message(
                role="tool",
                content='{"image_id":"img-dashboard","selection_id":"sel-open-items","region_id":"region-summary"}',
            ),
            Message(role="user", content="Filter to the latest open item and then read the remaining summary."),
        ],
        media=["img-dashboard"],
        tool_specs=[specs["refine_selection"], specs["read_region_text"]],
    )

    assert "Tool prompt contract candidate: visual_state_tool_selection_v4" in rendered
    assert "use an existing selection_id for narrowing" in rendered
    assert "Use an existing region_id for readback" in rendered
    assert "Allowed tool names for this turn: refine_selection, read_region_text." in rendered
    assert '{"name":"refine_selection"' not in rendered
    assert "sel-open-items" not in rendered
