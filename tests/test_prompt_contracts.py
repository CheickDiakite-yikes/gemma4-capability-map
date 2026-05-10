from __future__ import annotations

import pytest

from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.schemas import Message
from gemma4_capability_map.tools.prompt_contracts import (
    get_tool_prompt_contract,
    known_tool_prompt_contract_ids,
    render_tool_prompt_contract,
)
from gemma4_capability_map.tools.planner import render_tool_catalog_profile, tool_catalog_text
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
        "visual_refine_selection_v5",
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


def test_research_controls_carry_tool_catalog_profile_id_in_manifest() -> None:
    controls = ResearchControls.from_mapping(
        {
            "disable_tool_turn_directive": True,
            "tool_catalog_profile_id": "visual_role_catalog_v1",
        }
    )

    assert controls.disable_tool_turn_directive is True
    assert controls.tool_catalog_profile_id == "visual_role_catalog_v1"
    assert controls.manifest_payload() == {
        "disable_tool_turn_directive": True,
        "tool_catalog_profile_id": "visual_role_catalog_v1",
    }


def test_visual_role_catalog_profile_is_generic_and_tool_catalog_scoped() -> None:
    specs = build_default_registry().specs
    rendered = render_tool_catalog_profile(
        "visual_role_catalog_v1",
        tool_specs=[specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
    )

    assert "Tool catalog profile: visual_role_catalog_v1" in rendered
    assert "does not reveal the expected tool call" in rendered
    assert "extract_layout: start layout or region extraction" in rendered
    assert "refine_selection: filter, narrow, constrain" in rendered
    assert "read_region_text: read text from an existing region_id" in rendered
    assert '{"name":"refine_selection"' not in rendered
    assert "sel-open-items" not in rendered


def test_visual_role_catalog_argument_hints_profile_keeps_selector_guidance_generic() -> None:
    specs = build_default_registry().specs
    rendered = render_tool_catalog_profile(
        "visual_role_catalog_argument_hints_v2",
        tool_specs=[specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
    )

    assert "Tool catalog profile: visual_role_catalog_argument_hints_v2" in rendered
    assert "Visual argument field semantics:" in rendered
    assert "target_query is a compact visual selector label" in rendered
    assert "filter_query is a compact selector token" in rendered
    assert "shortest literal filter token" in rendered
    assert '{"name":"refine_selection"' not in rendered
    assert "sel-open-items" not in rendered


def test_visual_role_catalog_split_selector_profile_separates_region_and_filter_semantics() -> None:
    specs = build_default_registry().specs
    rendered = render_tool_catalog_profile(
        "visual_role_catalog_split_selector_hints_v3",
        tool_specs=[specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
    )

    assert "Tool catalog profile: visual_role_catalog_split_selector_hints_v3" in rendered
    assert "Split selector discipline:" in rendered
    assert "visible region class or UI state" in rendered
    assert "not the upstream task subject" in rendered
    assert "shortest literal narrowing token" in rendered
    assert '{"name":"extract_layout"' not in rendered
    assert "sel-open-items" not in rendered


def test_visual_role_catalog_schema_field_hints_profile_annotates_visual_schema_fields() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_schema_field_hints_v4",
    )

    assert "Tool catalog profile: visual_role_catalog_schema_field_hints_v4" in rendered
    assert "Visual argument field semantics:" not in rendered
    assert "Split selector discipline:" not in rendered
    assert '"target_query": {"type": "string", "description": "Visible region class or UI state to locate' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token' in rendered
    assert '"region_id": {"type": "string", "description": "Opaque region id copied exactly' in rendered
    assert "img-form-live-latest" not in rendered
    assert "sel-001" not in rendered
    assert "recruiter note" not in rendered


def test_visual_role_catalog_schema_literal_targets_profile_preserves_generic_target_labels() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_schema_literal_targets_v5",
    )

    assert "Tool catalog profile: visual_role_catalog_schema_literal_targets_v5" in rendered
    assert "Extract-layout target label discipline:" in rendered
    assert "Preserve the stable surface noun plus region class" in rendered
    assert "drop task/status adjectives" in rendered or "drop status or task adjectives" in rendered
    assert '"target_query": {"type": "string", "description": "Compact visible-region label' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token' in rendered
    assert '"region_id": {"type": "string", "description": "Opaque region id copied exactly' in rendered
    assert "img-hard-callout-decoy" not in rendered
    assert "sel-opaque-77" not in rendered
    assert "Dana" not in rendered


def test_visual_role_catalog_oblique_code_hints_profile_preserves_code_suffixes_and_negated_decoys() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_oblique_code_hints_v6",
    )

    assert "Tool catalog profile: visual_role_catalog_oblique_code_hints_v6" in rendered
    assert "Oblique visible-label discipline:" in rendered
    assert "keep the full visible label in target_query" in rendered
    assert "not X, not the X, or before reading X" in rendered
    assert '"target_query": {"type": "string", "description": "Compact literal visible-region label' in rendered
    assert "cell r42" not in rendered
    assert "alert p55" not in rendered
    assert "consent toggle" not in rendered


def test_visual_role_catalog_oblique_code_guard_profile_adds_stale_selection_guard() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_oblique_code_guard_v7",
    )

    assert "Tool catalog profile: visual_role_catalog_oblique_code_guard_v7" in rendered
    assert "Oblique visible-label discipline:" in rendered
    assert "a letter followed by digits" in rendered
    assert "Stale-selection activation guard:" in rendered
    assert "old, stale, saved, ignored, or previous selection_id" in rendered
    assert "latest passing visual tool result provides the current selection_id" in rendered
    assert "Do not use old, stale, saved, ignored, or previous selection ids" in rendered
    assert '"target_query": {"type": "string", "description": "Compact literal visible-region label' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered
    assert "sel-e19-archive" not in rendered
    assert "field e19" not in rendered


def test_visual_role_catalog_hybrid_label_guard_profile_adds_component_label_guard() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_hybrid_label_guard_v8",
    )

    assert "Tool catalog profile: visual_role_catalog_hybrid_label_guard_v8" in rendered
    assert "Oblique visible-label discipline:" in rendered
    assert "Stale-selection activation guard:" in rendered
    assert "Hybrid label activation guard:" in rendered
    assert "copy that component label in target_query instead of the text value inside it" in rendered
    assert "lowercase or uppercase" in rendered
    assert "pill, tile, toast, chip, badge, field, node, or alert" in rendered
    assert "open, saved, expired, urgent, low, or done" in rendered
    assert '"target_query": {"type": "string", "description": "Compact literal visible-component label' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered
    assert "chip l90" not in rendered
    assert "sel-b12-archive" not in rendered


def test_visual_role_catalog_component_value_guard_profile_adds_role_value_guard() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_component_value_guard_v9",
    )

    assert "Tool catalog profile: visual_role_catalog_component_value_guard_v9" in rendered
    assert "Hybrid label activation guard:" in rendered
    assert "Component value disambiguation guard:" in rendered
    assert "role noun plus component class together as the target_query" in rendered
    assert "state, status, phase, priority, severity, risk, result, and owner" in rendered
    assert "blocked, review, on hold, pending, failed, approved, or overdue" in rendered
    assert '"target_query": {"type": "string", "description": "Compact literal visible-component label' in rendered
    assert "state pill" not in rendered
    assert "residual_state_pill" not in rendered


def test_visual_role_catalog_no_call_control_rescue_profile_adds_generic_routing_guard() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_no_call_control_rescue_v10",
    )

    assert "Tool catalog profile: visual_role_catalog_no_call_control_rescue_v10" in rendered
    assert "Visual argument field semantics:" in rendered
    assert "No-call visual-control activation guard:" in rendered
    assert "return a visual tool call rather than prose" in rendered
    assert "start with extract_layout" in rendered
    assert "business reason, stale id text, or repeated explanatory value" in rendered
    assert '"target_query": {"type": "string", "description": "Compact selector label for the visible UI control' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered
    assert "status badge" not in rendered
    assert "owner field" not in rendered
    assert "state pill" not in rendered
    assert "component_value_status_badge_email_decoy" not in rendered


def test_visual_role_catalog_component_label_guard_profile_adds_narrow_component_copying_guard() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_component_label_guard_v11",
    )

    assert "Tool catalog profile: visual_role_catalog_component_label_guard_v11" in rendered
    assert "Visual argument field semantics:" in rendered
    assert "Stale-selection activation guard:" in rendered
    assert "Hybrid label activation guard:" in rendered
    assert "Narrow component-label guard:" in rendered
    assert "role-plus-component phrase" in rendered
    assert "drop the wrapper words" in rendered
    assert "displayed value inside it" in rendered
    assert '"target_query": {"type": "string", "description": "Compact visible-component label requested by the user' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered
    assert "component_value_status_badge_email_decoy" not in rendered


def test_visual_role_catalog_component_residual_guard_profile_targets_h1q_misses() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_component_residual_guard_v12",
    )

    assert "Tool catalog profile: visual_role_catalog_component_residual_guard_v12" in rendered
    assert "Oblique visible-label discipline:" in rendered
    assert "Stale-selection activation guard:" in rendered
    assert "Narrow component-label guard:" in rendered
    assert "Residual component-label guard:" in rendered
    assert "tag, toggle, switch, field, badge, chip, pill, tile, alert, and node" in rendered
    assert "owner field, assignee field, reviewer field, or mode field" in rendered
    assert "alert s92 or badge c08" in rendered
    assert '"target_query": {"type": "string", "description": "Compact residual visible-component label requested by the user' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered
    assert "component_value_status_badge_email_decoy" not in rendered


def test_visual_role_catalog_conditional_residual_route_profile_keeps_v11_default() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_conditional_residual_route_v13",
    )

    assert "Tool catalog profile: visual_role_catalog_conditional_residual_route_v13" in rendered
    assert "Narrow component-label guard:" in rendered
    assert "Conditional residual route guard:" in rendered
    assert "Default to the narrow component-label guard" in rendered
    assert "code suffix" in rendered
    assert "tag, toggle, or switch" in rendered
    assert "Do not add residual handling for ordinary pill, badge, chip, or tile targets" in rendered
    assert '"target_query": {"type": "string", "description": "Compact visible-component label requested by the user. Default to role-plus-component labels' in rendered
    assert "component_value_status_badge_email_decoy" not in rendered


def test_visual_role_catalog_nonstandard_component_class_guard_targets_tag_toggle_values() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_nonstandard_component_class_guard_v14",
    )

    assert "Tool catalog profile: visual_role_catalog_nonstandard_component_class_guard_v14" in rendered
    assert "Nonstandard component-class guard:" in rendered
    assert "tag, toggle, and switch" in rendered
    assert "state tag, mode toggle, consent switch, or status tag" in rendered
    assert "Closed, Manual, Enabled, Disabled, or Approved" in rendered
    assert '"target_query": {"type": "string", "description": "Compact visible-component label requested by the user. For tag, toggle, or switch targets' in rendered


def test_visual_role_catalog_code_label_exact_guard_targets_negated_decoys() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_code_label_exact_guard_v15",
    )

    assert "Tool catalog profile: visual_role_catalog_code_label_exact_guard_v15" in rendered
    assert "Oblique visible-label discipline:" in rendered
    assert "Code-label exact guard:" in rendered
    assert "alert s92 or badge c08" in rendered
    assert "nearby toggle, note, log, or table is a decoy" in rendered
    assert '"target_query": {"type": "string", "description": "Compact literal code label requested by the user' in rendered


def test_visual_role_catalog_routed_residual_guard_keeps_route_boundaries() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_routed_residual_guard_v16",
    )

    assert "Tool catalog profile: visual_role_catalog_routed_residual_guard_v16" in rendered
    assert "Narrow component-label guard:" in rendered
    assert "Routed residual component guard:" in rendered
    assert "Default to the narrow component-label guard" in rendered
    assert "explicit route evidence" in rendered
    assert "field target with stale, old, previous, saved, ignored, memo, table, or summary decoys" in rendered
    assert "tag/toggle/switch target" in rendered
    assert "component noun plus code suffix" in rendered
    assert "do not broaden ordinary pill, badge, chip, or tile targets" in rendered
    assert '"target_query": {"type": "string", "description": "Compact routed visible-component label requested by the user' in rendered


def test_visual_role_catalog_selection_origin_guard_targets_h1y_failures() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_selection_origin_guard_v17",
    )

    assert "Tool catalog profile: visual_role_catalog_selection_origin_guard_v17" in rendered
    assert "Selection-origin and component-phrase precedence guard:" in rendered
    assert "selection_id strings in the user's message as untrusted text" in rendered
    assert "do not call refine_selection; start with extract_layout" in rendered
    assert "If the prompt says label is X" in rendered
    assert "locate X exactly" in rendered
    assert "component itself" in rendered
    assert "lifecycle state tag becomes state tag" in rendered
    assert '"target_query": {"type": "string", "description": "Compact visible-component label from the prompt' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a selection_id copied from an earlier passing visual tool result only' in rendered


def test_visual_role_catalog_scoped_residual_exactness_preserves_h2b_route_boundaries() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_scoped_residual_exactness_v18",
    )

    assert "Tool catalog profile: visual_role_catalog_scoped_residual_exactness_v18" in rendered
    assert "Narrow component-label guard:" in rendered
    assert "Scoped residual exactness route:" in rendered
    assert "Default to the narrow component-label guard" in rendered
    assert "result pill stays result pill" in rendered
    assert "alert s92 stays alert s92" in rendered
    assert "badge c08 stays badge c08" in rendered
    assert "state tag stays state tag" in rendered
    assert "mode toggle stays mode toggle" in rendered
    assert "Leave stale selection_id repair to the controller gate" in rendered
    assert '"target_query": {"type": "string", "description": "Compact scoped residual label from the prompt' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered


def test_visual_role_catalog_class_preserving_residual_route_blocks_h2c_class_swap() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_class_preserving_residual_route_v19",
    )

    assert "Tool catalog profile: visual_role_catalog_class_preserving_residual_route_v19" in rendered
    assert "Class-preserving residual route:" in rendered
    assert "preserve both words the user named" in rendered
    assert "result chip, target_query is result chip" in rendered
    assert "result pill, target_query is result pill" in rendered
    assert "result badge, target_query is result badge" in rendered
    assert "never substitute a different component class" in rendered
    assert '"target_query": {"type": "string", "description": "Compact class-preserving residual label from the prompt' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered


def test_visual_role_catalog_route_arbitration_keeps_code_exactness_and_class_transfer() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_route_arbitration_residual_exactness_v20",
    )

    assert "Tool catalog profile: visual_role_catalog_route_arbitration_residual_exactness_v20" in rendered
    assert "Route-arbitrated residual exactness:" in rendered
    assert "badge c08 stays badge c08" in rendered
    assert "alert s92 stays alert s92" in rendered
    assert "escalated badge c08 must become badge c08" in rendered
    assert "result chip stays result chip" in rendered
    assert "result pill stays result pill" in rendered
    assert "Never import a component class from a previous example or fit row" in rendered
    assert "warning panel, or error notice" in rendered
    assert '"target_query": {"type": "string", "description": "Compact route-arbitrated residual label from the prompt' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered


def test_visual_role_catalog_component_identity_query_contract_preserves_h2f_labels() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_component_identity_query_contract_v21",
    )

    assert "Tool catalog profile: visual_role_catalog_component_identity_query_contract_v21" in rendered
    assert "Component-identity query contract:" in rendered
    assert "result tile, resolution badge, state marker, or mode switch" in rendered
    assert "state marker is not lifecycle state marker" in rendered
    assert "mode switch is not mode toggle" in rendered
    assert "Blocked, Deferred, Closed, and Manual are displayed values" in rendered
    assert "badge m31, alert t47" in rendered
    assert '"target_query": {"type": "string", "description": "Compact requested component identity from the prompt' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered


def test_visual_role_catalog_component_identity_negative_examples_targets_h2f_residuals() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_component_identity_negative_examples_v22",
    )

    assert "Tool catalog profile: visual_role_catalog_component_identity_negative_examples_v22" in rendered
    assert "Component-identity negative examples:" in rendered
    assert "result tile for Blocked, target_query is result tile; never use Blocked" in rendered
    assert "resolution badge for Deferred, target_query is resolution badge" in rendered
    assert "never use Deferred or resolution badge Deferred" in rendered
    assert "state marker; never expand it to lifecycle state marker" in rendered
    assert "mode switch; never rewrite it as mode toggle" in rendered
    assert "displayed values, status words, copied table values, and nearby alias phrases as decoys" in rendered
    assert "badge m31, alert t47" in rendered
    assert '"target_query": {"type": "string", "description": "Compact requested component identity from the prompt' in rendered
    assert "value-appended phrases like resolution badge Deferred" in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered


def test_visual_role_catalog_conditional_component_identity_arbitration_guards_h2h_regressions() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_conditional_component_identity_arbitration_v23",
    )

    assert "Tool catalog profile: visual_role_catalog_conditional_component_identity_arbitration_v23" in rendered
    assert "Conditional component-identity arbitration:" in rendered
    assert "Default to route-arbitrated residual exactness" in rendered
    assert "result pill, result chip, badge c08, error banner" in rendered
    assert "result tile for Blocked stays result tile" in rendered
    assert "resolution badge for Deferred stays resolution badge" in rendered
    assert "state marker; do not expand it to lifecycle state marker" in rendered
    assert "mode switch; do not rewrite it as mode toggle" in rendered
    assert "result tile does not change result pill or result chip" in rendered
    assert "error notice does not change error banner" in rendered
    assert "never combine them into badge m31 c08" in rendered
    assert '"target_query": {"type": "string", "description": "Compact route-arbitrated component label from the prompt' in rendered
    assert '"filter_query": {"type": "string", "description": "Shortest literal narrowing token for a current selection_id' in rendered


def test_tool_catalog_profile_renders_inside_catalog_without_exact_directive() -> None:
    specs = build_default_registry().specs
    rendered = tool_catalog_text(
        [specs["extract_layout"], specs["refine_selection"], specs["read_region_text"]],
        profile_id="visual_role_catalog_v1",
    )

    assert "Tool catalog profile: visual_role_catalog_v1" in rendered
    assert "Allowed tools. Use only these exact names." in rendered
    assert "Tool directive for this turn:" not in rendered
    assert '"name": "refine_selection"' in rendered


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


def test_wave_five_visual_refine_contract_targets_latest_selection_filtering_only() -> None:
    specs = build_default_registry().specs
    rendered = render_tool_prompt_contract(
        "visual_refine_selection_v5",
        messages=[
            Message(
                role="tool",
                content='{"image_id":"img-dashboard","selection_id":"sel-open-items","region_id":"region-summary"}',
            ),
            Message(role="user", content="Filter to the latest open item and then read the remaining summary."),
        ],
        media=["img-dashboard"],
        tool_specs=[specs["refine_selection"], specs["read_region_text"], specs["extract_layout"]],
    )

    assert "Tool prompt contract candidate: visual_refine_selection_v5" in rendered
    assert "choose refine_selection when it is an allowed tool" in rendered
    assert "Do not use read_region_text for filtering" in rendered
    assert "Do not use inspect_image or extract_layout again" in rendered
    assert "Allowed tool names for this turn: refine_selection, read_region_text, extract_layout." in rendered
    assert '{"name":"refine_selection"' not in rendered
    assert "sel-open-items" not in rendered
