from __future__ import annotations

import json
from pathlib import Path

from gemma4_capability_map.runtime.tool_directive_probe import (
    ToolDirectiveProbeCase,
    _apply_visual_composed_route_gating,
    _apply_visual_contextual_surface_alias_routing,
    _apply_visual_negated_component_target_preservation,
    _apply_visual_semantic_target_preservation,
    _apply_visual_stale_selection_gate,
    _apply_visual_target_query_normalization,
    _apply_visual_value_bearing_target_query_synthesis,
    _score_probe_case,
    build_tool_directive_probe_cases,
    compare_tool_directive_probe_packets,
    run_tool_directive_probe,
    write_tool_directive_probe_comparison,
)
from gemma4_capability_map.schemas import Message, ModelTurn, ToolCall
from gemma4_capability_map.tools.planner import plan_tool_calls
from gemma4_capability_map.tools.registry import build_default_registry


def test_tool_directive_probe_cases_cover_cli_api_visual_and_parallel_families() -> None:
    cases = build_tool_directive_probe_cases()
    families = {case.family for case in cases}

    assert "cli_canonicalization" in families
    assert "api_canonicalization" in families
    assert "visual_argument_copying" in families
    assert "parallel_tool_calling" in families

    specs = build_default_registry().specs
    form_case = next(case for case in cases if case.case_id == "visual_form_target_literal")
    planned = plan_tool_calls(
        form_case.messages,
        form_case.media,
        [specs[name] for name in form_case.tool_names],
    )
    assert planned[0].name == "extract_layout"
    assert planned[0].arguments == {"image_id": "img-form-live-latest", "target_query": "validation error"}


def test_run_tool_directive_probe_with_heuristic_system_writes_outputs(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_probe:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_probe",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:2],
    )

    assert result["summary"]["case_count"] == 2
    assert result["summary"]["exact_match_rate"] == 1.0
    assert result["summary"]["executor_equivalence_match_rate"] is None
    assert (output_dir / "manifest.json").exists()
    rows = json.loads((output_dir / "probe_results.json").read_text(encoding="utf-8"))
    assert rows[0]["exact_match"] is True
    assert rows[0]["executor_target_match"] is None
    assert (output_dir / "probe_results.csv").exists()


def test_run_tool_directive_probe_records_prompt_contract_controls(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_prompt_contract:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      tool_prompt_contract_id: schema_anchor_v1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_prompt_contract",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "tool_prompt_contract_id": "schema_anchor_v1",
    }
    assert result["manifest"]["runtime_info"]["tool_turn_directive_enabled"] is False
    assert result["manifest"]["runtime_info"]["tool_prompt_contract_id"] == "schema_anchor_v1"


def test_run_tool_directive_probe_records_tool_catalog_profile_controls(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_tool_catalog_profile:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      tool_catalog_profile_id: visual_role_catalog_v1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_tool_catalog_profile",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[4:5],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "tool_catalog_profile_id": "visual_role_catalog_v1",
    }
    assert result["manifest"]["runtime_info"]["tool_turn_directive_enabled"] is False
    assert result["manifest"]["runtime_info"]["tool_catalog_profile_id"] == "visual_role_catalog_v1"


def test_run_tool_directive_probe_records_visual_stale_selection_gate_control(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_stale_selection_gate:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      enable_visual_stale_selection_gate: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_stale_selection_gate",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "enable_visual_stale_selection_gate": True,
    }


def test_run_tool_directive_probe_records_visual_target_query_normalization_control(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_target_query_normalization:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      enable_visual_target_query_normalization: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_target_query_normalization",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "enable_visual_target_query_normalization": True,
    }


def test_run_tool_directive_probe_records_visual_scoped_target_query_normalization_control(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_scoped_target_query_normalization:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      enable_visual_scoped_target_query_normalization: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_scoped_target_query_normalization",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "enable_visual_scoped_target_query_normalization": True,
    }


def test_run_tool_directive_probe_records_visual_value_bearing_synthesis_control(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_value_bearing_target_query_synthesis:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      enable_visual_value_bearing_target_query_synthesis: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_value_bearing_target_query_synthesis",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "enable_visual_value_bearing_target_query_synthesis": True,
    }


def test_run_tool_directive_probe_records_visual_contextual_surface_alias_routing_control(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_contextual_surface_alias_routing:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      enable_visual_contextual_surface_alias_routing: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_contextual_surface_alias_routing",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "enable_visual_contextual_surface_alias_routing": True,
    }


def test_run_tool_directive_probe_records_visual_composed_route_gating_control(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_composed_route_gating:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      enable_visual_composed_route_gating: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_composed_route_gating",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "enable_visual_composed_route_gating": True,
    }


def test_run_tool_directive_probe_records_visual_semantic_target_preservation_control(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_semantic_target_preservation:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      enable_visual_semantic_target_preservation: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_semantic_target_preservation",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "enable_visual_semantic_target_preservation": True,
    }


def test_visual_stale_selection_gate_rewrites_missing_selection_to_layout_lookup() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="stale-selection",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-stale"),
            Message(
                role="user",
                content="Old selection_id sel-owner-memo points at the memo. Locate the owner field component.",
            ),
        ],
        media=["img-stale"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-stale": {
                    "local_layouts": [
                        {"region_id": "memo-1", "label": "owner memo", "text": "Iris"},
                        {"region_id": "field-1", "label": "owner field", "text": "Iris"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-owner-memo", "filter_query": "owner field"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_stale_selection_gate(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call == [
        ToolCall(
            name="extract_layout",
            arguments={"image_id": "img-stale", "target_query": "owner field"},
            source_format="heuristic",
            raw=patched.normalized_tool_call[0].raw,
        )
    ]
    assert patched.runtime_metadata["visual_stale_selection_gate"][0]["from_tool"] == "refine_selection"


def test_visual_stale_selection_gate_preserves_current_selection_ids() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="current-selection",
        family="visual",
        messages=[Message(role="user", content="Narrow the current selection to latest.")],
        media=["img-current"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_selections": {
                "sel-current": {
                    "image_id": "img-current",
                    "selection_kind": "regions",
                    "items": [],
                }
            }
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-current", "filter_query": "latest"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_stale_selection_gate(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched == turn


def test_visual_target_query_normalization_rewrites_value_to_prompt_state_label() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="target-query-normalization",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-result"),
            Message(
                role="user",
                content="The comment says Blocked too. Select the visible result tile for Blocked, not the comment.",
            ),
        ],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "comment-1", "label": "result comment", "text": "Blocked by legal"},
                        {"region_id": "tile-1", "label": "result tile", "text": "Blocked"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "Blocked"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_target_query_normalization(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call == [
        ToolCall(
            name="extract_layout",
            arguments={"image_id": "img-result", "target_query": "result tile"},
            source_format="heuristic",
            raw=patched.normalized_tool_call[0].raw,
        )
    ]
    assert patched.runtime_metadata["visual_target_query_normalization"][0]["from_tool"] == "extract_layout"
    assert patched.runtime_metadata["visual_target_query_normalization"][0]["prompt_state_label"] == "result tile"


def test_visual_target_query_normalization_preserves_when_prompt_has_no_state_label() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="target-query-normalization-no-match",
        family="visual",
        messages=[Message(role="user", content="Locate the visible blocked result component.")],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "tile-1", "label": "result tile", "text": "Blocked"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "blocked result"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_target_query_normalization(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched == turn


def test_visual_target_query_normalization_preserves_located_code_label_over_negated_decoy() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="target-query-normalization-negated-code-label",
        family="visual",
        messages=[
            Message(
                role="user",
                content="Before reading the consent toggle, locate alert s92. Do not target the toggle.",
            )
        ],
        media=["img-alert"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-alert": {
                    "local_layouts": [
                        {"region_id": "toggle-1", "label": "consent toggle", "text": "Enabled"},
                        {"region_id": "alert-1", "label": "alert s92", "text": "Escalated"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-alert", "target_query": "alert s92"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_target_query_normalization(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched == turn


def test_scoped_visual_target_query_normalization_blocks_value_bearing_overstrip() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="target-query-normalization-value-bearing-block",
        family="visual",
        messages=[
            Message(
                role="user",
                content=(
                    "From the status summary, pull the Blocked result badge chip. "
                    "The plain result badge is just a legend."
                ),
            )
        ],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "legend-1", "label": "result badge", "text": "Legend"},
                        {"region_id": "badge-1", "label": "result badge Blocked", "text": "Blocked"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "result chip"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_target_query_normalization(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_value_bearing_targets=True,
    )

    assert patched.normalized_tool_call == turn.normalized_tool_call
    blocked = patched.runtime_metadata["visual_target_query_normalization_blocked"][0]
    assert blocked["prompt_state_label"] == "result badge"
    assert blocked["preserved_target_query"] == "result chip"
    assert blocked["value_bearing_label"] == "result badge Blocked"
    assert blocked["reason"] == "value_bearing_label_requested"
    assert "visual_target_query_normalization" not in patched.runtime_metadata


def test_scoped_visual_target_query_normalization_preserves_contextual_label_repair() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="target-query-normalization-contextual-repair",
        family="visual",
        messages=[
            Message(
                role="user",
                content="For the archive panel, work from the error notice rather than the live banner or log.",
            )
        ],
        media=["img-error"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-error": {
                    "local_layouts": [
                        {"region_id": "banner-1", "label": "error banner", "text": "Error"},
                        {"region_id": "notice-1", "label": "error notice", "text": "Error archived"},
                        {"region_id": "log-1", "label": "error log", "text": "Error trace rows"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-error", "target_query": "archive panel"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_target_query_normalization(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_value_bearing_targets=True,
    )

    assert patched.normalized_tool_call == [
        ToolCall(
            name="extract_layout",
            arguments={"image_id": "img-error", "target_query": "error notice"},
            source_format="heuristic",
            raw=patched.normalized_tool_call[0].raw,
        )
    ]
    assert patched.runtime_metadata["visual_target_query_normalization"][0]["prompt_state_label"] == "error notice"
    assert "visual_target_query_normalization_blocked" not in patched.runtime_metadata


def test_scoped_visual_target_query_normalization_blocks_direct_longer_label_overstrip() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="target-query-normalization-direct-longer-label-block",
        family="visual",
        messages=[
            Message(
                role="user",
                content="The target is result badge Blocked, the full value-bearing badge.",
            )
        ],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "legend-1", "label": "result badge", "text": "Summary"},
                        {"region_id": "badge-1", "label": "result badge Blocked", "text": "Blocked"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "result badge"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_target_query_normalization(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_value_bearing_targets=True,
    )

    assert patched.normalized_tool_call == turn.normalized_tool_call
    assert patched.runtime_metadata == {}


def test_value_bearing_target_query_synthesis_canonicalizes_recoverable_longer_label() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="value-bearing-target-query-synthesis",
        family="visual",
        messages=[
            Message(
                role="user",
                content=(
                    "From the status summary, pull the Blocked result badge chip. "
                    "The plain result badge is just a legend."
                ),
            )
        ],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "legend-1", "label": "result badge", "text": "Legend"},
                        {"region_id": "badge-1", "label": "result badge Blocked", "text": "Blocked"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "result chip"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_value_bearing_target_query_synthesis(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call == [
        ToolCall(
            name="extract_layout",
            arguments={"image_id": "img-result", "target_query": "result badge Blocked"},
            source_format="heuristic",
            raw=patched.normalized_tool_call[0].raw,
        )
    ]
    metadata = patched.runtime_metadata["visual_value_bearing_target_query_synthesis"][0]
    assert metadata["prompt_state_label"] == "result badge"
    assert metadata["value_bearing_label"] == "result badge Blocked"
    assert metadata["matched_phrase"] == "blocked result badge"


def test_value_bearing_target_query_synthesis_canonicalizes_order_and_case() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="value-bearing-target-query-synthesis-order-case",
        family="visual",
        messages=[
            Message(
                role="user",
                content="Use the Critical priority badge in the risk strip.",
            )
        ],
        media=["img-priority"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-priority": {
                    "local_layouts": [
                        {"region_id": "normal-1", "label": "priority badge", "text": "Normal"},
                        {"region_id": "critical-1", "label": "priority badge Critical", "text": "Critical"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-priority", "target_query": "priority badge critical"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_value_bearing_target_query_synthesis(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call[0].arguments["target_query"] == "priority badge Critical"
    metadata = patched.runtime_metadata["visual_value_bearing_target_query_synthesis"][0]
    assert metadata["matched_phrase"] == "critical priority badge"
    assert metadata["reason"] == "value_bearing_label_recoverable"


def test_value_bearing_target_query_synthesis_preserves_contextual_label_repair() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="value-bearing-target-query-synthesis-contextual-repair",
        family="visual",
        messages=[
            Message(
                role="user",
                content="For the archive panel, work from the error notice rather than the live banner or log.",
            )
        ],
        media=["img-error"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-error": {
                    "local_layouts": [
                        {"region_id": "banner-1", "label": "error banner", "text": "Error"},
                        {"region_id": "notice-1", "label": "error notice", "text": "Error archived"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-error", "target_query": "archive panel"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_value_bearing_target_query_synthesis(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call[0].arguments["target_query"] == "error notice"
    assert "visual_value_bearing_target_query_synthesis" not in patched.runtime_metadata
    assert patched.runtime_metadata["visual_target_query_normalization"][0]["prompt_state_label"] == "error notice"


def test_negation_aware_target_query_normalization_blocks_context_label_overwrite() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="target-query-normalization-negation-scope-block",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-h2t-metric"),
            Message(
                role="user",
                content=(
                    "Use the metric panel at the top. The annotation saying 'not the metric panel' "
                    "is a training note about a prior screenshot, not the current target."
                ),
            ),
        ],
        media=["img-h2t-metric"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-h2t-metric": {
                    "local_layouts": [
                        {"region_id": "metric-1", "label": "metric panel", "text": "Escalations above target"},
                        {"region_id": "note-1", "label": "training note", "text": "old negative example"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-h2t-metric", "target_query": "metric panel"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_target_query_normalization(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_negated_exact_layout_targets=True,
    )

    assert patched.normalized_tool_call == turn.normalized_tool_call
    blocked = patched.runtime_metadata["visual_target_query_normalization_blocked"][0]
    assert blocked["prompt_state_label"] == "training note"
    assert blocked["preserved_target_query"] == "metric panel"
    assert blocked["reason"] == "negation_scope_exact_layout_label"
    assert "visual_target_query_normalization" not in patched.runtime_metadata


def test_negation_aware_value_bearing_synthesis_blocks_fallback_caption_overwrite() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="value-bearing-target-query-synthesis-negation-scope-block",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-h2t-summary"),
            Message(
                role="user",
                content=(
                    "Use the summary tile in the current image. The caption includes the phrase "
                    "not the summary tile, but it is describing an old example."
                ),
            ),
        ],
        media=["img-h2t-summary"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-h2t-summary": {
                    "local_layouts": [
                        {"region_id": "tile-1", "label": "summary tile", "text": "Ready for review"},
                        {"region_id": "caption-1", "label": "caption", "text": "old negative example"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-h2t-summary", "target_query": "summary tile"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_value_bearing_target_query_synthesis(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_negated_exact_layout_targets=True,
    )

    assert patched.normalized_tool_call == turn.normalized_tool_call
    blocked = patched.runtime_metadata["visual_target_query_normalization_blocked"][0]
    assert blocked["prompt_state_label"] == "caption"
    assert blocked["preserved_target_query"] == "summary tile"
    assert blocked["reason"] == "negation_scope_exact_layout_label"
    assert "visual_value_bearing_target_query_synthesis" not in patched.runtime_metadata
    assert "visual_target_query_normalization" not in patched.runtime_metadata


def test_value_bearing_target_query_synthesis_ignores_negated_longer_label() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="value-bearing-target-query-synthesis-negated-label",
        family="visual",
        messages=[
            Message(
                role="user",
                content="Do not use the Closed state tag. Locate the plain state tag in the draft lane.",
            )
        ],
        media=["img-state"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-state": {
                    "local_layouts": [
                        {"region_id": "draft-1", "label": "state tag", "text": "Draft"},
                        {"region_id": "closed-1", "label": "state tag Closed", "text": "Closed"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-state", "target_query": "state marker"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_value_bearing_target_query_synthesis(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call[0].arguments["target_query"] == "state tag"
    assert "visual_value_bearing_target_query_synthesis" not in patched.runtime_metadata
    assert patched.runtime_metadata["visual_target_query_normalization"][0]["prompt_state_label"] == "state tag"


def test_semantic_target_preservation_ignores_stale_example_negation() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="semantic-stale-example-negation",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-review"),
            Message(
                role="user",
                content=(
                    "Use the current review tile. The stale caption says not the review tile, "
                    "but that caption belongs to an old screenshot."
                ),
            ),
        ],
        media=["img-review"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-review": {
                    "local_layouts": [
                        {"region_id": "card-1", "label": "review card", "text": "Review queue"},
                        {"region_id": "tile-1", "label": "review tile", "text": "Review queue"},
                        {"region_id": "caption-1", "label": "stale caption", "text": "old screenshot caption"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-review", "target_query": "review tile"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_value_bearing_target_query_synthesis(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_semantic_targets=True,
    )

    assert patched.normalized_tool_call == turn.normalized_tool_call
    metadata = patched.runtime_metadata["visual_semantic_target_preservation"][0]
    assert metadata["preserved_target_query"] == "review tile"
    assert metadata["blocked_label"] == "stale caption"
    assert metadata["reason"] == "semantic_label_preserved_over_stale_context"


def test_semantic_target_preservation_routes_invalid_selection_to_current_target() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="semantic-stale-selection",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-risk"),
            Message(
                role="user",
                content="Use the risk lane for High. The example note says not the risk lane, but it is marked as a stale example.",
            ),
        ],
        media=["img-risk"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-risk": {
                    "local_layouts": [
                        {"region_id": "chip-1", "label": "risk chip High", "text": "High"},
                        {"region_id": "lane-1", "label": "risk lane", "text": "High"},
                        {"region_id": "note-1", "label": "example note", "text": "stale example note"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "img-risk", "filter_query": "High"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_stale_selection_gate(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_semantic_targets=True,
    )

    assert patched.normalized_tool_call[0].arguments == {"image_id": "img-risk", "target_query": "risk lane"}
    assert patched.runtime_metadata["visual_stale_selection_gate"][0]["to_arguments"]["target_query"] == "risk lane"


def test_stale_selection_negation_guard_rewrites_current_but_rejected_selection() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="stale-selection-negated-current",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-escalation"),
            Message(
                role="user",
                content=(
                    "The stale selection sel-old-note says not the escalation lane. "
                    "Ignore that old selection and use the current escalation lane for P1."
                ),
            ),
        ],
        media=["img-escalation"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-escalation": {
                    "local_layouts": [
                        {"region_id": "chip-1", "label": "escalation chip p1", "text": "P1"},
                        {"region_id": "lane-1", "label": "escalation lane", "text": "P1"},
                        {"region_id": "note-1", "label": "old note", "text": "not the escalation lane"},
                    ]
                }
            },
            "visual_selections": {
                "sel-old-note": {
                    "image_id": "img-escalation",
                    "selection_kind": "regions",
                    "items": [{"region_id": "note-1", "label": "old note", "text": "not the escalation lane"}],
                    "query": "not the escalation lane",
                }
            },
            "visual_last_selection_id": "sel-old-note",
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-old-note", "filter_query": "P1"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_stale_selection_gate(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_semantic_targets=True,
        reject_negated_current_selection=True,
    )

    assert patched.normalized_tool_call[0].name == "extract_layout"
    assert patched.normalized_tool_call[0].arguments == {
        "image_id": "img-escalation",
        "target_query": "escalation lane",
    }
    metadata = patched.runtime_metadata["visual_stale_selection_negation_guard"][0]
    assert metadata["replaced_selection_id"] == "sel-old-note"
    assert metadata["reason"] == "negated_current_selection_to_requested_surface"


def test_stale_selection_paraphrase_guard_rewrites_retired_selection_language() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="stale-selection-retired-paraphrase",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-renewal"),
            Message(
                role="user",
                content=(
                    "Selection handle sel-renewal-note belongs to a retired renewal view; "
                    "current target is the renewal lane."
                ),
            ),
        ],
        media=["img-renewal"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-renewal": {
                    "local_layouts": [
                        {"region_id": "lane-1", "label": "renewal lane", "text": "Renewal due"},
                        {"region_id": "note-1", "label": "renewal note", "text": "Old renewal note"},
                    ]
                }
            },
            "visual_selections": {
                "sel-renewal-note": {
                    "image_id": "img-renewal",
                    "selection_kind": "regions",
                    "items": [{"region_id": "note-1", "label": "renewal note", "text": "Old renewal note"}],
                    "query": "renewal note",
                }
            },
            "visual_last_selection_id": "sel-renewal-note",
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-renewal-note", "filter_query": "latest"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_stale_selection_gate(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_semantic_targets=True,
        reject_paraphrased_current_selection=True,
    )

    assert patched.normalized_tool_call[0].name == "extract_layout"
    assert patched.normalized_tool_call[0].arguments == {
        "image_id": "img-renewal",
        "target_query": "renewal lane",
    }
    metadata = patched.runtime_metadata["visual_stale_selection_paraphrase_guard"][0]
    assert metadata["replaced_selection_id"] == "sel-renewal-note"
    assert metadata["reason"] == "paraphrased_stale_selection_to_requested_surface"


def test_semantic_target_preservation_canonicalizes_inverted_negated_value() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="semantic-inverted-negated-value",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-status"),
            Message(
                role="user",
                content=(
                    "Use the Not ready status badge. Here Not ready is the displayed current value, "
                    "not an instruction to avoid readiness badges."
                ),
            ),
        ],
        media=["img-status"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-status": {
                    "local_layouts": [
                        {"region_id": "ready-1", "label": "status badge Ready", "text": "Ready"},
                        {"region_id": "not-ready-1", "label": "status badge Not ready", "text": "Not ready"},
                        {"region_id": "note-1", "label": "readiness note", "text": "Not ready until QA signs off"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-status", "target_query": "Not ready"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_value_bearing_target_query_synthesis(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_semantic_targets=True,
    )

    assert patched.normalized_tool_call[0].arguments["target_query"] == "status badge Not ready"
    assert patched.runtime_metadata["visual_target_query_normalization"][0]["prompt_state_label"] == (
        "status badge Not ready"
    )


def test_negated_component_target_preservation_expands_short_component_query() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="negated-component-short-query",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-alert"),
            Message(
                role="user",
                content="Use the not active alert banner. Not active is the displayed banner value.",
            ),
        ],
        media=["img-alert"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-alert": {
                    "local_layouts": [
                        {"region_id": "alert-1", "label": "alert banner active", "text": "Active"},
                        {"region_id": "alert-2", "label": "alert banner not active", "text": "Not active"},
                        {"region_id": "note-1", "label": "alert note", "text": "Not active after policy review"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-alert", "target_query": "alert"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_negated_component_target_preservation(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call[0].arguments["target_query"] == "alert banner not active"
    metadata = patched.runtime_metadata["visual_negated_component_target_preservation"][0]
    assert metadata["blocked_label"] == "alert"
    assert metadata["preserved_target_query"] == "alert banner not active"


def test_negative_value_component_target_preservation_expands_status_value_query() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="negative-value-component-query",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-alert"),
            Message(role="user", content="Use inactive alert banner. Inactive is the displayed status."),
        ],
        media=["img-alert"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-alert": {
                    "local_layouts": [
                        {"region_id": "alert-1", "label": "alert banner active", "text": "Active"},
                        {"region_id": "alert-2", "label": "alert banner inactive", "text": "Inactive"},
                        {"region_id": "note-1", "label": "alert note", "text": "Inactive after policy review"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-alert", "target_query": "alert inactive"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_negated_component_target_preservation(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_negative_value_targets=True,
    )

    assert patched.normalized_tool_call[0].arguments["target_query"] == "alert banner inactive"
    metadata = patched.runtime_metadata["visual_negative_value_component_target_preservation"][0]
    assert metadata["blocked_label"] == "alert inactive"
    assert metadata["preserved_target_query"] == "alert banner inactive"


def test_semantic_target_preservation_adds_no_call_visual_fallback() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="semantic-no-call-fallback",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-summary"),
            Message(
                role="user",
                content=(
                    "Use the summary tile. The caption quotes not the summary tile from a stale example; "
                    "the current target is still the tile."
                ),
            ),
        ],
        media=["img-summary"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-summary": {
                    "local_layouts": [
                        {"region_id": "tile-1", "label": "summary tile", "text": "Ready for review"},
                        {"region_id": "caption-1", "label": "caption", "text": "quoted stale example"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(raw_model_output="I can answer directly.", normalized_tool_call=[])

    patched = _apply_visual_semantic_target_preservation(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call[0].arguments == {"image_id": "img-summary", "target_query": "summary tile"}
    metadata = patched.runtime_metadata["visual_semantic_target_preservation"][0]
    assert metadata["reason"] == "no_call_clear_visual_target"
    assert metadata["preserved_target_query"] == "summary tile"


def test_contextual_surface_alias_routing_rewrites_display_value_to_requested_surface_alias() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="contextual-surface-alias-routing",
        family="visual",
        messages=[
            Message(
                role="user",
                content="Use the tile-style result surface for Blocked; the badge and comment are nearby context.",
            )
        ],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "badge-1", "label": "result badge", "text": "Blocked"},
                        {"region_id": "tile-1", "label": "result tile", "text": "Blocked"},
                        {"region_id": "comment-1", "label": "result comment", "text": "Blocked pending counsel"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "Blocked"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_contextual_surface_alias_routing(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call[0].arguments["target_query"] == "result tile"
    metadata = patched.runtime_metadata["visual_contextual_surface_alias_routing"][0]
    assert metadata["display_value"] == "Blocked"
    assert metadata["surface_label"] == "result tile"
    assert metadata["surface_region_id"] == "tile-1"
    assert metadata["reason"] == "contextual_surface_alias_recoverable"


def test_contextual_surface_alias_routing_preserves_without_surface_request() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="contextual-surface-alias-routing-no-request",
        family="visual",
        messages=[Message(role="user", content="Use the Blocked result; the badge and comment are nearby context.")],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "badge-1", "label": "result badge", "text": "Blocked"},
                        {"region_id": "tile-1", "label": "result tile", "text": "Blocked"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "Blocked"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_contextual_surface_alias_routing(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched == turn


def test_visual_composed_route_gating_prioritizes_requested_surface_over_deprioritized_context() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="composed-surface-decoy",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-composed"),
            Message(
                role="user",
                content=(
                    "Use the tile-style result surface for Blocked in the current card. The Blocked result "
                    "badge and result comment are nearby context, not the surface to use."
                ),
            ),
        ],
        media=["img-composed"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-composed": {
                    "local_layouts": [
                        {"region_id": "badge-1", "label": "result badge", "text": "Blocked"},
                        {"region_id": "tile-1", "label": "result tile", "text": "Blocked"},
                        {"region_id": "comment-1", "label": "result comment", "text": "Blocked pending counsel"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-composed", "target_query": "result comment"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_composed_route_gating(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call == [
        ToolCall(
            name="extract_layout",
            arguments={"image_id": "img-composed", "target_query": "result tile"},
            source_format="heuristic",
            raw=patched.normalized_tool_call[0].raw,
        )
    ]
    metadata = patched.runtime_metadata["visual_composed_route_gating"][0]
    assert metadata["from_arguments"] == {"image_id": "img-composed", "target_query": "result comment"}
    assert metadata["requested_label"] == "result tile"
    assert metadata["reason"] == "requested_surface_over_deprioritized_decoy"


def test_visual_composed_route_gating_rewrites_ignored_stale_selection_to_requested_surface() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="composed-stale-selection",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-stale-composed"),
            Message(
                role="user",
                content=(
                    "Ignore old selection sel-archived-result-badge. Use the tile-style result surface for "
                    "Blocked in the current visual state."
                ),
            ),
        ],
        media=["img-stale-composed"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-stale-composed": {
                    "local_layouts": [
                        {"region_id": "archived-1", "label": "result badge", "text": "Blocked"},
                        {"region_id": "tile-1", "label": "result tile", "text": "Blocked"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="refine_selection",
                arguments={"selection_id": "sel-archived-result-badge", "filter_query": "Blocked"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_composed_route_gating(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call == [
        ToolCall(
            name="extract_layout",
            arguments={"image_id": "img-stale-composed", "target_query": "result tile"},
            source_format="heuristic",
            raw=patched.normalized_tool_call[0].raw,
        )
    ]
    metadata = patched.runtime_metadata["visual_composed_route_gating"][0]
    assert metadata["from_tool"] == "refine_selection"
    assert metadata["requested_region_id"] == "tile-1"
    assert metadata["reason"] == "stale_selection_to_requested_surface"


def test_visual_composed_route_gating_restores_explicit_field_after_component_negation() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="composed-field-switch-decoy",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-mode"),
            Message(
                role="user",
                content=(
                    "Use the mode field in the current settings summary. The manual control and mode switch "
                    "are adjacent controls, not the field."
                ),
            ),
        ],
        media=["img-mode"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-mode": {
                    "local_layouts": [
                        {"region_id": "manual-1", "label": "manual control", "text": "Manual"},
                        {"region_id": "field-1", "label": "mode field", "text": "Manual"},
                        {"region_id": "switch-1", "label": "mode switch", "text": "Manual"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-mode", "target_query": "mode switch"},
                source_format="json",
                raw="{}",
            )
        ],
        runtime_metadata={
            "visual_target_query_normalization": [
                {
                    "from_tool": "extract_layout",
                    "from_arguments": {"image_id": "img-mode", "target_query": "mode field"},
                    "to_tool": "extract_layout",
                    "to_arguments": {"image_id": "img-mode", "target_query": "mode switch"},
                    "prompt_state_label": "mode switch",
                }
            ]
        },
    )

    patched = _apply_visual_composed_route_gating(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched.normalized_tool_call == [
        ToolCall(
            name="extract_layout",
            arguments={"image_id": "img-mode", "target_query": "mode field"},
            source_format="heuristic",
            raw=patched.normalized_tool_call[0].raw,
        )
    ]
    metadata = patched.runtime_metadata["visual_composed_route_gating"][0]
    assert metadata["requested_region_id"] == "field-1"
    assert metadata["reason"] == "requested_surface_over_deprioritized_decoy"


def test_visual_composed_route_gating_preserves_negated_explicit_surface_label() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="composed-negated-surface",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-result"),
            Message(
                role="user",
                content="Do not use the result tile. Select the result badge for Blocked, not the tile or the comment.",
            ),
        ],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "tile-1", "label": "result tile", "text": "Blocked"},
                        {"region_id": "badge-1", "label": "result badge", "text": "Blocked"},
                        {"region_id": "comment-1", "label": "result comment", "text": "Blocked pending counsel"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "result badge"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_composed_route_gating(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched == turn


def test_visual_composed_route_gating_preserves_exact_target_when_decoys_are_negated() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="composed-negated-decoys",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-result"),
            Message(role="user", content="The target is result tile. Do not use the result badge or the result comment."),
        ],
        media=["img-result"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-result": {
                    "local_layouts": [
                        {"region_id": "badge-1", "label": "result badge", "text": "Blocked"},
                        {"region_id": "tile-1", "label": "result tile", "text": "Blocked"},
                        {"region_id": "comment-1", "label": "result comment", "text": "Blocked pending counsel"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-result", "target_query": "result tile"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_composed_route_gating(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched == turn


def test_negation_aware_composed_route_gating_blocks_context_label_overwrite() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="composed-negation-scope-block",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-h2t-metric"),
            Message(
                role="user",
                content=(
                    "Use the metric panel at the top. The annotation saying 'not the metric panel' "
                    "is a training note about a prior screenshot, not the current target."
                ),
            ),
        ],
        media=["img-h2t-metric"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-h2t-metric": {
                    "local_layouts": [
                        {"region_id": "metric-1", "label": "metric panel", "text": "Escalations above target"},
                        {"region_id": "note-1", "label": "training note", "text": "old negative example"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-h2t-metric", "target_query": "metric panel"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_composed_route_gating(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
        preserve_negated_exact_layout_targets=True,
    )

    assert patched.normalized_tool_call == turn.normalized_tool_call
    blocked = patched.runtime_metadata["visual_composed_route_gating_blocked"][0]
    assert blocked["preserved_target_query"] == "metric panel"
    assert blocked["blocked_label"] == "training note"
    assert blocked["reason"] == "negation_scope_exact_layout_label"
    assert "visual_composed_route_gating" not in patched.runtime_metadata


def test_visual_composed_route_gating_preserves_field_target_when_switch_is_negated() -> None:
    specs = build_default_registry().specs
    case = ToolDirectiveProbeCase(
        case_id="composed-negated-switch",
        family="visual",
        messages=[
            Message(role="system", content="visual_image_ids: img-mode"),
            Message(role="user", content="Before reading the manual control, locate the mode field itself. Do not use the mode switch."),
        ],
        media=["img-mode"],
        tool_names=["extract_layout", "refine_selection"],
        initial_state={
            "visual_executor_mode": "local",
            "images": {
                "img-mode": {
                    "local_layouts": [
                        {"region_id": "manual-1", "label": "manual control", "text": "Manual"},
                        {"region_id": "field-1", "label": "mode field", "text": "Manual"},
                        {"region_id": "switch-1", "label": "mode switch", "text": "Auto"},
                    ]
                }
            },
        },
    )
    turn = ModelTurn(
        raw_model_output="{}",
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-mode", "target_query": "mode field"},
                source_format="json",
                raw="{}",
            )
        ],
    )

    patched = _apply_visual_composed_route_gating(
        turn=turn,
        case=case,
        tool_specs=[specs["extract_layout"], specs["refine_selection"]],
    )

    assert patched == turn


def test_tool_directive_probe_comparison_reports_case_and_family_deltas(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_probe:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cases = build_tool_directive_probe_cases()[:2]
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    run_tool_directive_probe(system_id="heuristic_probe", output_dir=baseline_dir, registry_path=registry_path, cases=cases)
    run_tool_directive_probe(system_id="heuristic_probe", output_dir=candidate_dir, registry_path=registry_path, cases=cases)

    candidate_rows = json.loads((candidate_dir / "probe_results.json").read_text(encoding="utf-8"))
    candidate_rows[0]["exact_match"] = False
    candidate_rows[0]["actual_calls"][0]["arguments"]["query"] = "invoice-lock failure"
    (candidate_dir / "probe_results.json").write_text(json.dumps(candidate_rows, indent=2) + "\n", encoding="utf-8")
    candidate_manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_manifest["summary"]["exact_match_count"] = 1
    candidate_manifest["summary"]["exact_match_rate"] = 0.5
    (candidate_dir / "manifest.json").write_text(json.dumps(candidate_manifest, indent=2) + "\n", encoding="utf-8")

    comparison = compare_tool_directive_probe_packets(baseline_dir, candidate_dir)
    outputs = write_tool_directive_probe_comparison(baseline_dir, candidate_dir)

    assert comparison["shared_case_count"] == 2
    assert comparison["delta_exact_match_rate"] == -0.5
    assert comparison["delta_executor_equivalence_match_rate"] is None
    first = next(row for row in comparison["case_deltas"] if row["case_id"] == cases[0].case_id)
    assert first["delta_exact_match"] == -1
    assert first["baseline_failure_mode"] == "exact"
    assert first["candidate_failure_mode"] == "argument_mismatch"
    assert first["baseline_executor_target_match"] is None
    assert first["candidate_executor_target_match"] is None
    family = next(row for row in comparison["family_deltas"] if row["family"] == cases[0].family)
    assert family["delta_exact_rate"] == -1.0
    assert family["delta_executor_target_rate"] is None
    assert Path(outputs["summary"]).exists()
    assert Path(outputs["case_deltas"]).exists()


def test_tool_directive_probe_scores_visual_paraphrase_execution() -> None:
    specs = build_default_registry().specs
    case = next(case for case in build_tool_directive_probe_cases() if case.case_id == "visual_form_target_literal")
    tool_specs = [specs[name] for name in case.tool_names]
    expected_calls = plan_tool_calls(case.messages, case.media, tool_specs)
    turn = ModelTurn(
        raw_model_output='{"name":"extract_layout","arguments":{"image_id":"img-form-live-latest","target_query":"phone issue"}}',
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-form-live-latest", "target_query": "phone issue"},
                source_format="json",
                raw='{"name":"extract_layout","arguments":{"image_id":"img-form-live-latest","target_query":"phone issue"}}',
            )
        ],
        runtime_metadata={"visual_target_query_normalization": [{"from_tool": "extract_layout"}]},
    )

    row = _score_probe_case(case, tool_specs, expected_calls, turn)

    assert row["exact_match"] is False
    assert row["executable_match"] is True
    assert row["executor_target_match"] is True
    assert row["actual_execution"][0]["output"]["region_ids"] == ["form-err-202"]
    assert row["runtime_metadata"] == {"visual_target_query_normalization": [{"from_tool": "extract_layout"}]}
