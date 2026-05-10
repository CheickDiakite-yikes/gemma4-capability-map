from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from gemma4_capability_map.schemas import ToolCall, ToolSpec
from gemma4_capability_map.tools.executor import DeterministicExecutor


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_visual_hard_slice_live_stress_packet.py"
SPEC = importlib.util.spec_from_file_location("build_visual_hard_slice_live_stress_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_visual_hard_slice_live_stress_packet_writes_replay_artifacts(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_dry_run",
    )

    packet_dir = Path(packet["packet_dir"])
    assert packet["summary"]["case_count"] == 4
    assert packet["summary"]["family_counts"] == {
        "visual_argument_copying_stress": 2,
        "visual_tool_routing_stress": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 2,
        "wrong_tool_or_stale_selection_risk": 2,
    }
    assert packet["manifest"]["operator_surface"] == "rich_cli_visual_hard_slice_live_stress_v1"
    assert packet["manifest"]["entrypoint"] == "moonie-agent replay-live"
    assert packet["rows"][0]["expected_call_count"] == 1
    assert packet["replay_cases"][0]["expected_execution"]
    assert packet["replay_cases"][0]["live_entrypoint_status"] == "visual_hard_slice_live_stress_packet_v1"
    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "summary.json").exists()
    assert (packet_dir / "commands.json").exists()
    assert (packet_dir / "replay_cases.json").exists()
    assert (packet_dir / "replay_cases.csv").exists()
    assert (packet_dir / "cases" / "stress_metric_panel_with_chart_table_decoys.json").exists()
    command = packet["commands"][0]["command"]
    assert command[1] == "-m"
    assert "replay-live" in command
    assert "--case-id" in command


def test_visual_hard_slice_live_stress_packet_filters_cases(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_filtered",
        case_ids=["stress_form_error_stale_selection_status_decoy"],
    )

    assert packet["summary"]["case_count"] == 1
    assert packet["rows"][0]["case_id"] == "stress_form_error_stale_selection_status_decoy"
    assert packet["rows"][0]["family"] == "visual_tool_routing_stress"


def test_visual_hard_slice_live_stress_packet_supports_alias_repeat_suite(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_alias_repeat",
        suite="alias_repeat_v2",
    )

    assert packet["summary"]["suite"] == "alias_repeat_v2"
    assert packet["summary"]["case_count"] == 8
    assert packet["summary"]["family_counts"] == {
        "visual_argument_copying_stress": 6,
        "visual_tool_routing_stress": 2,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "stress_metric_panel_status_banner_decoy" in case_ids
    assert "stress_callout_warning_risk_note_decoy" in case_ids


def test_visual_hard_slice_live_stress_packet_supports_alias_transfer_suite(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_alias_transfer",
        suite="alias_transfer_v3",
    )

    assert packet["summary"]["suite"] == "alias_transfer_v3"
    assert packet["summary"]["case_count"] == 6
    assert packet["summary"]["family_counts"] == {
        "visual_argument_transfer": 4,
        "visual_tool_routing_transfer": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 4,
        "wrong_tool_or_stale_selection_risk": 2,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "transfer_review_tile_notice_table_decoy" in case_ids
    assert "transfer_signature_warning_checkbox_decoy" in case_ids
    assert packet["replay_cases"][0]["live_entrypoint_status"] == "visual_hard_slice_live_stress_packet_v1"
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["transfer_review_tile_notice_table_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-transfer-review-tile",
                "target_query": "review tile",
            },
        }
    ]
    assert cases["transfer_queue_badge_person_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-transfer-queue-badge",
                "target_query": "queue badge",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_alias_transfer_repeat_suite(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_alias_transfer_repeat",
        suite="alias_transfer_repeat_v4",
    )

    assert packet["summary"]["suite"] == "alias_transfer_repeat_v4"
    assert packet["summary"]["case_count"] == 6
    assert packet["summary"]["family_counts"] == {
        "visual_argument_transfer_repeat": 4,
        "visual_tool_routing_transfer_repeat": 2,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "transfer_repeat_audit_card_email_decoy" in case_ids
    assert "transfer_repeat_consent_alert_toggle_decoy" in case_ids
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["transfer_repeat_audit_card_email_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-transfer-repeat-audit-card",
                "target_query": "audit card",
            },
        }
    ]
    assert cases["transfer_repeat_missing_field_old_selection_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-transfer-repeat-missing-field",
                "target_query": "missing field message",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_alias_transfer_oblique_suite(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_alias_transfer_oblique",
        suite="alias_transfer_oblique_v5",
    )

    assert packet["summary"]["suite"] == "alias_transfer_oblique_v5"
    assert packet["summary"]["case_count"] == 6
    assert packet["summary"]["family_counts"] == {
        "visual_argument_transfer_oblique": 4,
        "visual_tool_routing_transfer_oblique": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 4,
        "wrong_tool_or_stale_selection_risk": 2,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "transfer_oblique_node_q17_table_decoy" in case_ids
    assert "transfer_oblique_alert_p55_toggle_decoy" in case_ids
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["transfer_oblique_node_q17_table_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-transfer-oblique-node-q17",
                "target_query": "node q17",
            },
        }
    ]
    assert cases["transfer_oblique_field_e19_old_selection_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-transfer-oblique-field-e19",
                "target_query": "field e19",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_post_repair_suite(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_alias_transfer_post_repair",
        suite="alias_transfer_post_repair_v6",
    )

    assert packet["summary"]["suite"] == "alias_transfer_post_repair_v6"
    assert packet["summary"]["case_count"] == 8
    assert packet["summary"]["family_counts"] == {
        "visual_argument_transfer_post_repair_code": 3,
        "visual_argument_transfer_post_repair_noncode": 3,
        "visual_tool_routing_transfer_post_repair": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 6,
        "wrong_tool_or_stale_selection_risk": 2,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "post_repair_node_k21_chart_decoy" in case_ids
    assert "post_repair_field_b12_stale_selection_decoy" in case_ids
    assert "post_repair_warning_toast_email_decoy" in case_ids
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["post_repair_node_k21_chart_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-post-node-k21",
                "target_query": "node k21",
            },
        }
    ]
    assert cases["post_repair_status_pill_note_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-post-status-pill",
                "target_query": "status pill",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_residual_suite(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_alias_transfer_residual",
        suite="alias_transfer_residual_v7",
    )

    assert packet["summary"]["suite"] == "alias_transfer_residual_v7"
    assert packet["summary"]["case_count"] == 8
    assert packet["summary"]["family_counts"] == {
        "visual_argument_transfer_residual_code": 3,
        "visual_argument_transfer_residual_noncode": 3,
        "visual_tool_routing_transfer_residual": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 6,
        "wrong_tool_or_stale_selection_risk": 2,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "residual_chip_n31_owner_note_decoy" in case_ids
    assert "residual_state_pill_note_decoy" in case_ids
    assert "residual_field_m20_stale_selection_decoy" in case_ids
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["residual_chip_n31_owner_note_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-residual-chip-n31",
                "target_query": "chip n31",
            },
        }
    ]
    assert cases["residual_state_pill_note_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-residual-state-pill",
                "target_query": "state pill",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_component_value_suite(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_component_value",
        suite="component_value_v9",
    )

    assert packet["summary"]["suite"] == "component_value_v9"
    assert packet["summary"]["case_count"] == 8
    assert packet["summary"]["family_counts"] == {
        "visual_argument_transfer_component_value_nonpill": 3,
        "visual_argument_transfer_component_value_pill": 3,
        "visual_tool_routing_component_value": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 6,
        "wrong_tool_or_stale_selection_risk": 2,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "component_value_state_pill_note_decoy" in case_ids
    assert "component_value_status_badge_email_decoy" in case_ids
    assert "component_value_owner_field_stale_selection_decoy" in case_ids
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["component_value_state_pill_note_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-component-state-pill",
                "target_query": "state pill",
            },
        }
    ]
    assert cases["component_value_owner_field_stale_selection_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-component-owner-field",
                "target_query": "owner field",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_h1o_control_factorial_suite(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_h1o_control_factorial",
        suite="h1o_control_factorial_v10",
    )

    assert packet["summary"]["suite"] == "h1o_control_factorial_v10"
    assert packet["summary"]["case_count"] == 12
    assert packet["summary"]["family_counts"] == {
        "h1o_activation_no_call": 4,
        "h1o_code_negation_preservation": 4,
        "h1o_component_value_boundary": 4,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 8,
        "wrong_tool_or_stale_selection_risk": 4,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "h1o_activation_status_badge_email_decoy" in case_ids
    assert "h1o_code_alert_s92_negated_toggle_decoy" in case_ids
    assert "h1o_component_priority_chip_value_decoy" in case_ids
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["h1o_activation_status_badge_email_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1o-activation-status-badge",
                "target_query": "status badge",
            },
        }
    ]
    assert cases["h1o_code_alert_s92_negated_toggle_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1o-code-alert-s92",
                "target_query": "alert s92",
            },
        }
    ]
    assert cases["h1o_component_priority_chip_value_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1o-component-priority-chip",
                "target_query": "priority chip",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_h1p_component_value_holdout_suite(
    tmp_path: Path,
) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_h1p_component_value",
        suite="h1p_component_value_holdout_v11",
    )

    assert packet["summary"]["suite"] == "h1p_component_value_holdout_v11"
    assert packet["summary"]["case_count"] == 12
    assert packet["summary"]["family_counts"] == {
        "h1p_component_value_compact": 4,
        "h1p_component_value_stale_selection": 4,
        "h1p_component_value_surface": 4,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 8,
        "wrong_tool_or_stale_selection_risk": 4,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "h1p_compact_status_pill_summary_value_decoy" in case_ids
    assert "h1p_surface_mode_toggle_note_value_decoy" in case_ids
    assert "h1p_stale_phase_tile_archive_decoy" in case_ids
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["h1p_compact_stage_chip_email_value_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1p-compact-stage-chip",
                "target_query": "stage chip",
            },
        }
    ]
    assert cases["h1p_surface_owner_field_note_value_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1p-surface-owner-field",
                "target_query": "owner field",
            },
        }
    ]
    assert cases["h1p_stale_risk_badge_old_selection_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1p-stale-risk-badge",
                "target_query": "risk badge",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def _expected_call_reaches_oracle(case: dict[str, object]) -> bool:
    tool_specs = [ToolSpec.model_validate(payload) for payload in case["tool_specs"]]  # type: ignore[index]
    executor = DeterministicExecutor(tool_specs=tool_specs)
    state = case["initial_state"]  # type: ignore[index]
    execution = []
    for step, payload in enumerate(case["expected_calls"], start=1):  # type: ignore[index]
        call = ToolCall(
            name=payload["name"],
            arguments=payload["arguments"],
            source_format="oracle",
            raw=str(payload),
        )
        result = executor.step(state, call, step=step)
        state = result.state_after
        execution.append(result)
    expected_region_ids = [str(region_id) for region_id in case["expected_execution"]["region_ids"]]  # type: ignore[index]
    actual_region_ids = execution[-1].output.get("region_ids", [])
    return execution[-1].validator_result == "pass" and actual_region_ids == expected_region_ids
