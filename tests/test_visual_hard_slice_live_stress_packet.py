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


def test_visual_hard_slice_live_stress_packet_supports_h1r_component_label_residual_suite(
    tmp_path: Path,
) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_h1r_component_label_residual",
        suite="h1r_component_label_residual_v12",
    )

    assert packet["summary"]["suite"] == "h1r_component_label_residual_v12"
    assert packet["summary"]["case_count"] == 6
    assert packet["summary"]["family_counts"] == {
        "h1r_code_label_exactness": 2,
        "h1r_nonstandard_component_class": 2,
        "h1r_stale_selection_component_label": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 4,
        "wrong_tool_or_stale_selection_risk": 2,
    }
    case_ids = {row["case_id"] for row in packet["rows"]}
    assert "h1r_owner_field_stale_selection_note_decoy" in case_ids
    assert "h1r_state_tag_log_value_decoy" in case_ids
    assert "h1r_alert_s92_toggle_negation_decoy" in case_ids
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["h1r_owner_field_stale_selection_note_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1r-owner-field",
                "target_query": "owner field",
            },
        }
    ]
    assert cases["h1r_state_tag_log_value_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1r-state-tag",
                "target_query": "state tag",
            },
        }
    ]
    assert cases["h1r_alert_s92_toggle_negation_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1r-alert-s92",
                "target_query": "alert s92",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_h1w_residual_overlap_suite(
    tmp_path: Path,
) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_h1w_residual_overlap",
        suite="h1w_residual_overlap_v13",
    )

    assert packet["summary"]["suite"] == "h1w_residual_overlap_v13"
    assert packet["summary"]["case_count"] == 8
    assert packet["summary"]["family_counts"] == {
        "h1w_activation_no_call": 2,
        "h1w_nonstandard_component_class": 2,
        "h1w_stale_field_routing": 2,
        "h1w_surface_component_value": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 4,
        "wrong_tool_or_stale_selection_risk": 4,
    }
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["h1w_owner_field_memo_stale_selection_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1w-owner-field",
                "target_query": "owner field",
            },
        }
    ]
    assert cases["h1w_mode_toggle_settings_note_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1w-mode-toggle",
                "target_query": "mode toggle",
            },
        }
    ]
    assert cases["h1w_result_badge_comment_value_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1w-result-badge",
                "target_query": "result badge",
            },
        }
    ]
    assert cases["h1w_warning_tile_no_call_note_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1w-warning-tile",
                "target_query": "warning tile",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_h1x_v11_breaker_suite(
    tmp_path: Path,
) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_h1x_v11_breaker",
        suite="h1x_v11_breaker_v14",
    )

    assert packet["summary"]["suite"] == "h1x_v11_breaker_v14"
    assert packet["summary"]["case_count"] == 8
    assert packet["summary"]["family_counts"] == {
        "h1x_oblique_activation_no_call": 2,
        "h1x_oblique_nonstandard_class": 2,
        "h1x_oblique_stale_field": 2,
        "h1x_oblique_surface_value": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 4,
        "wrong_tool_or_stale_selection_risk": 4,
    }
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["h1x_responsible_party_field_old_owner_memo_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1x-owner-field",
                "target_query": "owner field",
            },
        }
    ]
    assert cases["h1x_resolution_chip_comment_result_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1x-resolution-chip",
                "target_query": "result chip",
            },
        }
    ]
    assert cases["h1x_lifecycle_marker_log_state_tag_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1x-lifecycle-marker",
                "target_query": "state tag",
            },
        }
    ]
    assert cases["h1x_warning_panel_note_activation_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1x-warning-panel",
                "target_query": "warning tile",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_h1y_routed_residual_suite(
    tmp_path: Path,
) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_h1y_routed_residual",
        suite="h1y_routed_residual_v15",
    )

    assert packet["summary"]["suite"] == "h1y_routed_residual_v15"
    assert packet["summary"]["case_count"] == 10
    assert packet["summary"]["family_counts"] == {
        "h1y_activation_no_call": 1,
        "h1y_preserve_surface_value": 2,
        "h1y_route_code_label": 2,
        "h1y_route_nonstandard_class": 2,
        "h1y_route_stale_field": 3,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 6,
        "wrong_tool_or_stale_selection_risk": 4,
    }
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["h1y_responsible_party_field_old_owner_memo_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1y-owner-field",
                "target_query": "owner field",
            },
        }
    ]
    assert cases["h1y_alert_s92_negated_toggle_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1y-alert-s92",
                "target_query": "alert s92",
            },
        }
    ]
    assert cases["h1y_status_pill_summary_value_holdout"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1y-status-pill",
                "target_query": "status pill",
            },
        }
    ]
    assert cases["h1y_warning_tile_note_activation_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h1y-warning-tile",
                "target_query": "warning tile",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_h2f_route_arbitration_suite(
    tmp_path: Path,
) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_h2f_route_arbitration",
        suite="h2f_route_arbitration_v16",
    )

    assert packet["summary"]["suite"] == "h2f_route_arbitration_v16"
    assert packet["summary"]["case_count"] == 10
    assert packet["summary"]["family_counts"] == {
        "h2f_activation_panel_notice": 2,
        "h2f_route_code_label": 2,
        "h2f_route_component_class_transfer": 2,
        "h2f_route_nonstandard_class": 2,
        "h2f_route_stale_field": 2,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 6,
        "wrong_tool_or_stale_selection_risk": 4,
    }
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["h2f_alert_t47_negated_switch_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2f-alert-t47",
                "target_query": "alert t47",
            },
        }
    ]
    assert cases["h2f_badge_m31_summary_value_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2f-badge-m31",
                "target_query": "badge m31",
            },
        }
    ]
    assert cases["h2f_result_tile_comment_value_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2f-result-tile",
                "target_query": "result tile",
            },
        }
    ]
    assert cases["h2f_owner_field_previous_memo_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2f-owner-field",
                "target_query": "owner field",
            },
        }
    ]
    assert cases["h2f_error_notice_history_activation_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2f-error-notice",
                "target_query": "error notice",
            },
        }
    ]
    for case in packet["replay_cases"]:
        assert _expected_call_reaches_oracle(case)


def test_visual_hard_slice_live_stress_packet_supports_h2k_target_decoy_overlap_suite(
    tmp_path: Path,
) -> None:
    packet = SCRIPT.build_visual_hard_slice_live_stress_packet(
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_stress_h2k_target_decoy_overlap",
        suite="h2k_target_decoy_overlap_v17",
    )

    assert packet["summary"]["suite"] == "h2k_target_decoy_overlap_v17"
    assert packet["summary"]["case_count"] == 8
    assert packet["summary"]["family_counts"] == {
        "h2k_before_reading_decoy": 2,
        "h2k_code_label_overlap": 2,
        "h2k_negated_same_component_decoy": 3,
        "h2k_transfer_regression_guard": 1,
    }
    assert packet["summary"]["failure_mode_counts"] == {
        "argument_alias_or_decoy_risk": 8,
    }
    cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert cases["h2k_priority_badge_negated_status_badge_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2k-priority-badge",
                "target_query": "priority badge",
            },
        }
    ]
    assert cases["h2k_result_badge_negated_result_tile_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2k-result-badge",
                "target_query": "result badge",
            },
        }
    ]
    assert cases["h2k_state_tag_before_reading_state_marker_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2k-state-tag",
                "target_query": "state tag",
            },
        }
    ]
    assert cases["h2k_badge_c18_negated_badge_c08_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2k-badge-c18",
                "target_query": "badge c18",
            },
        }
    ]
    assert cases["h2k_alert_t47_archived_alert_s92_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-h2k-alert-t47",
                "target_query": "alert t47",
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
