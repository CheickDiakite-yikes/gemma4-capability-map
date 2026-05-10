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
