from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
