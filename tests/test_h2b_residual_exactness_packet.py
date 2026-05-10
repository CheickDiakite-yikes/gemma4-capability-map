from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2b_residual_exactness_packet.py"
SPEC = importlib.util.spec_from_file_location("build_h2b_residual_exactness_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2b_residual_exactness_packet_composes_h2a_transfer_residuals(tmp_path: Path) -> None:
    packet = SCRIPT.build_h2b_residual_exactness_packet(
        output_root=tmp_path,
        run_group_id="test_h2b_residual_exactness_packet",
    )

    summary = packet["summary"]
    assert summary["case_count"] == 5
    assert summary["h2a_failure_mode_counts"] == {
        "argument_mismatch": 2,
        "executable_paraphrase": 3,
    }
    assert summary["residual_axis_counts"] == {
        "executor_equivalent_alias": 3,
        "non_executor_argument_mismatch": 2,
    }
    assert summary["h2a_executor_equivalent_count"] == 3
    assert summary["h2a_non_executor_count"] == 2

    rows = {row["case_id"]: row for row in packet["rows"]}
    assert rows["component_value_result_pill_log_decoy"]["residual_class"] == "result_pill_exact_label"
    assert rows["h1o_code_alert_s92_negated_toggle_decoy"]["residual_class"] == "alert_s92_code_label"
    assert rows["h1o_code_badge_c08_note_decoy"]["residual_class"] == "badge_c08_code_label"
    assert rows["h1p_compact_state_tag_log_value_decoy"]["residual_class"] == "state_tag_component_class"
    assert rows["h1p_surface_mode_toggle_note_value_decoy"]["residual_class"] == "mode_toggle_component_class"
    assert rows["h1p_surface_mode_toggle_note_value_decoy"]["h2a_failure_mode"] == "argument_mismatch"
    assert rows["h1p_surface_mode_toggle_note_value_decoy"]["source_failure_mode"] == "argument_mismatch"
    assert rows["h1p_surface_mode_toggle_note_value_decoy"]["expected_call_count"] == 1
    assert rows["h1p_surface_mode_toggle_note_value_decoy"]["source_executable_match"] is False
    assert rows["h1o_code_alert_s92_negated_toggle_decoy"]["source_executable_match"] is True

    replay_cases = {case["case_id"]: case for case in packet["replay_cases"]}
    assert replay_cases["component_value_result_pill_log_decoy"]["expected_calls"] == [
        {
            "name": "extract_layout",
            "arguments": {
                "image_id": "img-component-result-pill",
                "target_query": "result pill",
            },
        }
    ]
    assert replay_cases["h1o_code_alert_s92_negated_toggle_decoy"]["h2a_actual_arguments"] == {
        "image_id": "img-h1o-code-alert-s92",
        "target_query": "alert",
    }
    assert replay_cases["h1p_compact_state_tag_log_value_decoy"]["source_executable_match"] is False

    packet_dir = Path(packet["packet_dir"])
    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "summary.json").exists()
    assert (packet_dir / "commands.json").exists()
    assert (packet_dir / "replay_cases.json").exists()
    assert (packet_dir / "replay_cases.csv").exists()
    assert (packet_dir / "cases" / "h1p_surface_mode_toggle_note_value_decoy.json").exists()
