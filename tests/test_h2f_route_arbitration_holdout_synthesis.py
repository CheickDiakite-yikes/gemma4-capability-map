from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2f_route_arbitration_holdout_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2f_route_arbitration_holdout_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2f_route_arbitration_holdout_synthesis_breaks_h2e_saturation(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2f_route_arbitration_holdout_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 7
    assert manifest["comparison_count"] == 8
    assert manifest["h2e_exact_success_count"] == 6
    assert manifest["h2e_executor_success_count"] == 6
    assert manifest["h2g_exact_success_count"] == 6
    assert manifest["h2g_executor_success_count"] == 7
    assert manifest["h2c_exact_success_count"] == 6
    assert manifest["no_directive_exact_success_count"] == 1
    assert manifest["h2e_non_exact_count"] == 4
    assert manifest["h2g_non_exact_count"] == 4
    assert manifest["h2e_delta_exact_vs_h2c"] == 0.0
    assert manifest["h2e_delta_executor_vs_h2c"] == 0.0
    assert manifest["h2e_delta_exact_vs_no_directive"] == 0.5
    assert manifest["h2e_delta_executor_vs_no_directive"] == 0.5
    assert manifest["h2g_delta_exact_vs_h2e"] == 0.0
    assert manifest["h2g_delta_executor_vs_h2e"] == 0.09999999999999998
    assert manifest["h2e_failure_family_count"] == 2
    assert manifest["promotion_decision"] == "reject_global_h2e_and_h2g_build_h2h_stronger_component_identity_contract"

    packet_rows = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packet_rows["no_directive"]["exact_success_count"] == 1
    assert packet_rows["h2a_component_label_guard"]["exact_success_count"] == 4
    assert packet_rows["component_residual_guard_v12"]["exact_success_count"] == 5
    assert packet_rows["component_residual_guard_v12"]["executor_success_count"] == 6
    assert packet_rows["h2d_class_preserving_route"]["exact_success_count"] == 5
    assert packet_rows["h2c_scoped_residual_gate"]["exact_success_count"] == 6
    assert packet_rows["h2e_route_arbitration"]["exact_success_count"] == 6
    assert packet_rows["h2g_component_identity_query_contract"]["exact_success_count"] == 6
    assert packet_rows["h2g_component_identity_query_contract"]["executor_success_count"] == 7

    h2e_non_exact = {row["case_id"]: row for row in payload["h2e_non_exact_rows"]}
    assert set(h2e_non_exact) == {
        "h2f_result_tile_comment_value_decoy",
        "h2f_resolution_badge_log_result_decoy",
        "h2f_state_marker_history_value_decoy",
        "h2f_mode_switch_note_value_decoy",
    }
    assert h2e_non_exact["h2f_result_tile_comment_value_decoy"]["expected_target_query"] == "result tile"
    assert h2e_non_exact["h2f_result_tile_comment_value_decoy"]["actual_target_query"] == "Blocked"
    assert h2e_non_exact["h2f_resolution_badge_log_result_decoy"]["actual_target_query"] == "Deferred"
    assert h2e_non_exact["h2f_state_marker_history_value_decoy"]["actual_target_query"] == "lifecycle state marker"
    assert h2e_non_exact["h2f_mode_switch_note_value_decoy"]["actual_target_query"] == "mode toggle"
    assert {
        row["query_error_class"] for row in h2e_non_exact.values()
    } == {"value_or_alias_query_substitution"}

    h2g_non_exact = {row["case_id"]: row for row in payload["h2g_non_exact_rows"]}
    assert set(h2g_non_exact) == set(h2e_non_exact)
    assert h2g_non_exact["h2f_resolution_badge_log_result_decoy"]["actual_target_query"] == (
        "resolution badge Deferred"
    )
    assert h2g_non_exact["h2f_resolution_badge_log_result_decoy"]["executor_equivalence_match"] is True

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2e reaches only 6/10 exact" in findings["h2f_breaks_h2e_saturation"]
    assert "delta exact=0.0" in findings["route_arbitration_does_not_beat_h2c_on_h2f"]
    assert "No-directive reaches 1/10 exact" in findings["controllers_remain_causal_against_floor"]
    assert "substituted displayed values" in findings["remaining_failure_is_component_identity_binding"]
    assert "H2g keeps strict exactness at 6/10" in findings["next_slice"]
    assert "Build H2h as a stronger component-identity contract" in findings["next_contract"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2f_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2f_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2f_h2e_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2f_h2g_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2f_all_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2f_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2f_failure_mode_summary.csv").exists()
    assert (tmp_path / "tables" / "h2f_findings.csv").exists()
    assert (tmp_path / "figures" / "h2f_holdout_profile_bars.svg").exists()
