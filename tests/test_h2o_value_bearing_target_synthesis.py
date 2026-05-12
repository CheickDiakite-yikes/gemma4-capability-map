from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2O_MODULE_PATH = ROOT / "scripts" / "build_h2o_value_bearing_target_synthesis.py"

H2L_SPEC = importlib.util.spec_from_file_location("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
assert H2L_SPEC and H2L_SPEC.loader
H2L_SCRIPT = importlib.util.module_from_spec(H2L_SPEC)
sys.modules[H2L_SPEC.name] = H2L_SCRIPT
H2L_SPEC.loader.exec_module(H2L_SCRIPT)

H2O_SPEC = importlib.util.spec_from_file_location("build_h2o_value_bearing_target_synthesis", H2O_MODULE_PATH)
assert H2O_SPEC and H2O_SPEC.loader
H2O_SCRIPT = importlib.util.module_from_spec(H2O_SPEC)
sys.modules[H2O_SPEC.name] = H2O_SCRIPT
H2O_SPEC.loader.exec_module(H2O_SCRIPT)


def test_h2o_value_bearing_target_synthesis_repairs_h2m_strict_with_alias_residue(tmp_path: Path) -> None:
    payload = H2O_SCRIPT.build_h2o_value_bearing_target_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 10
    assert manifest["comparison_count"] == 6
    assert manifest["h2m_h2e_exact_success_count"] == 1
    assert manifest["h2m_h2e_executor_success_count"] == 3
    assert manifest["h2m_h2j_exact_success_count"] == 3
    assert manifest["h2m_h2j_executor_success_count"] == 3
    assert manifest["h2m_h2n_exact_success_count"] == 3
    assert manifest["h2m_h2n_executor_success_count"] == 5
    assert manifest["h2m_h2o_exact_success_count"] == 7
    assert manifest["h2m_h2o_executor_success_count"] == 7
    assert manifest["h2m_h2o_delta_exact_vs_h2n"] == 0.5
    assert manifest["h2m_h2o_delta_executor_vs_h2n"] == 0.25
    assert manifest["h2m_h2o_delta_exact_vs_h2j"] == 0.5
    assert manifest["h2m_h2o_delta_executor_vs_h2j"] == 0.5
    assert manifest["h2m_h2o_delta_exact_vs_h2e"] == 0.75
    assert manifest["h2m_h2o_delta_executor_vs_h2e"] == 0.5
    assert manifest["h2k_h2o_exact_success_count"] == 8
    assert manifest["h2k_h2o_executor_success_count"] == 8
    assert manifest["h2l_h2o_exact_success_count"] == 8
    assert manifest["h2l_h2o_executor_success_count"] == 8
    assert manifest["h2f_h2o_exact_success_count"] == 10
    assert manifest["h2f_h2o_executor_success_count"] == 10
    assert manifest["h2k_h2o_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2l_h2o_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2f_h2o_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2m_value_bearing_synthesis_count"] == 4
    assert manifest["h2m_target_query_normalization_count"] == 2
    assert manifest["h2m_stale_selection_count"] == 0
    assert manifest["h2m_non_exact_count"] == 1
    assert manifest["h2m_remaining_non_exact_case_id"] == "h2m_result_tile_contextual_alias"
    assert manifest["promotion_decision"] == (
        "h2o_value_bearing_synthesis_repairs_h2m_strict_with_contextual_alias_residue"
    )

    h2m_non_exact = {
        row["case_id"]: row
        for row in payload["non_exact_rows"]
        if row["profile_label"] == "h2m_h2o_value_bearing_target_query_synthesis"
    }
    assert set(h2m_non_exact) == {"h2m_result_tile_contextual_alias"}
    assert h2m_non_exact["h2m_result_tile_contextual_alias"]["expected_target_query"] == "result tile"
    assert h2m_non_exact["h2m_result_tile_contextual_alias"]["actual_target_query"] == "Blocked"
    assert h2m_non_exact["h2m_result_tile_contextual_alias"]["executor_equivalence_match"] is False

    synthesis = {
        row["case_id"]: row
        for row in payload["synthesis_rows"]
        if row["profile_label"] == "h2m_h2o_value_bearing_target_query_synthesis"
    }
    assert set(synthesis) == {
        "h2m_result_badge_blocked_contextual_value",
        "h2m_state_tag_closed_contextual_value",
        "h2m_mode_toggle_manual_contextual_value",
        "h2m_priority_badge_critical_contextual_value",
    }
    assert synthesis["h2m_result_badge_blocked_contextual_value"]["to_arguments"] == (
        '{"image_id":"img-h2m-result-badge-blocked","target_query":"result badge Blocked"}'
    )
    assert synthesis["h2m_state_tag_closed_contextual_value"]["to_arguments"] == (
        '{"image_id":"img-h2m-state-tag-closed","target_query":"state tag Closed"}'
    )
    assert synthesis["h2m_mode_toggle_manual_contextual_value"]["to_arguments"] == (
        '{"image_id":"img-h2m-mode-toggle-manual","target_query":"mode toggle Manual"}'
    )
    assert synthesis["h2m_priority_badge_critical_contextual_value"]["to_arguments"] == (
        '{"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge Critical"}'
    )
    assert {row["reason"] for row in synthesis.values()} == {"value_bearing_label_recoverable"}

    rewrites = {
        row["case_id"]: row
        for row in payload["intervention_rows"]
        if row["intervention_kind"] == "visual_target_query_normalization"
        and row["profile_label"] == "h2m_h2o_value_bearing_target_query_synthesis"
    }
    assert set(rewrites) == {
        "h2m_error_notice_contextual_alias",
        "h2m_mode_field_contextual_regression_guard",
    }
    assert rewrites["h2m_error_notice_contextual_alias"]["to_arguments"] == (
        '{"image_id":"img-h2m-error-notice","target_query":"error notice"}'
    )
    assert rewrites["h2m_mode_field_contextual_regression_guard"]["to_arguments"] == (
        '{"image_id":"img-h2m-mode-field-short","target_query":"mode field"}'
    )

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "from H2n's 3/8 to 7/8" in findings["h2o_repairs_h2m_strict_value_bearing_rows"]
    assert "zero exact-rate delta versus H2j" in findings["h2o_transfers_without_regression"]
    assert "surface-type alias" in findings["h2p_should_target_contextual_surface_aliases"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2o_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2o_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2o_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2o_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2o_value_bearing_synthesis_rows.csv").exists()
    assert (tmp_path / "tables" / "h2o_findings.csv").exists()
    assert (tmp_path / "figures" / "h2o_value_bearing_target_synthesis_gate.svg").exists()
