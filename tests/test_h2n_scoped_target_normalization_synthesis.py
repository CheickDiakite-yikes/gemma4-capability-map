from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2N_MODULE_PATH = ROOT / "scripts" / "build_h2n_scoped_target_normalization_synthesis.py"

H2L_SPEC = importlib.util.spec_from_file_location("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
assert H2L_SPEC and H2L_SPEC.loader
H2L_SCRIPT = importlib.util.module_from_spec(H2L_SPEC)
sys.modules[H2L_SPEC.name] = H2L_SCRIPT
H2L_SPEC.loader.exec_module(H2L_SCRIPT)

H2N_SPEC = importlib.util.spec_from_file_location("build_h2n_scoped_target_normalization_synthesis", H2N_MODULE_PATH)
assert H2N_SPEC and H2N_SPEC.loader
H2N_SCRIPT = importlib.util.module_from_spec(H2N_SPEC)
sys.modules[H2N_SPEC.name] = H2N_SCRIPT
H2N_SPEC.loader.exec_module(H2N_SCRIPT)


def test_h2n_scoped_target_normalization_synthesis_is_executor_gain_not_strict_repair(tmp_path: Path) -> None:
    payload = H2N_SCRIPT.build_h2n_scoped_target_normalization_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 9
    assert manifest["comparison_count"] == 5
    assert manifest["h2m_h2e_exact_success_count"] == 1
    assert manifest["h2m_h2e_executor_success_count"] == 3
    assert manifest["h2m_h2j_exact_success_count"] == 3
    assert manifest["h2m_h2j_executor_success_count"] == 3
    assert manifest["h2m_h2n_exact_success_count"] == 3
    assert manifest["h2m_h2n_executor_success_count"] == 5
    assert manifest["h2m_h2n_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2m_h2n_delta_executor_vs_h2j"] == 0.25
    assert manifest["h2m_h2n_delta_exact_vs_h2e"] == 0.25
    assert manifest["h2m_h2n_delta_executor_vs_h2e"] == 0.25
    assert manifest["h2k_h2n_exact_success_count"] == 8
    assert manifest["h2k_h2n_executor_success_count"] == 8
    assert manifest["h2l_h2n_exact_success_count"] == 8
    assert manifest["h2l_h2n_executor_success_count"] == 8
    assert manifest["h2f_h2n_exact_success_count"] == 10
    assert manifest["h2f_h2n_executor_success_count"] == 10
    assert manifest["h2k_h2n_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2l_h2n_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2f_h2n_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2m_blocked_value_bearing_count"] == 3
    assert manifest["h2m_target_query_normalization_count"] == 2
    assert manifest["h2m_stale_selection_count"] == 0
    assert manifest["promotion_decision"] == "h2n_scoped_target_normalization_executor_gain_needs_strict_repair"

    h2m_non_exact = {
        row["case_id"]: row
        for row in payload["non_exact_rows"]
        if row["profile_label"] == "h2m_h2n_scoped_target_query_normalization"
    }
    assert len(h2m_non_exact) == 5
    assert h2m_non_exact["h2m_result_badge_blocked_contextual_value"]["actual_target_query"] == "result chip"
    assert h2m_non_exact["h2m_result_badge_blocked_contextual_value"]["executor_equivalence_match"] is False
    assert h2m_non_exact["h2m_state_tag_closed_contextual_value"]["actual_target_query"] == "Closed state tag"
    assert h2m_non_exact["h2m_state_tag_closed_contextual_value"]["executor_equivalence_match"] is True
    assert h2m_non_exact["h2m_priority_badge_critical_contextual_value"]["actual_target_query"] == (
        "priority badge critical"
    )
    assert h2m_non_exact["h2m_priority_badge_critical_contextual_value"]["executor_equivalence_match"] is True
    assert h2m_non_exact["h2m_mode_toggle_manual_contextual_value"]["actual_target_query"] == "mode toggle"
    assert h2m_non_exact["h2m_result_tile_contextual_alias"]["actual_target_query"] == "Blocked"

    blocked = {row["case_id"]: row for row in payload["blocked_rows"]}
    assert set(blocked) == {
        "h2m_result_badge_blocked_contextual_value",
        "h2m_state_tag_closed_contextual_value",
        "h2m_priority_badge_critical_contextual_value",
    }
    assert blocked["h2m_result_badge_blocked_contextual_value"]["preserved_target_query"] == "result chip"
    assert blocked["h2m_result_badge_blocked_contextual_value"]["value_bearing_label"] == "result badge Blocked"
    assert blocked["h2m_state_tag_closed_contextual_value"]["preserved_target_query"] == "Closed state tag"
    assert blocked["h2m_state_tag_closed_contextual_value"]["value_bearing_label"] == "state tag Closed"
    assert blocked["h2m_priority_badge_critical_contextual_value"]["preserved_target_query"] == (
        "priority badge critical"
    )
    assert blocked["h2m_priority_badge_critical_contextual_value"]["value_bearing_label"] == (
        "priority badge Critical"
    )
    assert {row["reason"] for row in blocked.values()} == {"value_bearing_label_requested"}

    rewrites = {
        row["case_id"]: row
        for row in payload["intervention_rows"]
        if row["intervention_kind"] == "visual_target_query_normalization"
        and row["profile_label"] == "h2m_h2n_scoped_target_query_normalization"
    }
    assert set(rewrites) == {
        "h2m_error_notice_contextual_alias",
        "h2m_mode_field_contextual_regression_guard",
    }
    assert rewrites["h2m_error_notice_contextual_alias"]["from_arguments"] == (
        '{"image_id":"img-h2m-error-notice","target_query":"archive panel"}'
    )
    assert rewrites["h2m_error_notice_contextual_alias"]["to_arguments"] == (
        '{"image_id":"img-h2m-error-notice","target_query":"error notice"}'
    )
    assert rewrites["h2m_mode_field_contextual_regression_guard"]["from_arguments"] == (
        '{"image_id":"img-h2m-mode-field-short","target_query":"mode switch"}'
    )
    assert rewrites["h2m_mode_field_contextual_regression_guard"]["to_arguments"] == (
        '{"image_id":"img-h2m-mode-field-short","target_query":"mode field"}'
    )

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "improves executor-equivalence from 3/8 to 5/8" in findings[
        "h2n_improves_h2m_executor_equivalence_not_strict"
    ]
    assert "zero exact-rate delta versus H2j" in findings["h2n_transfers_without_regression"]
    assert "canonical value-bearing target queries" in findings[
        "next_gate_needs_canonical_value_bearing_target_synthesis"
    ]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2n_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2n_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2n_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2n_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2n_blocked_rows.csv").exists()
    assert (tmp_path / "tables" / "h2n_findings.csv").exists()
    assert (tmp_path / "figures" / "h2n_scoped_target_normalization_gate.svg").exists()
