from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2M_MODULE_PATH = ROOT / "scripts" / "build_h2m_less_direct_overreach_synthesis.py"

H2L_SPEC = importlib.util.spec_from_file_location("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
assert H2L_SPEC and H2L_SPEC.loader
H2L_SCRIPT = importlib.util.module_from_spec(H2L_SPEC)
sys.modules[H2L_SPEC.name] = H2L_SCRIPT
H2L_SPEC.loader.exec_module(H2L_SCRIPT)

H2M_SPEC = importlib.util.spec_from_file_location("build_h2m_less_direct_overreach_synthesis", H2M_MODULE_PATH)
assert H2M_SPEC and H2M_SPEC.loader
H2M_SCRIPT = importlib.util.module_from_spec(H2M_SPEC)
sys.modules[H2M_SPEC.name] = H2M_SCRIPT
H2M_SPEC.loader.exec_module(H2M_SCRIPT)


def test_h2m_less_direct_overreach_synthesis_rejects_current_scope(tmp_path: Path) -> None:
    payload = H2M_SCRIPT.build_h2m_less_direct_overreach_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 3
    assert manifest["comparison_count"] == 2
    assert manifest["family_row_count"] == 3
    assert manifest["h2e_exact_success_count"] == 1
    assert manifest["h2e_executor_success_count"] == 3
    assert manifest["h2j_no_stale_exact_success_count"] == 3
    assert manifest["h2j_no_stale_executor_success_count"] == 3
    assert manifest["h2j_exact_success_count"] == 3
    assert manifest["h2j_executor_success_count"] == 3
    assert manifest["h2j_delta_exact_vs_h2e"] == 0.25
    assert manifest["h2j_delta_executor_vs_h2e"] == 0.0
    assert manifest["h2j_delta_exact_vs_no_stale_gate"] == 0.0
    assert manifest["h2j_delta_executor_vs_no_stale_gate"] == 0.0
    assert manifest["h2e_non_exact_count"] == 7
    assert manifest["h2j_no_stale_non_exact_count"] == 5
    assert manifest["h2j_non_exact_count"] == 5
    assert manifest["target_query_normalization_count"] == 5
    assert manifest["visual_stale_selection_gate_count"] == 0
    assert manifest["h2j_no_stale_target_query_normalization_count"] == 5
    assert manifest["h2j_no_stale_visual_stale_selection_gate_count"] == 0
    assert manifest["h2j_helpful_target_query_normalization_count"] == 2
    assert manifest["h2j_value_bearing_overstrip_count"] == 3
    assert manifest["promotion_decision"] == "h2m_rejects_current_target_normalization_scope_under_less_direct_wording"

    family_rows = {row["family"]: row for row in payload["family_rows"]}
    assert family_rows["h2m_less_direct_value_bearing_target"]["case_count"] == 4
    assert "result badge Blocked" in family_rows["h2m_less_direct_value_bearing_target"]["expected_target_queries"]
    assert family_rows["h2m_contextual_alias_is_target"]["case_count"] == 2
    assert family_rows["h2m_h2k_regression_guard_less_direct"]["case_count"] == 2

    h2j_non_exact = {
        row["case_id"]: row
        for row in payload["non_exact_rows"]
        if row["profile_label"] == "h2j_target_query_normalization"
    }
    assert h2j_non_exact["h2m_result_badge_blocked_contextual_value"]["expected_target_query"] == (
        "result badge Blocked"
    )
    assert h2j_non_exact["h2m_result_badge_blocked_contextual_value"]["actual_target_query"] == "result badge"
    assert h2j_non_exact["h2m_state_tag_closed_contextual_value"]["expected_target_query"] == "state tag Closed"
    assert h2j_non_exact["h2m_state_tag_closed_contextual_value"]["actual_target_query"] == "state tag"
    assert h2j_non_exact["h2m_priority_badge_critical_contextual_value"]["expected_target_query"] == (
        "priority badge Critical"
    )
    assert h2j_non_exact["h2m_priority_badge_critical_contextual_value"]["actual_target_query"] == (
        "priority badge"
    )
    assert h2j_non_exact["h2m_result_tile_contextual_alias"]["expected_target_query"] == "result tile"
    assert h2j_non_exact["h2m_result_tile_contextual_alias"]["actual_target_query"] == "Blocked"

    overstrip = {row["case_id"]: row for row in payload["overstrip_rows"]}
    assert set(overstrip) == {
        "h2m_result_badge_blocked_contextual_value",
        "h2m_state_tag_closed_contextual_value",
        "h2m_priority_badge_critical_contextual_value",
    }
    assert overstrip["h2m_result_badge_blocked_contextual_value"]["from_arguments"] == (
        '{"image_id":"img-h2m-result-badge-blocked","target_query":"result chip"}'
    )
    assert overstrip["h2m_result_badge_blocked_contextual_value"]["to_arguments"] == (
        '{"image_id":"img-h2m-result-badge-blocked","target_query":"result badge"}'
    )
    assert overstrip["h2m_state_tag_closed_contextual_value"]["from_arguments"] == (
        '{"image_id":"img-h2m-state-tag-closed","target_query":"Closed state tag"}'
    )
    assert overstrip["h2m_state_tag_closed_contextual_value"]["to_arguments"] == (
        '{"image_id":"img-h2m-state-tag-closed","target_query":"state tag"}'
    )

    interventions = {
        (
            row["profile_label"],
            row["case_id"],
            row["intervention_kind"],
            row["prompt_state_label"],
            row["from_arguments"],
            row["to_arguments"],
        )
        for row in payload["intervention_rows"]
    }
    assert (
        "h2j_target_query_normalization",
        "h2m_error_notice_contextual_alias",
        "visual_target_query_normalization",
        "error notice",
        '{"image_id":"img-h2m-error-notice","target_query":"archive panel"}',
        '{"image_id":"img-h2m-error-notice","target_query":"error notice"}',
    ) in interventions
    assert (
        "h2j_target_query_normalization",
        "h2m_mode_field_contextual_regression_guard",
        "visual_target_query_normalization",
        "mode field",
        '{"image_id":"img-h2m-mode-field-short","target_query":"mode switch"}',
        '{"image_id":"img-h2m-mode-field-short","target_query":"mode field"}',
    ) in interventions
    assert all(row["intervention_kind"] != "visual_stale_selection_gate" for row in payload["intervention_rows"])

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2m breaks the H2l saturation" in findings["h2m_breaks_h2l_saturation"]
    assert "does not improve executor-equivalence" in findings["h2m_target_normalization_is_mixed"]
    assert "3 of them over-strip" in findings["h2m_exposes_overstrip"]
    assert "H2n move should make target-query normalization conditional" in findings[
        "next_gate_should_scope_normalization"
    ]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2m_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2m_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2m_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2m_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2m_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2m_overstrip_rows.csv").exists()
    assert (tmp_path / "tables" / "h2m_findings.csv").exists()
    assert (tmp_path / "figures" / "h2m_less_direct_overreach_gate.svg").exists()
