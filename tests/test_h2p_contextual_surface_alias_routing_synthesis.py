from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2P_MODULE_PATH = ROOT / "scripts" / "build_h2p_contextual_surface_alias_routing_synthesis.py"

H2L_SPEC = importlib.util.spec_from_file_location("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
assert H2L_SPEC and H2L_SPEC.loader
H2L_SCRIPT = importlib.util.module_from_spec(H2L_SPEC)
sys.modules[H2L_SPEC.name] = H2L_SCRIPT
H2L_SPEC.loader.exec_module(H2L_SCRIPT)

H2P_SPEC = importlib.util.spec_from_file_location("build_h2p_contextual_surface_alias_routing_synthesis", H2P_MODULE_PATH)
assert H2P_SPEC and H2P_SPEC.loader
H2P_SCRIPT = importlib.util.module_from_spec(H2P_SPEC)
sys.modules[H2P_SPEC.name] = H2P_SCRIPT
H2P_SPEC.loader.exec_module(H2P_SCRIPT)


def test_h2p_contextual_surface_alias_routing_saturates_h2m_with_transfer_preserved(tmp_path: Path) -> None:
    payload = H2P_SCRIPT.build_h2p_contextual_surface_alias_routing_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 11
    assert manifest["comparison_count"] == 7
    assert manifest["h2m_h2e_exact_success_count"] == 1
    assert manifest["h2m_h2e_executor_success_count"] == 3
    assert manifest["h2m_h2j_exact_success_count"] == 3
    assert manifest["h2m_h2j_executor_success_count"] == 3
    assert manifest["h2m_h2n_exact_success_count"] == 3
    assert manifest["h2m_h2n_executor_success_count"] == 5
    assert manifest["h2m_h2o_exact_success_count"] == 7
    assert manifest["h2m_h2o_executor_success_count"] == 7
    assert manifest["h2m_h2p_exact_success_count"] == 8
    assert manifest["h2m_h2p_executor_success_count"] == 8
    assert manifest["h2m_h2p_delta_exact_vs_h2o"] == 0.125
    assert manifest["h2m_h2p_delta_executor_vs_h2o"] == 0.125
    assert manifest["h2m_h2p_delta_exact_vs_h2n"] == 0.625
    assert manifest["h2m_h2p_delta_executor_vs_h2n"] == 0.375
    assert manifest["h2m_h2p_delta_exact_vs_h2j"] == 0.625
    assert manifest["h2m_h2p_delta_executor_vs_h2j"] == 0.625
    assert manifest["h2m_h2p_delta_exact_vs_h2e"] == 0.875
    assert manifest["h2m_h2p_delta_executor_vs_h2e"] == 0.625
    assert manifest["h2k_h2p_exact_success_count"] == 8
    assert manifest["h2k_h2p_executor_success_count"] == 8
    assert manifest["h2l_h2p_exact_success_count"] == 8
    assert manifest["h2l_h2p_executor_success_count"] == 8
    assert manifest["h2f_h2p_exact_success_count"] == 10
    assert manifest["h2f_h2p_executor_success_count"] == 10
    assert manifest["h2k_h2p_delta_exact_vs_h2o"] == 0.0
    assert manifest["h2l_h2p_delta_exact_vs_h2o"] == 0.0
    assert manifest["h2f_h2p_delta_exact_vs_h2o"] == 0.0
    assert manifest["h2m_contextual_surface_alias_routing_count"] == 1
    assert manifest["h2m_value_bearing_synthesis_count"] == 4
    assert manifest["h2m_target_query_normalization_count"] == 2
    assert manifest["h2m_stale_selection_count"] == 0
    assert manifest["h2m_non_exact_count"] == 0
    assert manifest["h2m_remaining_non_exact_case_id"] == ""
    assert manifest["promotion_decision"] == (
        "h2p_contextual_surface_alias_routing_saturates_h2m_with_transfer_preserved"
    )

    h2m_non_exact = [
        row for row in payload["non_exact_rows"] if row["profile_label"] == "h2m_h2p_contextual_surface_alias_routing"
    ]
    assert h2m_non_exact == []

    alias_rows = {
        row["case_id"]: row
        for row in payload["alias_rows"]
        if row["profile_label"] == "h2m_h2p_contextual_surface_alias_routing"
    }
    assert set(alias_rows) == {"h2m_result_tile_contextual_alias"}
    alias = alias_rows["h2m_result_tile_contextual_alias"]
    assert alias["from_arguments"] == '{"image_id":"img-h2m-result-tile","target_query":"Blocked"}'
    assert alias["to_arguments"] == '{"image_id":"img-h2m-result-tile","target_query":"result tile"}'
    assert alias["display_value"] == "Blocked"
    assert alias["surface_label"] == "result tile"
    assert alias["surface_text"] == "Blocked"
    assert alias["surface_region_id"] == "h2m-result-tile-12052"
    assert alias["reason"] == "contextual_surface_alias_recoverable"

    h2m_value = {
        row["case_id"]
        for row in payload["intervention_rows"]
        if row["profile_label"] == "h2m_h2p_contextual_surface_alias_routing"
        and row["intervention_kind"] == "visual_value_bearing_target_query_synthesis"
    }
    assert h2m_value == {
        "h2m_result_badge_blocked_contextual_value",
        "h2m_state_tag_closed_contextual_value",
        "h2m_mode_toggle_manual_contextual_value",
        "h2m_priority_badge_critical_contextual_value",
    }

    h2m_rewrites = {
        row["case_id"]
        for row in payload["intervention_rows"]
        if row["profile_label"] == "h2m_h2p_contextual_surface_alias_routing"
        and row["intervention_kind"] == "visual_target_query_normalization"
    }
    assert h2m_rewrites == {
        "h2m_error_notice_contextual_alias",
        "h2m_mode_field_contextual_regression_guard",
    }

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "from H2o's 7/8 to 8/8" in findings["h2p_saturates_h2m_surface_alias_boundary"]
    assert "1 contextual surface-alias intervention" in findings["h2p_mechanism_is_single_alias_gate"]
    assert "leaves 0 non-exact H2m rows" in findings["h2p_closes_h2m_current_exact_boundary"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2p_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2p_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2p_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2p_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2p_contextual_surface_alias_rows.csv").exists()
    assert (tmp_path / "tables" / "h2p_findings.csv").exists()
    assert (tmp_path / "figures" / "h2p_contextual_surface_alias_routing_gate.svg").exists()
