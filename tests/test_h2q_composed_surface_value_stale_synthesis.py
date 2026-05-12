from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2Q_MODULE_PATH = ROOT / "scripts" / "build_h2q_composed_surface_value_stale_synthesis.py"

H2L_SPEC = importlib.util.spec_from_file_location("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
assert H2L_SPEC and H2L_SPEC.loader
H2L_SCRIPT = importlib.util.module_from_spec(H2L_SPEC)
sys.modules[H2L_SPEC.name] = H2L_SCRIPT
H2L_SPEC.loader.exec_module(H2L_SCRIPT)

H2Q_SPEC = importlib.util.spec_from_file_location("build_h2q_composed_surface_value_stale_synthesis", H2Q_MODULE_PATH)
assert H2Q_SPEC and H2Q_SPEC.loader
H2Q_SCRIPT = importlib.util.module_from_spec(H2Q_SPEC)
sys.modules[H2Q_SPEC.name] = H2Q_SCRIPT
H2Q_SPEC.loader.exec_module(H2Q_SCRIPT)


def test_h2q_composed_surface_value_stale_synthesis_marks_h2p_boundary(tmp_path: Path) -> None:
    payload = H2Q_SCRIPT.build_h2q_composed_surface_value_stale_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 4
    assert manifest["comparison_count"] == 3
    assert manifest["family_row_count"] == 16
    assert manifest["h2q_h2e_exact_success_count"] == 1
    assert manifest["h2q_h2e_executor_success_count"] == 2
    assert manifest["h2q_h2n_exact_success_count"] == 0
    assert manifest["h2q_h2n_executor_success_count"] == 1
    assert manifest["h2q_h2o_exact_success_count"] == 2
    assert manifest["h2q_h2o_executor_success_count"] == 2
    assert manifest["h2q_h2p_exact_success_count"] == 3
    assert manifest["h2q_h2p_executor_success_count"] == 3
    assert manifest["h2q_h2p_delta_exact_vs_h2o"] == 0.125
    assert manifest["h2q_h2p_delta_executor_vs_h2o"] == 0.125
    assert manifest["h2q_h2p_delta_exact_vs_h2n"] == 0.375
    assert manifest["h2q_h2p_delta_executor_vs_h2n"] == 0.25
    assert manifest["h2q_h2p_delta_exact_vs_h2e"] == 0.25
    assert manifest["h2q_h2p_delta_executor_vs_h2e"] == 0.125
    assert manifest["h2q_h2p_non_exact_count"] == 5
    assert manifest["h2q_h2p_wrong_tool_count"] == 2
    assert manifest["h2q_h2p_argument_mismatch_count"] == 3
    assert manifest["h2q_h2p_contextual_surface_alias_routing_count"] == 1
    assert manifest["h2q_h2p_value_bearing_synthesis_count"] == 2
    assert manifest["h2q_h2p_target_query_normalization_count"] == 3
    assert manifest["h2q_h2p_stale_selection_gate_count"] == 1
    assert manifest["promotion_decision"] == "h2q_breaks_h2p_composed_surface_value_stale_boundary"

    h2p_family_rows = {
        row["family"]: row
        for row in payload["family_rows"]
        if row["profile_label"] == "h2q_h2p_contextual_surface_alias_routing"
    }
    assert h2p_family_rows["h2q_value_bearing_stale_decoy"]["exact_success_count"] == 2
    assert h2p_family_rows["h2q_surface_alias_value_decoy"]["exact_success_count"] == 1
    assert h2p_family_rows["h2q_contextual_alias_decoy_overlap"]["exact_success_count"] == 0
    assert h2p_family_rows["h2q_stale_surface_alias"]["exact_success_count"] == 0

    h2p_non_exact = [
        row for row in payload["non_exact_rows"] if row["profile_label"] == "h2q_h2p_contextual_surface_alias_routing"
    ]
    assert {row["case_id"] for row in h2p_non_exact} == {
        "h2q_result_tile_blocked_value_badge_decoy",
        "h2q_archive_panel_error_notice_banner_decoy",
        "h2q_mode_field_manual_switch_decoy",
        "h2q_result_tile_stale_selection_hint",
        "h2q_state_panel_stale_selection_hint",
    }

    intervention_kinds = {
        row["intervention_kind"]
        for row in payload["intervention_rows"]
        if row["profile_label"] == "h2q_h2p_contextual_surface_alias_routing"
    }
    assert intervention_kinds == {
        "visual_contextual_surface_alias_routing",
        "visual_stale_selection_gate",
        "visual_target_query_normalization",
        "visual_value_bearing_target_query_synthesis",
    }

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "3/8 strict" in findings["h2q_breaks_h2p_saturation"]
    assert "H2p is still the best current row" in findings["h2q_h2p_remains_directionally_best"]
    assert "2 wrong-tool rows" in findings["h2q_failures_are_tool_route_and_decoy_selection_failures"]
    assert "composed route gating" in findings["next_slice_should_target_composed_route_gating"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2q_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2q_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2q_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2q_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2q_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2q_findings.csv").exists()
    assert (tmp_path / "figures" / "h2q_composed_surface_value_stale_gate.svg").exists()
