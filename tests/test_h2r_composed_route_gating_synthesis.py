from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2Q_MODULE_PATH = ROOT / "scripts" / "build_h2q_composed_surface_value_stale_synthesis.py"
H2R_MODULE_PATH = ROOT / "scripts" / "build_h2r_composed_route_gating_synthesis.py"

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

H2R_SPEC = importlib.util.spec_from_file_location("build_h2r_composed_route_gating_synthesis", H2R_MODULE_PATH)
assert H2R_SPEC and H2R_SPEC.loader
H2R_SCRIPT = importlib.util.module_from_spec(H2R_SPEC)
sys.modules[H2R_SPEC.name] = H2R_SCRIPT
H2R_SPEC.loader.exec_module(H2R_SCRIPT)


def test_h2r_composed_route_gating_synthesis_marks_local_h2q_repair(tmp_path: Path) -> None:
    payload = H2R_SCRIPT.build_h2r_composed_route_gating_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 5
    assert manifest["comparison_count"] == 1
    assert manifest["family_row_count"] == 20
    assert manifest["h2q_h2p_exact_success_count"] == 3
    assert manifest["h2q_h2p_executor_success_count"] == 3
    assert manifest["h2q_h2r_exact_success_count"] == 8
    assert manifest["h2q_h2r_executor_success_count"] == 8
    assert manifest["h2q_h2r_delta_exact_vs_h2p"] == 0.625
    assert manifest["h2q_h2r_delta_executor_vs_h2p"] == 0.625
    assert manifest["h2q_h2p_non_exact_count"] == 5
    assert manifest["h2q_h2r_non_exact_count"] == 0
    assert manifest["h2q_h2r_composed_route_gating_count"] == 5
    assert manifest["h2q_h2r_target_query_normalization_count"] == 3
    assert manifest["h2q_h2r_contextual_surface_alias_routing_count"] == 1
    assert manifest["h2q_h2r_value_bearing_synthesis_count"] == 2
    assert manifest["h2q_h2r_stale_selection_gate_count"] == 1
    assert manifest["promotion_decision"] == "h2r_solves_h2q_locally_transfer_backtested_requires_h2s"

    h2r_family_rows = {
        row["family"]: row for row in payload["family_rows"] if row["profile_label"] == "h2q_h2r_composed_route_gating"
    }
    assert h2r_family_rows["h2q_surface_alias_value_decoy"]["exact_success_count"] == 2
    assert h2r_family_rows["h2q_value_bearing_stale_decoy"]["exact_success_count"] == 2
    assert h2r_family_rows["h2q_contextual_alias_decoy_overlap"]["exact_success_count"] == 2
    assert h2r_family_rows["h2q_stale_surface_alias"]["exact_success_count"] == 2

    h2r_composed_rows = [
        row
        for row in payload["intervention_rows"]
        if row["profile_label"] == "h2q_h2r_composed_route_gating"
        and row["intervention_kind"] == "visual_composed_route_gating"
    ]
    assert {row["case_id"] for row in h2r_composed_rows} == {
        "h2q_result_tile_blocked_value_badge_decoy",
        "h2q_archive_panel_error_notice_banner_decoy",
        "h2q_mode_field_manual_switch_decoy",
        "h2q_result_tile_stale_selection_hint",
        "h2q_state_panel_stale_selection_hint",
    }
    assert sum(1 for row in h2r_composed_rows if row["reason"] == "stale_selection_to_requested_surface") == 2
    assert sum(1 for row in h2r_composed_rows if row["reason"] == "requested_surface_over_deprioritized_decoy") == 3

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "8/8 strict" in findings["h2r_solves_h2q_local_boundary"]
    assert "5 composed-route interventions" in findings["h2r_matches_h2q_failure_cardinality"]
    assert "route-selection problem" in findings["h2r_mechanism_splits_stale_selection_and_decoy_surface_routes"]
    assert "fresh H2s composed holdout" in findings["h2r_transfer_backtested_but_needs_fresh_h2s"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2r_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2r_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2r_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2r_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2r_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2r_findings.csv").exists()
    assert (tmp_path / "figures" / "h2r_composed_route_gating_gate.svg").exists()
