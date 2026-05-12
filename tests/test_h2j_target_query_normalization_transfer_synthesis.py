from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_h2j_target_query_normalization_transfer_synthesis.py"
)
SPEC = importlib.util.spec_from_file_location("build_h2j_target_query_normalization_transfer_synthesis", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2j_target_query_normalization_transfer_synthesis_promotes_harder_holdout(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2j_target_query_normalization_transfer_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 9
    assert manifest["comparison_count"] == 7
    assert manifest["h2j_h2f_exact_success_count"] == 10
    assert manifest["h2j_h2f_executor_success_count"] == 10
    assert manifest["h2j_h2b_exact_success_count"] == 5
    assert manifest["h2j_h2b_executor_success_count"] == 5
    assert manifest["h2j_h1x_exact_success_count"] == 8
    assert manifest["h2j_h1x_executor_success_count"] == 8
    assert manifest["h2j_delta_exact_vs_h2e_on_h2f"] == 0.4
    assert manifest["h2j_delta_exact_vs_h2h_on_h2f"] == 0.09999999999999998
    assert manifest["h2j_delta_exact_vs_h2e_on_h2b"] == 0.0
    assert manifest["h2j_delta_exact_vs_h2h_on_h2b"] == 0.4
    assert manifest["h2j_delta_exact_vs_h2e_on_h1x"] == 0.0
    assert manifest["h2j_delta_exact_vs_h2h_on_h1x"] == 0.25
    assert manifest["h2j_non_exact_count"] == 0
    assert manifest["target_query_normalization_count"] == 4
    assert manifest["visual_stale_selection_gate_count"] == 4
    assert manifest["promotion_decision"] == "promote_h2j_to_next_harder_holdout_not_global_default"

    packet_rows = {(row["suite"], row["profile_label"]): row for row in payload["packet_rows"]}
    assert packet_rows[("h2f", "h2j_target_query_normalization")]["exact_success_count"] == 10
    assert packet_rows[("h2b", "h2j_target_query_normalization")]["exact_success_count"] == 5
    assert packet_rows[("h1x", "h2j_target_query_normalization")]["exact_success_count"] == 8

    assert payload["h2j_non_exact_rows"] == []
    interventions = {(row["suite"], row["case_id"], row["intervention_kind"]) for row in payload["intervention_rows"]}
    assert ("h2f", "h2f_result_tile_comment_value_decoy", "visual_target_query_normalization") in interventions
    assert ("h2f", "h2f_state_marker_history_value_decoy", "visual_target_query_normalization") in interventions
    assert ("h2b", "component_value_result_pill_log_decoy", "visual_stale_selection_gate") in interventions
    assert ("h1x", "h1x_responsible_party_field_old_owner_memo_decoy", "visual_stale_selection_gate") in interventions

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2j reaches 10/10 strict" in findings["h2j_closes_h2f"]
    assert "5/5 on H2b and 8/8 on H1x" in findings["h2j_preserves_transfer_gates"]
    assert "4 target-query-normalization interventions" in findings["h2j_controller_mechanism"]
    assert "promotion to a harder holdout" in findings["h2j_remaining_risk"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2j_transfer_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2j_transfer_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2j_transfer_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2j_transfer_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2j_transfer_findings.csv").exists()
    assert (tmp_path / "figures" / "h2j_transfer_gate.svg").exists()
