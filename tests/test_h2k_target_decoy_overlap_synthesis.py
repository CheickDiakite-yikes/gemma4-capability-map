from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2k_target_decoy_overlap_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2k_target_decoy_overlap_synthesis", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2k_target_decoy_overlap_synthesis_is_discriminative(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2k_target_decoy_overlap_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 4
    assert manifest["comparison_count"] == 3
    assert manifest["h2e_exact_success_count"] == 3
    assert manifest["h2e_executor_success_count"] == 6
    assert manifest["h2h_exact_success_count"] == 6
    assert manifest["h2h_executor_success_count"] == 6
    assert manifest["h2j_no_stale_exact_success_count"] == 8
    assert manifest["h2j_no_stale_executor_success_count"] == 8
    assert manifest["h2j_exact_success_count"] == 8
    assert manifest["h2j_executor_success_count"] == 8
    assert manifest["h2j_delta_exact_vs_h2e"] == 0.625
    assert manifest["h2j_delta_executor_vs_h2e"] == 0.25
    assert manifest["h2j_delta_exact_vs_h2h"] == 0.25
    assert manifest["h2j_delta_executor_vs_h2h"] == 0.25
    assert manifest["h2j_delta_exact_vs_no_stale_gate"] == 0.0
    assert manifest["h2j_delta_executor_vs_no_stale_gate"] == 0.0
    assert manifest["h2e_non_exact_count"] == 5
    assert manifest["h2h_non_exact_count"] == 2
    assert manifest["h2j_no_stale_non_exact_count"] == 0
    assert manifest["h2j_non_exact_count"] == 0
    assert manifest["target_query_normalization_count"] == 5
    assert manifest["visual_stale_selection_gate_count"] == 0
    assert manifest["h2j_no_stale_target_query_normalization_count"] == 5
    assert manifest["h2j_no_stale_visual_stale_selection_gate_count"] == 0
    assert manifest["promotion_decision"] == "h2k_supports_target_query_normalization_not_stale_selection_gate"

    non_exact = {(row["profile_label"], row["case_id"]): row for row in payload["non_exact_rows"]}
    assert non_exact[
        ("h2e_route_arbitration", "h2k_error_banner_archived_error_notice_decoy")
    ]["actual_target_query"] == "error notice"
    assert non_exact[
        ("h2h_component_identity_negative_examples", "h2k_state_tag_before_reading_state_marker_decoy")
    ]["actual_target_query"] == "state marker Closed"
    assert all(row["profile_label"] != "h2j_target_query_normalization" for row in payload["non_exact_rows"])
    assert all(
        row["profile_label"] != "h2j_target_query_normalization_no_stale_gate"
        for row in payload["non_exact_rows"]
    )

    interventions = {
        (row["profile_label"], row["case_id"], row["prompt_state_label"]) for row in payload["intervention_rows"]
    }
    assert (
        "h2j_target_query_normalization",
        "h2k_error_banner_archived_error_notice_decoy",
        "error banner",
    ) in interventions
    assert (
        "h2j_target_query_normalization",
        "h2k_mode_field_before_reading_mode_switch_decoy",
        "mode field",
    ) in interventions
    assert (
        "h2j_target_query_normalization_no_stale_gate",
        "h2k_result_badge_negated_result_tile_decoy",
        "result badge",
    ) in interventions
    assert (
        "h2j_target_query_normalization_no_stale_gate",
        "h2k_state_tag_before_reading_state_marker_decoy",
        "state tag",
    ) in interventions

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2e reaches 3/8 exact" in findings["h2k_is_discriminative"]
    assert "H2j improves over H2e by 0.625" in findings["h2j_passes_target_decoy_overlap"]
    assert "stale-gate-off ablation records 5 target-query-normalization interventions and 0" in findings[
        "h2j_mechanism_is_target_normalization"
    ]
    assert "no-target-normalizer ablation" in findings["next_transfer_required"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2k_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2k_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2k_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2k_h2j_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2k_findings.csv").exists()
    assert (tmp_path / "figures" / "h2k_target_decoy_overlap_gate.svg").exists()
