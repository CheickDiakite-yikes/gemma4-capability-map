from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
)
SPEC = importlib.util.spec_from_file_location("build_h2l_target_normalization_overreach_synthesis", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2l_target_normalization_overreach_synthesis_is_discriminative(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2l_target_normalization_overreach_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 3
    assert manifest["comparison_count"] == 2
    assert manifest["family_row_count"] == 3
    assert manifest["h2e_exact_success_count"] == 7
    assert manifest["h2e_executor_success_count"] == 7
    assert manifest["h2j_no_stale_exact_success_count"] == 8
    assert manifest["h2j_no_stale_executor_success_count"] == 8
    assert manifest["h2j_exact_success_count"] == 8
    assert manifest["h2j_executor_success_count"] == 8
    assert manifest["h2j_delta_exact_vs_h2e"] == 0.125
    assert manifest["h2j_delta_executor_vs_h2e"] == 0.125
    assert manifest["h2j_delta_exact_vs_no_stale_gate"] == 0.0
    assert manifest["h2j_delta_executor_vs_no_stale_gate"] == 0.0
    assert manifest["h2e_non_exact_count"] == 1
    assert manifest["h2j_no_stale_non_exact_count"] == 0
    assert manifest["h2j_non_exact_count"] == 0
    assert manifest["target_query_normalization_count"] == 1
    assert manifest["visual_stale_selection_gate_count"] == 0
    assert manifest["h2j_no_stale_target_query_normalization_count"] == 1
    assert manifest["h2j_no_stale_visual_stale_selection_gate_count"] == 0
    assert manifest["promotion_decision"] == "h2l_overreach_holdout_passes_target_normalization"

    family_rows = {row["family"]: row for row in payload["family_rows"]}
    assert family_rows["h2l_value_bearing_target"]["case_count"] == 4
    assert "result badge Blocked" in family_rows["h2l_value_bearing_target"]["expected_target_queries"]
    assert family_rows["h2l_alias_is_target"]["case_count"] == 2
    assert family_rows["h2l_h2k_regression_guard"]["case_count"] == 2

    non_exact = {(row["profile_label"], row["case_id"]): row for row in payload["non_exact_rows"]}
    h2e_miss = non_exact[("h2e_route_arbitration", "h2l_status_badge_short_label_regression_guard")]
    assert h2e_miss["failure_mode"] == "argument_mismatch"
    assert h2e_miss["expected_target_query"] == "status badge"
    assert h2e_miss["actual_target_query"] == "critical chip"
    assert h2e_miss["executor_equivalence_match"] is False
    assert all(row["profile_label"] != "h2j_target_query_normalization" for row in payload["non_exact_rows"])
    assert all(
        row["profile_label"] != "h2j_target_query_normalization_no_stale_gate"
        for row in payload["non_exact_rows"]
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
        "h2l_status_badge_short_label_regression_guard",
        "visual_target_query_normalization",
        "status badge",
        '{"image_id":"img-h2l-status-badge-short","target_query":"critical chip"}',
        '{"image_id":"img-h2l-status-badge-short","target_query":"status badge"}',
    ) in interventions
    assert (
        "h2j_target_query_normalization_no_stale_gate",
        "h2l_status_badge_short_label_regression_guard",
        "visual_target_query_normalization",
        "status badge",
        '{"image_id":"img-h2l-status-badge-short","target_query":"critical chip"}',
        '{"image_id":"img-h2l-status-badge-short","target_query":"status badge"}',
    ) in interventions

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2l does not expose target-query over-normalization" in findings["h2l_overreach_holdout_passed"]
    assert "H2e reaches 7/8 exact" in findings["h2l_repairs_h2e_regression_guard"]
    assert "stale-gate-off ablation records 1 target-query-normalization intervention and 0" in findings[
        "h2l_mechanism_is_target_normalization_not_stale_gate"
    ]
    assert "reduce direct target-is wording" in findings["next_holdout_should_reduce_prompt_directness"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2l_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2l_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2l_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2l_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2l_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2l_findings.csv").exists()
    assert (tmp_path / "figures" / "h2l_target_normalization_overreach_gate.svg").exists()
