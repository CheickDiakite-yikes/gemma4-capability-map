from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2X_MODULE_PATH = ROOT / "scripts" / "build_h2x_cli_semantic_pressure_synthesis.py"
H3A_MODULE_PATH = ROOT / "scripts" / "build_h3a_controller_repair_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
_load_module("build_h2x_cli_semantic_pressure_synthesis", H2X_MODULE_PATH)
H3A_SCRIPT = _load_module("build_h3a_controller_repair_synthesis", H3A_MODULE_PATH)


def test_h3a_controller_repair_synthesis_separates_h3_residuals(tmp_path: Path) -> None:
    payload = H3A_SCRIPT.build_h3a_controller_repair_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 4
    assert manifest["comparison_count"] == 5
    assert manifest["family_row_count"] == 20
    assert manifest["h3_case_count"] == 20
    assert manifest["h2z_combined_exact_success_count"] == 15
    assert manifest["h2z_combined_executor_success_count"] == 15
    assert manifest["h3a_stale_exact_success_count"] == 19
    assert manifest["h3a_stale_executor_success_count"] == 19
    assert manifest["h3a_negative_exact_success_count"] == 16
    assert manifest["h3a_negative_executor_success_count"] == 16
    assert manifest["h3a_combined_exact_success_count"] == 20
    assert manifest["h3a_combined_executor_success_count"] == 20
    assert manifest["h3a_stale_delta_exact_vs_h2z"] == 0.19999999999999996
    assert manifest["h3a_stale_delta_executor_vs_h2z"] == 0.19999999999999996
    assert manifest["h3a_negative_delta_exact_vs_h2z"] == 0.050000000000000044
    assert manifest["h3a_negative_delta_executor_vs_h2z"] == 0.050000000000000044
    assert manifest["h3a_combined_delta_exact_vs_h2z"] == 0.25
    assert manifest["h3a_combined_delta_executor_vs_h2z"] == 0.25
    assert manifest["h3a_combined_delta_exact_vs_stale"] == 0.050000000000000044
    assert manifest["h3a_combined_delta_exact_vs_negative"] == 0.19999999999999996
    assert manifest["h3a_stale_fixed_case_count_vs_h2z"] == 4
    assert manifest["h3a_negative_fixed_case_count_vs_h2z"] == 1
    assert manifest["h3a_combined_fixed_case_count_vs_h2z"] == 5
    assert manifest["h3a_combined_fixed_case_count_vs_stale"] == 1
    assert manifest["h3a_combined_fixed_case_count_vs_negative"] == 4
    assert manifest["h3a_stale_paraphrase_intervention_count"] == 4
    assert manifest["h3a_negative_value_intervention_count"] == 1
    assert manifest["h3a_combined_stale_paraphrase_intervention_count"] == 4
    assert manifest["h3a_combined_negative_value_intervention_count"] == 1
    assert manifest["h3a_combined_non_exact_count"] == 0
    assert manifest["h3a_closes_h3_exact_and_executor"] is True
    assert manifest["h3a_factorial_residuals_separate"] is True
    assert manifest["promotion_decision"] == (
        "use_h3a_as_next_candidate_but_require_h2y_h2z_h3_transfer_reruns_before_global_promotion"
    )

    combined_fixed = {
        row["case_id"] for row in payload["fixed_case_rows"] if row["comparison_label"] == "h3a_combined_vs_h2z_combined"
    }
    assert combined_fixed == {
        "h3_finance_renewal_lane_retired_view",
        "h3_finance_forecast_tile_leftover_evidence",
        "h3_research_claim_panel_carryover_selection",
        "h3_research_evidence_badge_retired_selection",
        "h3_support_inactive_alert_banner",
    }
    stale_fixed = {
        row["case_id"] for row in payload["fixed_case_rows"] if row["comparison_label"] == "h3a_stale_vs_h2z_combined"
    }
    assert stale_fixed == {
        "h3_finance_renewal_lane_retired_view",
        "h3_finance_forecast_tile_leftover_evidence",
        "h3_research_claim_panel_carryover_selection",
        "h3_research_evidence_badge_retired_selection",
    }
    negative_fixed = {
        row["case_id"] for row in payload["fixed_case_rows"] if row["comparison_label"] == "h3a_negative_vs_h2z_combined"
    }
    assert negative_fixed == {"h3_support_inactive_alert_banner"}

    h3a_interventions = {row["intervention_kind"] for row in payload["intervention_rows"]}
    assert "visual_stale_selection_paraphrase_guard" in h3a_interventions
    assert "visual_negative_value_component_target_preservation" in h3a_interventions
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "20/20 strict" in findings["h3a_combined_closes_h3"]
    assert "fixes 4 H2z misses" in findings["h3a_stale_paraphrase_guard_is_separable"]
    assert "fixes 1 H2z miss" in findings["h3a_negative_value_guard_is_separable"]
    assert "4 stale-paraphrase interventions" in findings["h3a_interventions_match_repair_surface"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h3a_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h3a_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h3a_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h3a_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h3a_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h3a_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h3a_findings.csv").exists()
    assert (tmp_path / "figures" / "h3a_controller_repair_gate.svg").exists()
