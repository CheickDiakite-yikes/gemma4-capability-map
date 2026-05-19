from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2X_MODULE_PATH = ROOT / "scripts" / "build_h2x_cli_semantic_pressure_synthesis.py"
H3_MODULE_PATH = ROOT / "scripts" / "build_h3_cli_controller_holdout_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
_load_module("build_h2x_cli_semantic_pressure_synthesis", H2X_MODULE_PATH)
H3_SCRIPT = _load_module("build_h3_cli_controller_holdout_synthesis", H3_MODULE_PATH)


def test_h3_cli_controller_holdout_synthesis_blocks_h2z_global_promotion(tmp_path: Path) -> None:
    payload = H3_SCRIPT.build_h3_cli_controller_holdout_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 4
    assert manifest["comparison_count"] == 5
    assert manifest["family_row_count"] == 20
    assert manifest["h3_case_count"] == 20
    assert manifest["h2w_exact_success_count"] == 15
    assert manifest["h2w_executor_success_count"] == 15
    assert manifest["h2z_stale_exact_success_count"] == 15
    assert manifest["h2z_stale_executor_success_count"] == 15
    assert manifest["h2z_component_exact_success_count"] == 15
    assert manifest["h2z_component_executor_success_count"] == 15
    assert manifest["h2z_combined_exact_success_count"] == 15
    assert manifest["h2z_combined_executor_success_count"] == 15
    assert manifest["h2z_stale_delta_exact_vs_h2w"] == 0.0
    assert manifest["h2z_component_delta_exact_vs_h2w"] == 0.0
    assert manifest["h2z_combined_delta_exact_vs_h2w"] == 0.0
    assert manifest["h2z_stale_delta_executor_vs_h2w"] == 0.0
    assert manifest["h2z_component_delta_executor_vs_h2w"] == 0.0
    assert manifest["h2z_combined_delta_executor_vs_h2w"] == 0.0
    assert manifest["h2z_combined_delta_exact_vs_stale"] == 0.0
    assert manifest["h2z_combined_delta_exact_vs_component"] == 0.0
    assert manifest["h2z_stale_fixed_case_count_vs_h2w"] == 0
    assert manifest["h2z_component_fixed_case_count_vs_h2w"] == 0
    assert manifest["h2z_combined_fixed_case_count_vs_h2w"] == 0
    assert manifest["h2z_combined_fixed_case_count_vs_stale"] == 0
    assert manifest["h2z_combined_fixed_case_count_vs_component"] == 0
    assert manifest["h2z_stale_negation_intervention_count"] == 0
    assert manifest["h2z_component_preservation_intervention_count"] == 0
    assert manifest["h2z_combined_stale_negation_intervention_count"] == 0
    assert manifest["h2z_combined_component_preservation_intervention_count"] == 0
    assert manifest["h2z_combined_non_exact_count"] == 5
    assert manifest["h3_breaks_h2z_known_boundary_closure"] is True
    assert manifest["promotion_decision"] == (
        "do_not_promote_h2z_globally_until_h3_stale_paraphrase_and_negative_value_syntax_are_repaired"
    )

    combined_non_exact = {
        row["case_id"] for row in payload["non_exact_rows"] if row["profile_label"] == "h3_h2z_boundary_combined"
    }
    assert combined_non_exact == {
        "h3_finance_renewal_lane_retired_view",
        "h3_finance_forecast_tile_leftover_evidence",
        "h3_research_claim_panel_carryover_selection",
        "h3_research_evidence_badge_retired_selection",
        "h3_support_inactive_alert_banner",
    }
    assert payload["fixed_case_rows"] == []
    h2z_specific_interventions = {
        row["intervention_kind"]
        for row in payload["intervention_rows"]
        if row["intervention_kind"]
        in {
            "visual_stale_selection_negation_guard",
            "visual_negated_component_target_preservation",
        }
    }
    assert h2z_specific_interventions == set()
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "15/20 strict" in findings["h3_breaks_h2z_global_promotion"]
    assert "fixed cases vs H2w: 0" in findings["h3_zero_delta_across_h2z_profiles"]
    assert "four stale-selection paraphrase rows" in findings["h3_residual_mechanism_classes"]
    assert "0 stale-selection negation interventions" in findings["h3_current_helpers_do_not_trigger_on_new_language"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h3_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h3_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h3_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h3_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h3_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h3_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h3_findings.csv").exists()
    assert (tmp_path / "figures" / "h3_cli_controller_holdout_gate.svg").exists()
