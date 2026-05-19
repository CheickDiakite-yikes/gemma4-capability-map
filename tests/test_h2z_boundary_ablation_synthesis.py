from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2X_MODULE_PATH = ROOT / "scripts" / "build_h2x_cli_semantic_pressure_synthesis.py"
H2Z_MODULE_PATH = ROOT / "scripts" / "build_h2z_boundary_ablation_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
_load_module("build_h2x_cli_semantic_pressure_synthesis", H2X_MODULE_PATH)
H2Z_SCRIPT = _load_module("build_h2z_boundary_ablation_synthesis", H2Z_MODULE_PATH)


def test_h2z_boundary_ablation_synthesis_finds_additive_closure(tmp_path: Path) -> None:
    payload = H2Z_SCRIPT.build_h2z_boundary_ablation_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 4
    assert manifest["comparison_count"] == 5
    assert manifest["family_row_count"] == 16
    assert manifest["h2y_case_count"] == 16
    assert manifest["h2w_exact_success_count"] == 12
    assert manifest["h2w_executor_success_count"] == 12
    assert manifest["h2z_stale_exact_success_count"] == 15
    assert manifest["h2z_stale_executor_success_count"] == 15
    assert manifest["h2z_component_exact_success_count"] == 13
    assert manifest["h2z_component_executor_success_count"] == 13
    assert manifest["h2z_combined_exact_success_count"] == 16
    assert manifest["h2z_combined_executor_success_count"] == 16
    assert manifest["h2z_stale_delta_exact_vs_h2w"] == 0.1875
    assert manifest["h2z_component_delta_exact_vs_h2w"] == 0.0625
    assert manifest["h2z_combined_delta_exact_vs_h2w"] == 0.25
    assert manifest["h2z_stale_fixed_case_count_vs_h2w"] == 3
    assert manifest["h2z_component_fixed_case_count_vs_h2w"] == 1
    assert manifest["h2z_combined_fixed_case_count_vs_h2w"] == 4
    assert manifest["h2z_combined_fixed_case_count_vs_stale"] == 1
    assert manifest["h2z_combined_fixed_case_count_vs_component"] == 3
    assert manifest["h2z_stale_negation_intervention_count"] == 3
    assert manifest["h2z_component_preservation_intervention_count"] == 1
    assert manifest["h2z_combined_stale_negation_intervention_count"] == 3
    assert manifest["h2z_combined_component_preservation_intervention_count"] == 1
    assert manifest["h2z_combined_non_exact_count"] == 0
    assert manifest["additive_boundary_closure_holds"] is True
    assert manifest["promotion_decision"] == (
        "h2z_closes_h2y_boundary_but_requires_harder_holdout_before_global_promotion"
    )

    stale_fixed = {
        row["case_id"] for row in payload["fixed_case_rows"] if row["comparison_label"] == "h2z_stale_vs_h2w"
    }
    assert stale_fixed == {
        "h2y_escalation_lane_stale_selection_not_lane",
        "h2y_exception_panel_stale_selection_not_panel",
        "h2y_approval_field_stale_selection_not_field",
    }
    component_fixed = {
        row["case_id"] for row in payload["fixed_case_rows"] if row["comparison_label"] == "h2z_component_vs_h2w"
    }
    assert component_fixed == {"h2y_not_active_alert_banner_value_before_component"}
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "16/16 strict" in findings["h2z_closes_h2y_scaled_cli_boundary"]
    assert "Stale-selection negation alone reaches 15/16" in findings["h2z_factorial_split_is_additive"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2z_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2z_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2z_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2z_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2z_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2z_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h2z_findings.csv").exists()
    assert (tmp_path / "figures" / "h2z_boundary_ablation_gate.svg").exists()
