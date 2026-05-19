from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2X_MODULE_PATH = ROOT / "scripts" / "build_h2x_cli_semantic_pressure_synthesis.py"
H3A_H2Y_MODULE_PATH = ROOT / "scripts" / "build_h3a_h2y_transfer_gate_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
_load_module("build_h2x_cli_semantic_pressure_synthesis", H2X_MODULE_PATH)
H3A_H2Y_SCRIPT = _load_module("build_h3a_h2y_transfer_gate_synthesis", H3A_H2Y_MODULE_PATH)


def test_h3a_h2y_transfer_gate_synthesis_preserves_h2z_boundary(tmp_path: Path) -> None:
    payload = H3A_H2Y_SCRIPT.build_h3a_h2y_transfer_gate_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 3
    assert manifest["comparison_count"] == 2
    assert manifest["family_row_count"] == 12
    assert manifest["h2y_case_count"] == 16
    assert manifest["h2w_exact_success_count"] == 12
    assert manifest["h2w_executor_success_count"] == 12
    assert manifest["h2z_exact_success_count"] == 16
    assert manifest["h2z_executor_success_count"] == 16
    assert manifest["h3a_exact_success_count"] == 16
    assert manifest["h3a_executor_success_count"] == 16
    assert manifest["h3a_delta_exact_vs_h2z"] == 0.0
    assert manifest["h3a_delta_executor_vs_h2z"] == 0.0
    assert manifest["h3a_delta_exact_vs_h2w"] == 0.25
    assert manifest["h3a_delta_executor_vs_h2w"] == 0.25
    assert manifest["h3a_fixed_case_count_vs_h2z"] == 0
    assert manifest["h3a_fixed_case_count_vs_h2w"] == 4
    assert manifest["h3a_stale_negation_intervention_count"] == 3
    assert manifest["h3a_negated_component_intervention_count"] == 1
    assert manifest["h3a_stale_paraphrase_intervention_count"] == 0
    assert manifest["h3a_negative_value_intervention_count"] == 0
    assert manifest["h3a_non_exact_count"] == 0
    assert manifest["h3a_preserves_h2z_h2y_closure"] is True
    assert manifest["h3a_new_helpers_do_not_overtrigger_on_h2y"] is True
    assert manifest["promotion_decision"] == (
        "h3a_passes_first_h2y_transfer_gate_but_needs_broader_backcompat_before_global_promotion"
    )

    fixed_vs_h2w = {
        row["case_id"] for row in payload["fixed_case_rows"] if row["comparison_label"] == "h2y_h3a_vs_h2w"
    }
    assert fixed_vs_h2w == {
        "h2y_escalation_lane_stale_selection_not_lane",
        "h2y_exception_panel_stale_selection_not_panel",
        "h2y_approval_field_stale_selection_not_field",
        "h2y_not_active_alert_banner_value_before_component",
    }
    assert [row for row in payload["fixed_case_rows"] if row["comparison_label"] == "h2y_h3a_vs_h2z"] == []
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "0.0 exact-rate delta" in findings["h3a_preserves_h2z_h2y_closure"]
    assert "H2w is 12/16" in findings["h3a_retains_h2w_delta_on_h2y"]
    assert "4 H2w misses" in findings["h3a_h2y_fixed_cases_match_h2z_boundary"]
    assert "3 stale-selection negation interventions" in findings["h3a_h2y_uses_original_h2z_helpers"]
    assert "0 stale-paraphrase interventions" in findings["h3a_new_helpers_do_not_overtrigger_on_h2y"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h3a_h2y_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h3a_h2y_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h3a_h2y_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h3a_h2y_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h3a_h2y_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h3a_h2y_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h3a_h2y_findings.csv").exists()
    assert (tmp_path / "figures" / "h3a_h2y_transfer_gate.svg").exists()
