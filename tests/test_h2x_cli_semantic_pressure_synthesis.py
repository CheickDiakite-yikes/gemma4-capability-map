from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2X_MODULE_PATH = ROOT / "scripts" / "build_h2x_cli_semantic_pressure_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
H2X_SCRIPT = _load_module("build_h2x_cli_semantic_pressure_synthesis", H2X_MODULE_PATH)


def test_h2x_cli_semantic_pressure_synthesis_is_fallback_independent(tmp_path: Path) -> None:
    payload = H2X_SCRIPT.build_h2x_cli_semantic_pressure_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 4
    assert manifest["comparison_count"] == 3
    assert manifest["family_row_count"] == 16
    assert manifest["h2x_case_count"] == 8
    assert manifest["h2u_exact_success_count"] == 3
    assert manifest["h2u_executor_success_count"] == 4
    assert manifest["h2u_no_fallback_exact_success_count"] == 3
    assert manifest["h2u_no_fallback_executor_success_count"] == 4
    assert manifest["h2w_exact_success_count"] == 8
    assert manifest["h2w_executor_success_count"] == 8
    assert manifest["h2w_no_fallback_exact_success_count"] == 8
    assert manifest["h2w_no_fallback_executor_success_count"] == 8
    assert manifest["h2w_delta_exact_vs_h2u"] == 0.625
    assert manifest["h2w_delta_executor_vs_h2u"] == 0.5
    assert manifest["h2u_fallback_delta_exact"] == 0.0
    assert manifest["h2u_fallback_delta_executor"] == 0.0
    assert manifest["h2w_fallback_delta_exact"] == 0.0
    assert manifest["h2w_fallback_delta_executor"] == 0.0
    assert manifest["h2w_non_exact_count"] == 0
    assert manifest["h2w_fixed_case_count_vs_h2u"] == 5
    assert manifest["h2w_semantic_target_preservation_count"] == 1
    assert manifest["h2w_target_query_normalization_count"] == 5
    assert manifest["h2w_composed_route_gating_count"] == 1
    assert manifest["h2w_composed_route_gating_blocked_count"] == 1
    assert manifest["h2w_no_fallback_semantic_target_preservation_count"] == 1
    assert manifest["h2w_no_fallback_target_query_normalization_count"] == 5
    assert manifest["fallback_independence_holds"] is True
    assert manifest["promotion_decision"] == "h2x_promotes_semantic_target_preservation_to_packaged_cli_gate"

    fixed_cases = {
        row["case_id"]
        for row in payload["fixed_case_rows"]
        if row["comparison_label"] == "h2x_h2w_vs_h2u"
    }
    assert fixed_cases == {
        "h2x_risk_lane_stale_selection_not_lane",
        "h2x_not_ready_status_badge_value_before_component",
        "h2x_not_applicable_reason_chip_value_before_component",
        "h2x_not_approved_approval_toggle_value_before_component",
        "h2x_not_blocked_result_tile_value_before_component",
    }

    h2w_interventions = [
        row for row in payload["intervention_rows"] if row["profile_label"] == "h2x_h2w_semantic_target_preservation"
    ]
    intervention_kinds = {row["intervention_kind"] for row in h2w_interventions}
    assert "visual_semantic_target_preservation" in intervention_kinds
    assert "visual_target_query_normalization" in intervention_kinds
    assert "visual_composed_route_gating" in intervention_kinds
    assert "visual_composed_route_gating_blocked" in intervention_kinds
    assert any(
        row["case_id"] == "h2x_status_badge_quoted_not_badge_note"
        and row["reason"] == "semantic_label_preserved_over_stale_context"
        for row in h2w_interventions
    )
    assert any(
        row["case_id"] == "h2x_risk_lane_stale_selection_not_lane"
        and row["intervention_kind"] == "visual_composed_route_gating"
        and row["requested_label"] == "risk lane"
        and row["requested_region_id"] == "h2x-risk-lane-17022"
        for row in h2w_interventions
    )

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "drops H2u to 3/8" in findings["h2x_breaks_h2u_topline_saturation"]
    assert "not the causal helper" in findings["semantic_preservation_is_causal_not_fallback"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2x_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2x_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2x_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2x_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2x_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2x_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h2x_findings.csv").exists()
    assert (tmp_path / "figures" / "h2x_cli_semantic_pressure_gate.svg").exists()
