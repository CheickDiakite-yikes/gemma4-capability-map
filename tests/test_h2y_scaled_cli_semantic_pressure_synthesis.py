from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2X_MODULE_PATH = ROOT / "scripts" / "build_h2x_cli_semantic_pressure_synthesis.py"
H2Y_MODULE_PATH = ROOT / "scripts" / "build_h2y_scaled_cli_semantic_pressure_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
_load_module("build_h2x_cli_semantic_pressure_synthesis", H2X_MODULE_PATH)
H2Y_SCRIPT = _load_module("build_h2y_scaled_cli_semantic_pressure_synthesis", H2Y_MODULE_PATH)


def test_h2y_scaled_cli_semantic_pressure_synthesis_finds_boundary(tmp_path: Path) -> None:
    payload = H2Y_SCRIPT.build_h2y_scaled_cli_semantic_pressure_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 4
    assert manifest["comparison_count"] == 3
    assert manifest["family_row_count"] == 16
    assert manifest["h2y_case_count"] == 16
    assert manifest["h2u_exact_success_count"] == 4
    assert manifest["h2u_executor_success_count"] == 5
    assert manifest["h2u_no_fallback_exact_success_count"] == 4
    assert manifest["h2u_no_fallback_executor_success_count"] == 5
    assert manifest["h2w_exact_success_count"] == 12
    assert manifest["h2w_executor_success_count"] == 12
    assert manifest["h2w_no_fallback_exact_success_count"] == 12
    assert manifest["h2w_no_fallback_executor_success_count"] == 12
    assert manifest["h2w_delta_exact_vs_h2u"] == 0.5
    assert manifest["h2w_delta_executor_vs_h2u"] == 0.4375
    assert manifest["h2u_fallback_delta_exact"] == 0.0
    assert manifest["h2u_fallback_delta_executor"] == 0.0
    assert manifest["h2w_fallback_delta_exact"] == 0.0
    assert manifest["h2w_fallback_delta_executor"] == 0.0
    assert manifest["h2w_non_exact_count"] == 4
    assert manifest["h2w_fixed_case_count_vs_h2u"] == 8
    assert manifest["h2w_semantic_target_preservation_count"] == 2
    assert manifest["h2w_target_query_normalization_count"] == 7
    assert manifest["h2w_no_fallback_semantic_target_preservation_count"] == 2
    assert manifest["h2w_no_fallback_target_query_normalization_count"] == 7
    assert manifest["h2w_unresolved_boundary_count"] == 4
    assert manifest["h2w_unresolved_stale_selection_count"] == 3
    assert manifest["fallback_independence_holds"] is True
    assert manifest["promotion_decision"] == "h2y_confirms_semantic_preservation_gain_but_blocks_global_promotion"

    fixed_cases = {
        row["case_id"]
        for row in payload["fixed_case_rows"]
        if row["comparison_label"] == "h2y_h2w_vs_h2u"
    }
    assert fixed_cases == {
        "h2y_action_banner_quoted_not_banner_note",
        "h2y_resolution_marker_old_quote_not_marker",
        "h2y_not_replied_status_pill_value_before_component",
        "h2y_not_sent_delivery_tag_value_before_component",
        "h2y_not_required_approval_marker_value_before_component",
        "h2y_not_escalated_risk_chip_value_before_component",
        "h2y_not_available_owner_field_value_before_component",
        "h2y_not_started_phase_tile_value_before_component",
    }

    full_h2w_boundaries = {
        row["case_id"]
        for row in payload["unresolved_boundary_rows"]
        if row["profile_label"] == "h2y_h2w_semantic_target_preservation"
    }
    assert full_h2w_boundaries == {
        "h2y_escalation_lane_stale_selection_not_lane",
        "h2y_exception_panel_stale_selection_not_panel",
        "h2y_approval_field_stale_selection_not_field",
        "h2y_not_active_alert_banner_value_before_component",
    }
    assert any(
        row["case_id"] == "h2y_not_active_alert_banner_value_before_component"
        and '"target_query": "alert"' in row["actual_calls"]
        for row in payload["unresolved_boundary_rows"]
    )

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2y expands H2x to 16 cases" in findings["h2y_scales_h2x_pressure_and_breaks_h2w_saturation"]
    assert "not fallback" in findings["next_helper_target_is_stale_selection_negation_and_short_component_value"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2y_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2y_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2y_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2y_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2y_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2y_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h2y_unresolved_boundary_rows.csv").exists()
    assert (tmp_path / "tables" / "h2y_findings.csv").exists()
    assert (tmp_path / "figures" / "h2y_scaled_cli_semantic_pressure_gate.svg").exists()
