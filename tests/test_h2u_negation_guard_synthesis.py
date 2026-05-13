from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2U_MODULE_PATH = ROOT / "scripts" / "build_h2u_negation_guard_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
H2U_SCRIPT = _load_module("build_h2u_negation_guard_synthesis", H2U_MODULE_PATH)


def test_h2u_negation_guard_synthesis_marks_repair_and_transfer(tmp_path: Path) -> None:
    payload = H2U_SCRIPT.build_h2u_negation_guard_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["h2t_case_count"] == 10
    assert manifest["h2t_h2r_exact_success_count"] == 8
    assert manifest["h2t_h2u_exact_success_count"] == 10
    assert manifest["h2t_h2u_executor_success_count"] == 10
    assert manifest["h2t_delta_exact_vs_h2r"] == 0.19999999999999996
    assert manifest["h2t_delta_executor_vs_h2r"] == 0.19999999999999996
    assert manifest["h2t_fixed_case_count"] == 2
    assert manifest["h2s_h2u_exact_success_count"] == 10
    assert manifest["h2q_h2u_exact_success_count"] == 8
    assert manifest["h2m_h2u_exact_success_count"] == 8
    assert manifest["packet_row_count"] == 18
    assert manifest["comparison_count"] == 9
    assert manifest["transfer_case_count"] == 26
    assert manifest["transfer_exact_success_count"] == 26
    assert manifest["transfer_delta_exact_sum_vs_h2r"] == 0.0
    assert manifest["first_pass_transfer_case_count"] == 39
    assert manifest["first_pass_transfer_exact_success_count"] == 39
    assert manifest["first_pass_transfer_delta_exact_sum_vs_h2r"] == 0.0
    assert manifest["broad_transfer_case_count"] == 65
    assert manifest["broad_transfer_exact_success_count"] == 65
    assert manifest["broad_transfer_delta_exact_sum_vs_h2r"] == 0.0
    assert manifest["h2k_h2u_exact_success_count"] == 8
    assert manifest["h2l_h2u_exact_success_count"] == 8
    assert manifest["h2f_h2u_exact_success_count"] == 10
    assert manifest["h2b_h2u_exact_success_count"] == 5
    assert manifest["h1x_h2u_exact_success_count"] == 8
    assert manifest["blocked_guard_count"] == 7
    assert manifest["h2t_blocked_guard_count"] == 4
    assert manifest["transfer_blocked_guard_count"] == 3
    assert manifest["target_normalization_blocked_count"] == 4
    assert manifest["composed_route_gating_blocked_count"] == 3
    assert manifest["h2u_non_exact_count"] == 0
    assert manifest["promotion_decision"] == "h2u_promotes_to_broader_transfer_backtest"

    fixed_cases = {row["case_id"] for row in payload["fixed_case_rows"]}
    assert fixed_cases == {
        "h2t_metric_panel_negation_scope_note",
        "h2t_summary_tile_negation_scope_caption",
    }

    blocked_rows = {(row["case_id"], row["intervention_kind"]) for row in payload["blocked_guard_rows"]}
    assert (
        "h2t_metric_panel_negation_scope_note",
        "visual_target_query_normalization_blocked",
    ) in blocked_rows
    assert (
        "h2t_metric_panel_negation_scope_note",
        "visual_composed_route_gating_blocked",
    ) in blocked_rows
    assert (
        "h2t_summary_tile_negation_scope_caption",
        "visual_target_query_normalization_blocked",
    ) in blocked_rows
    assert (
        "h2t_summary_tile_negation_scope_caption",
        "visual_composed_route_gating_blocked",
    ) in blocked_rows
    assert (
        "h2k_alert_t47_archived_alert_s92_decoy",
        "visual_composed_route_gating_blocked",
    ) in blocked_rows

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "raises H2t from H2r's 8/10" in findings["h2u_repairs_h2t_negation_scope"]
    assert "both target normalization and composed-route gating" in findings["h2u_fix_is_pipeline_ordered"]
    assert "26/26 strict exactness" in findings["h2u_transfer_preserves_h2r"]
    assert "39/39 strict exactness" in findings["h2u_first_pass_transfer_preserves_h2r"]
    assert "without breaking prior wins" in findings["h2u_guard_fires_without_transfer_cost"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2u_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2u_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2u_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2u_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h2u_blocked_guard_rows.csv").exists()
    assert (tmp_path / "tables" / "h2u_findings.csv").exists()
    assert (tmp_path / "figures" / "h2u_negation_guard_transfer_gate.svg").exists()
