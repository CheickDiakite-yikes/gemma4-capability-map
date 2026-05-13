from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2W_MODULE_PATH = ROOT / "scripts" / "build_h2w_semantic_target_preservation_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
H2W_SCRIPT = _load_module("build_h2w_semantic_target_preservation_synthesis", H2W_MODULE_PATH)


def test_h2w_semantic_target_preservation_synthesis_marks_repair_and_next_gate(tmp_path: Path) -> None:
    payload = H2W_SCRIPT.build_h2w_semantic_target_preservation_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 4
    assert manifest["comparison_count"] == 3
    assert manifest["family_row_count"] == 20
    assert manifest["h2v_case_count"] == 10
    assert manifest["h2j_exact_success_count"] == 3
    assert manifest["h2r_exact_success_count"] == 3
    assert manifest["h2u_exact_success_count"] == 4
    assert manifest["h2w_exact_success_count"] == 10
    assert manifest["h2w_executor_success_count"] == 10
    assert manifest["h2w_delta_exact_vs_h2u"] == 0.6
    assert manifest["h2w_delta_executor_vs_h2u"] == 0.5
    assert manifest["h2w_delta_exact_vs_h2r"] == 0.7
    assert manifest["h2w_delta_executor_vs_h2r"] == 0.6
    assert manifest["h2w_delta_exact_vs_h2j"] == 0.7
    assert manifest["h2w_delta_executor_vs_h2j"] == 0.6
    assert manifest["h2w_non_exact_count"] == 0
    assert manifest["h2w_fixed_case_count_vs_h2u"] == 6
    assert manifest["h2w_semantic_target_preservation_count"] == 4
    assert manifest["h2w_target_query_normalization_count"] == 3
    assert manifest["h2w_stale_selection_gate_count"] == 1
    assert manifest["h2w_composed_route_gating_blocked_count"] == 1
    assert manifest["h2w_all_families_exact"] is True
    assert manifest["promotion_decision"] == "h2w_repairs_h2v_requires_transfer_backtest_before_packaged_workflows"

    fixed_cases = {
        row["case_id"]
        for row in payload["fixed_case_rows"]
        if row["comparison_label"] == "h2v_h2w_vs_h2u"
    }
    assert fixed_cases == {
        "h2v_summary_tile_quoted_not_label_caption",
        "h2v_review_tile_stale_caption_old_not_tile",
        "h2v_risk_lane_stale_example_not_lane",
        "h2v_not_ready_badge_genuine_value",
        "h2v_not_applicable_chip_genuine_value",
        "h2v_not_approved_toggle_genuine_value",
    }

    h2w_interventions = [
        row for row in payload["intervention_rows"] if row["profile_label"] == "h2v_h2w_semantic_target_preservation"
    ]
    intervention_kinds = {row["intervention_kind"] for row in h2w_interventions}
    assert "visual_semantic_target_preservation" in intervention_kinds
    assert "visual_target_query_normalization" in intervention_kinds
    assert any(row["reason"] == "semantic_label_preserved_over_stale_context" for row in h2w_interventions)
    assert any(
        row["case_id"] == "h2v_summary_tile_quoted_not_label_caption"
        and row["reason"] == "semantic_label_preserved_over_stale_context"
        for row in h2w_interventions
    )

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "repairs H2v" in findings["h2w_repairs_h2v_strict_and_executor"]
    assert "six H2u misses" not in findings["h2w_gain_is_causal_on_six_h2u_misses"]
    assert "bounded no-call visual fallback" in findings["h2w_next_requires_transfer_backtest"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2w_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2w_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2w_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2w_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2w_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2w_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h2w_findings.csv").exists()
    assert (tmp_path / "figures" / "h2w_semantic_target_preservation_gate.svg").exists()
