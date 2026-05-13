from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2V_MODULE_PATH = ROOT / "scripts" / "build_h2v_semantic_negation_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
H2V_SCRIPT = _load_module("build_h2v_semantic_negation_synthesis", H2V_MODULE_PATH)


def test_h2v_semantic_negation_synthesis_marks_break_and_next_repair(tmp_path: Path) -> None:
    payload = H2V_SCRIPT.build_h2v_semantic_negation_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 3
    assert manifest["comparison_count"] == 3
    assert manifest["h2v_case_count"] == 10
    assert manifest["h2j_exact_success_count"] == 3
    assert manifest["h2j_executor_success_count"] == 4
    assert manifest["h2r_exact_success_count"] == 3
    assert manifest["h2r_executor_success_count"] == 4
    assert manifest["h2u_exact_success_count"] == 4
    assert manifest["h2u_executor_success_count"] == 5
    assert manifest["h2u_delta_exact_vs_h2r"] == 0.10000000000000003
    assert manifest["h2u_delta_executor_vs_h2r"] == 0.09999999999999998
    assert manifest["h2u_delta_exact_vs_h2j"] == 0.10000000000000003
    assert manifest["h2u_delta_executor_vs_h2j"] == 0.09999999999999998
    assert manifest["h2r_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2r_delta_executor_vs_h2j"] == 0.0
    assert manifest["h2u_non_exact_count"] == 6
    assert manifest["h2u_executor_non_equivalent_count"] == 5
    assert manifest["h2u_quoted_exact_success_count"] == 1
    assert manifest["h2u_instructional_exact_success_count"] == 2
    assert manifest["h2u_stale_example_exact_success_count"] == 0
    assert manifest["h2u_genuine_negated_exact_success_count"] == 0
    assert manifest["h2u_genuine_negated_executor_success_count"] == 1
    assert manifest["h2u_fixed_case_count_vs_h2r"] == 1
    assert manifest["promotion_decision"] == "h2u_not_promoted_until_h2w_semantic_target_preservation"

    fixed_cases = {row["case_id"] for row in payload["fixed_case_rows"]}
    assert fixed_cases == {"h2v_metric_panel_quoted_not_label_note"}

    h2u_non_exact = {
        row["case_id"]: row
        for row in payload["non_exact_rows"]
        if row["profile_label"] == "h2v_h2u_negation_guard"
    }
    assert h2u_non_exact["h2v_not_ready_badge_genuine_value"]["actual_target_query"] == "Not ready"
    assert h2u_non_exact["h2v_not_ready_badge_genuine_value"]["expected_target_query"] == "status badge Not ready"
    assert h2u_non_exact["h2v_review_tile_stale_caption_old_not_tile"]["actual_target_query"] == "stale caption"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "breaks the prior H2u same-family transfer saturation" in findings["h2v_breaks_h2u_transfer_saturation"]
    assert "only by one case" not in findings["h2u_negation_guard_help_is_real_but_small"]
    assert "Composed route gating alone does not solve" in findings["h2r_and_h2j_tie_on_h2v"]
    assert "genuine negated targets" in findings["h2v_family_split_identifies_next_repair"]
    assert "h2v_metric_panel_quoted_not_label_note" in findings["h2u_fixed_case_is_one_quoted_context_row"]
    assert "distinguish negated context" in findings["next_h2w_should_preserve_semantic_targets"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2v_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2v_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2v_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2v_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2v_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2v_fixed_case_rows.csv").exists()
    assert (tmp_path / "tables" / "h2v_findings.csv").exists()
    assert (tmp_path / "figures" / "h2v_semantic_negation_gate.svg").exists()
