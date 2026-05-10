from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1w_residual_overlap_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1w_residual_overlap_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1w_residual_overlap_synthesis_marks_v11_default(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1w_residual_overlap_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["profile_count"] == 4
    assert payload["manifest"]["case_count"] == 8
    assert payload["manifest"]["no_directive_exact_success_count"] == 0
    assert payload["manifest"]["v11_exact_success_count"] == 8
    assert payload["manifest"]["v11_executor_success_count"] == 8
    assert payload["manifest"]["promotion_decision"] == "component_label_guard_remains_default"

    packet_rows = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packet_rows["no_directive"]["exact_success_count"] == 0
    assert packet_rows["component_label_guard_v11"]["exact_success_count"] == 8
    assert packet_rows["component_residual_guard_v12"]["exact_success_count"] == 7
    assert packet_rows["code_label_exact_guard_v15"]["exact_success_count"] == 6

    family_rows = {(row["profile_label"], row["family"]): row for row in payload["family_rows"]}
    assert family_rows[("component_label_guard_v11", "h1w_surface_component_value")][
        "exact_success_count"
    ] == 2
    assert family_rows[("code_label_exact_guard_v15", "h1w_surface_component_value")][
        "exact_success_count"
    ] == 0
    assert family_rows[("code_label_exact_guard_v15", "h1w_stale_field_routing")][
        "exact_success_count"
    ] == 2

    comparisons = {row["comparison_dir"]: row for row in payload["comparison_rows"]}
    v11_vs_no = next(row for key, row in comparisons.items() if "component_label_guard_vs_no_directive" in key)
    v15_vs_v11 = next(row for key, row in comparisons.items() if "code_label_exact_guard_vs_component_label_guard" in key)
    assert v11_vs_no["delta_exact_rate"] == 1.0
    assert v11_vs_no["delta_executor_equivalence_rate"] == 1.0
    assert v15_vs_v11["delta_exact_rate"] == -0.25
    assert v15_vs_v11["delta_executor_equivalence_rate"] == -0.25

    failures = {(row["profile_label"], row["case_id"]): row for row in payload["non_exact_rows"]}
    assert failures[("component_residual_guard_v12", "h1w_status_pill_summary_value_decoy")][
        "failure_mode"
    ] == "argument_mismatch"
    assert failures[("code_label_exact_guard_v15", "h1w_result_badge_comment_value_decoy")][
        "failure_mode"
    ] == "argument_mismatch"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "0/8" in findings["h1w_breaks_no_directive"]
    assert "8/8" in findings["v11_saturates_h1w"]
    assert "surface component-value" in findings["v15_surface_value_weakness"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1w_residual_overlap_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1w_residual_overlap_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h1w_residual_overlap_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1w_residual_overlap_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h1w_residual_overlap_findings.csv").exists()
