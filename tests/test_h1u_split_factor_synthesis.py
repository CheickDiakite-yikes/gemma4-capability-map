from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1u_split_factor_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1u_split_factor_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_build_h1u_split_factor_synthesis_marks_v15_transfer_candidate(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1u_split_factor_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["profile_count"] == 6
    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["v15_exact_success_count"] == 6
    assert payload["manifest"]["v15_executor_success_count"] == 6
    assert payload["manifest"]["promotion_decision"] == "transfer_test_v15"

    packet_rows = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packet_rows["conditional_residual_route_v13"]["exact_success_count"] == 3
    assert packet_rows["nonstandard_component_class_guard_v14"]["exact_success_count"] == 5
    assert packet_rows["code_label_exact_guard_v15"]["exact_success_count"] == 6

    family_rows = {
        (row["profile_label"], row["family"]): row for row in payload["family_rows"]
    }
    assert family_rows[
        ("nonstandard_component_class_guard_v14", "h1r_nonstandard_component_class")
    ]["exact_success_count"] == 2
    assert family_rows[
        ("code_label_exact_guard_v15", "h1r_code_label_exactness")
    ]["exact_success_count"] == 2

    comparisons = {row["comparison_dir"]: row for row in payload["comparison_rows"]}
    v15_vs_v11 = next(row for key, row in comparisons.items() if "code_label_exact_guard_h1r_vs_component_label_guard" in key)
    v15_vs_v12 = next(row for key, row in comparisons.items() if "code_label_exact_guard_h1r_vs_component_residual_guard" in key)
    assert v15_vs_v11["delta_exact_rate"] > 0
    assert v15_vs_v12["delta_executor_equivalence_rate"] == 0.0

    failures = {
        (row["profile_label"], row["case_id"]): row for row in payload["non_exact_rows"]
    }
    assert failures[
        ("nonstandard_component_class_guard_v14", "h1r_alert_s92_toggle_negation_decoy")
    ]["failure_mode"] == "argument_mismatch"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "v15 reaches 6/6 exact" in findings["v15_saturates_h1r"]
    assert "Transfer-test v15" in findings["transfer_decision"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1u_split_factor_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1u_split_factor_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h1u_split_factor_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1u_split_factor_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h1u_split_factor_findings.csv").exists()
