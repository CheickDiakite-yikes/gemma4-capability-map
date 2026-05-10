from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1t_conditional_residual_route_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1t_conditional_residual_route_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_build_h1t_conditional_residual_route_synthesis_marks_early_stop(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1t_conditional_residual_route_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["profile_count"] == 4
    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["v13_exact_success_count"] == 3
    assert payload["manifest"]["v13_executor_success_count"] == 3
    assert payload["manifest"]["early_stop"] is True
    assert payload["manifest"]["promotion_decision"] == "reject_before_broader_transfer"

    packet_rows = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packet_rows["component_label_guard_v11"]["exact_success_count"] == 5
    assert packet_rows["component_residual_guard_v12"]["exact_success_count"] == 6
    assert packet_rows["conditional_residual_route_v13"]["exact_success_count"] == 3

    family_rows = {
        (row["profile_label"], row["family"]): row for row in payload["family_rows"]
    }
    assert family_rows[
        ("conditional_residual_route_v13", "h1r_stale_selection_component_label")
    ]["exact_success_count"] == 2
    assert family_rows[
        ("conditional_residual_route_v13", "h1r_nonstandard_component_class")
    ]["exact_success_count"] == 0

    comparisons = {row["comparison_dir"]: row for row in payload["comparison_rows"]}
    v13_vs_v11 = next(row for key, row in comparisons.items() if "component_label_guard" in key)
    v13_vs_v12 = next(row for key, row in comparisons.items() if "component_residual_guard" in key)
    assert v13_vs_v11["delta_exact_rate"] < 0
    assert v13_vs_v12["delta_executor_equivalence_rate"] == -0.5

    failures = {row["case_id"]: row for row in payload["v13_failure_rows"]}
    assert failures["h1r_state_tag_log_value_decoy"]["failure_mode"] == "argument_mismatch"
    assert failures["h1r_mode_toggle_note_value_decoy"]["failure_mode"] == "argument_mismatch"
    assert failures["h1r_alert_s92_toggle_negation_decoy"]["failure_mode"] == "argument_mismatch"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "3/6" in findings["v13_fails_h1r_gate"]
    assert "Stop before H1n/H1o/H1p transfer" in findings["early_stop_decision"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "h1t_conditional_residual_route_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1t_conditional_residual_route_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h1t_conditional_residual_route_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1t_conditional_residual_route_v13_failures.csv").exists()
    assert (tmp_path / "tables" / "h1t_conditional_residual_route_findings.csv").exists()
