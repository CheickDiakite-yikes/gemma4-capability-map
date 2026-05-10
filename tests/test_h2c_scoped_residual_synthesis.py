from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2c_scoped_residual_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2c_scoped_residual_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2c_scoped_residual_synthesis_promotes_transfer_gate_not_global_default(
    tmp_path: Path,
) -> None:
    payload = SCRIPT.build_h2c_scoped_residual_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["profile_count"] == 5
    assert manifest["case_count"] == 5
    assert manifest["h2c_exact_success_count"] == 5
    assert manifest["h2c_executor_success_count"] == 5
    assert manifest["v12_exact_success_count"] == 4
    assert manifest["v12_executor_success_count"] == 4
    assert manifest["h2a_exact_success_count"] == 0
    assert manifest["h2a_executor_success_count"] == 3
    assert manifest["strict_winner"] == "h2c_scoped_residual_gate"
    assert manifest["executor_winner"] == "h2c_scoped_residual_gate"
    assert manifest["promotion_decision"] == "transfer_gate_required_before_global_or_default_promotion"

    packet_rows = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packet_rows["h2c_scoped_residual_gate"]["exact_success_count"] == 5
    assert packet_rows["component_residual_guard_v12"]["exact_success_count"] == 4
    assert packet_rows["h2a_stale_selection_gate"]["exact_success_count"] == 0

    comparisons = {row["comparison_label"]: row for row in payload["comparison_rows"]}
    assert comparisons["h2c_vs_v12"]["delta_exact_rate"] == 0.19999999999999996
    assert comparisons["h2c_vs_h2a"]["delta_exact_rate"] == 1.0
    assert comparisons["h2c_vs_no_directive"]["delta_executor_equivalence_rate"] == 0.6

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "5/5 strict" in findings["h2c_saturates_h2b_residuals"]
    assert "minimal transfer check" in findings["next_slice"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2c_scoped_residual_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2c_scoped_residual_case_matrix.csv").exists()
    assert (tmp_path / "tables" / "h2c_scoped_residual_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2c_scoped_residual_findings.csv").exists()
