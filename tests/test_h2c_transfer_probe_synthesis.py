from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2c_transfer_probe_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2c_transfer_probe_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2c_transfer_probe_synthesis_rejects_global_h2c(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2c_transfer_probe_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 5
    assert manifest["comparison_count"] == 4
    assert manifest["h2c_h2b_exact_success_count"] == 5
    assert manifest["h2c_h2b_executor_success_count"] == 5
    assert manifest["h2c_h1x_exact_success_count"] == 7
    assert manifest["h2c_h1x_executor_success_count"] == 7
    assert manifest["h1x_delta_exact_vs_h2a"] == -0.125
    assert manifest["h1x_delta_executor_vs_h2a"] == -0.125
    assert manifest["promotion_decision"] == "reject_global_h2c_build_h2d_class_preserving_route"

    non_exact = payload["h2c_h1x_non_exact_rows"]
    assert len(non_exact) == 1
    assert non_exact[0]["case_id"] == "h1x_resolution_chip_comment_result_decoy"
    assert '"target_query": "result chip"' in non_exact[0]["expected_arguments"]
    assert '"target_query": "result pill"' in non_exact[0]["actual_arguments"]

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "only 7/8 on H1x" in findings["h2c_local_fit_does_not_transfer_cleanly_to_h1x"]
    assert "class-swap overfit" in findings["h2c_regression_is_component_class_swap"]
    assert "Build H2d" in findings["next_slice"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2c_transfer_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2c_transfer_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2c_transfer_h1x_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2c_transfer_findings.csv").exists()
