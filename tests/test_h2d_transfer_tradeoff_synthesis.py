from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2d_transfer_tradeoff_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2d_transfer_tradeoff_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2d_transfer_tradeoff_synthesis_promotes_route_arbitration_not_global_replacement(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2d_transfer_tradeoff_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 8
    assert manifest["comparison_count"] == 7
    assert manifest["h2d_h2b_exact_success_count"] == 4
    assert manifest["h2d_h2b_executor_success_count"] == 5
    assert manifest["h2d_h1x_exact_success_count"] == 8
    assert manifest["h2d_h1x_executor_success_count"] == 8
    assert manifest["h2b_delta_exact_vs_h2c"] == -0.19999999999999996
    assert manifest["h2b_delta_executor_vs_h2c"] == 0.0
    assert manifest["h1x_delta_exact_vs_h2c"] == 0.125
    assert manifest["h1x_delta_executor_vs_h2c"] == 0.125
    assert manifest["promotion_decision"] == "reject_global_h2d_build_h2e_route_arbitration"

    h2d_misses = payload["h2d_h2b_non_exact_rows"]
    assert len(h2d_misses) == 1
    assert h2d_misses[0]["case_id"] == "h1o_code_badge_c08_note_decoy"
    assert '"target_query": "badge c08"' in h2d_misses[0]["expected_arguments"]
    assert '"target_query": "escalated badge c08"' in h2d_misses[0]["actual_arguments"]
    assert h2d_misses[0]["executor_equivalence_match"] is True

    h2c_misses = payload["h2c_h1x_non_exact_rows"]
    assert len(h2c_misses) == 1
    assert h2c_misses[0]["case_id"] == "h1x_resolution_chip_comment_result_decoy"
    assert '"target_query": "result chip"' in h2c_misses[0]["expected_arguments"]
    assert '"target_query": "result pill"' in h2c_misses[0]["actual_arguments"]
    assert h2c_misses[0]["executor_equivalence_match"] is False

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2d is 8/8 on H1x" in findings["h2d_repairs_h2c_transfer_regression"]
    assert "H2d is 4/5 on H2b" in findings["h2d_pays_local_h2b_exactness_cost"]
    assert "executor still selected the same region" in findings["h2d_h2b_miss_is_executor_equivalent_over_specific_query"]
    assert "broke executor-equivalence" in findings["h2c_h1x_miss_is_not_executor_equivalent"]
    assert "Build H2e as route arbitration" in findings["next_slice"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2d_tradeoff_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2d_tradeoff_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2d_h2b_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2c_h1x_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2d_tradeoff_findings.csv").exists()
