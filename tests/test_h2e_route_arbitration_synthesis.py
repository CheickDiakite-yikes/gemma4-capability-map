from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2e_route_arbitration_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2e_route_arbitration_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2e_route_arbitration_synthesis_saturates_h2b_and_h1x(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2e_route_arbitration_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 10
    assert manifest["comparison_count"] == 9
    assert manifest["h2e_h2b_exact_success_count"] == 5
    assert manifest["h2e_h2b_executor_success_count"] == 5
    assert manifest["h2e_h1x_exact_success_count"] == 8
    assert manifest["h2e_h1x_executor_success_count"] == 8
    assert manifest["h2b_delta_exact_vs_h2c"] == 0.0
    assert manifest["h2b_delta_executor_vs_h2c"] == 0.0
    assert manifest["h1x_delta_exact_vs_h2c"] == 0.125
    assert manifest["h1x_delta_executor_vs_h2c"] == 0.125
    assert manifest["h2e_non_exact_count"] == 0
    assert manifest["promotion_decision"] == "promote_to_fresh_h2f_holdout_not_global_default"

    assert payload["h2e_non_exact_rows"] == []
    counterfactual = {row["case_id"]: row for row in payload["counterfactual_miss_rows"]}
    assert set(counterfactual) == {
        "h1x_resolution_chip_comment_result_decoy",
        "h1o_code_badge_c08_note_decoy",
    }
    assert '"target_query": "result pill"' in counterfactual[
        "h1x_resolution_chip_comment_result_decoy"
    ]["actual_arguments"]
    assert '"target_query": "escalated badge c08"' in counterfactual[
        "h1o_code_badge_c08_note_decoy"
    ]["actual_arguments"]

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2e reaches 5/5 exact on H2b and 8/8 exact on H1x" in findings[
        "h2e_saturates_both_h2b_and_h1x"
    ]
    assert "H2e preserves the max of both" in findings["h2e_reconciles_h2c_h2d_tradeoff"]
    assert "H2e has zero non-exact rows" in findings["counterfactual_misses_are_covered"]
    assert "fresh H2f holdout gate" in findings["next_slice"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2e_route_arbitration_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2e_route_arbitration_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2e_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2e_counterfactual_miss_rows.csv").exists()
    assert (tmp_path / "tables" / "h2e_route_arbitration_findings.csv").exists()
    assert (tmp_path / "figures" / "h2e_route_arbitration_gate.svg").exists()
