from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1n_code_hints_transfer_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1n_code_hints_transfer_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1n_code_hints_transfer_synthesis_reports_localized_repair(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1n_code_hints_transfer_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["comparison_count"] == 3
    assert payload["manifest"]["total_case_count"] == 18
    assert payload["aggregate"]["baseline_exact_success_count"] == 14
    assert payload["aggregate"]["candidate_exact_success_count"] == 11
    assert payload["aggregate"]["baseline_executor_success_count"] == 16
    assert payload["aggregate"]["candidate_executor_success_count"] == 12

    rows = {row["label"]: row for row in payload["summary_rows"]}
    assert rows["oracle_transfer_v2"]["delta_executor_equivalence_rate"] == -0.5
    assert rows["oracle_repeat_v1"]["delta_executor_equivalence_rate"] == -0.33333333333333337
    assert rows["oblique_v5"]["delta_executor_equivalence_rate"] == 0.16666666666666674

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "oblique_v5" in findings["localized_oblique_repair"]
    assert "oracle_transfer_v2, oracle_repeat_v1" in findings["negative_transfer_elsewhere"]
    assert "not a replacement for argument hints" in findings["promotion_decision"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1n_code_hints_transfer_summary.csv").exists()
