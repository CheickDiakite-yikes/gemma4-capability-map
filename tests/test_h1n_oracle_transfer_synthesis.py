from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1n_oracle_transfer_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1n_oracle_transfer_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_h1n_oracle_transfer_synthesis_writes_report(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1n_oracle_transfer_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["surface_row_count"] == 10
    assert payload["manifest"]["helper_row_count"] == 3
    assert payload["manifest"]["finding_count"] == 4

    rows = {(row["surface"], row["label"]): row for row in payload["synthesis_rows"]}
    assert rows[("oracle_v2", "argument_hints_v2")]["candidate_executor_equivalence_rate"] == 1.0
    assert rows[("oracle_v2", "argument_hints_v2")]["candidate_exact_rate"] == 0.8333333333333334
    assert rows[("repeat_v1", "schema_literal_targets_v5")]["candidate_executor_equivalence_rate"] == 1.0
    assert rows[("repeat_v1", "contracted")]["candidate_exact_rate"] == 0.0

    findings = {row["finding_id"]: row["finding"] for row in payload["findings"]}
    assert "argument_hints_v2" in findings["oracle_v2_winner"]
    assert "argument_hints_v2" in findings["repeat_winner_set"]
    assert "schema_literal_targets_v5" in findings["repeat_winner_set"]
    assert "True" in findings["helper_dependence"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h1n_oracle_transfer_synthesis.csv").exists()
    assert (tmp_path / "tables" / "h1n_oracle_helper_synthesis.csv").exists()
    assert (tmp_path / "tables" / "h1n_oracle_transfer_findings.csv").exists()
