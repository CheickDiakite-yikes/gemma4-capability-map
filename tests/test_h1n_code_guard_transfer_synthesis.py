from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1n_code_guard_transfer_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1n_code_guard_transfer_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1n_code_guard_transfer_synthesis_keeps_argument_hints_boundary(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1n_code_guard_transfer_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["argument_hint_comparison_count"] == 3
    assert payload["manifest"]["code_hint_comparison_count"] == 3
    assert payload["manifest"]["total_case_count"] == 18
    assert payload["manifest"]["argument_hints_exact_success_count"] == 14
    assert payload["manifest"]["code_guard_exact_success_count"] == 14
    assert payload["manifest"]["argument_hints_executor_success_count"] == 16
    assert payload["manifest"]["code_guard_executor_success_count"] == 15
    assert payload["manifest"]["code_hints_exact_success_count"] == 11
    assert payload["manifest"]["code_hints_executor_success_count"] == 12

    rows = {row["label"]: row for row in payload["argument_hint_rows"]}
    assert rows["oracle_transfer_v2"]["delta_executor_equivalence_rate"] == -0.33333333333333337
    assert rows["oracle_repeat_v1"]["delta_executor_equivalence_rate"] == -0.16666666666666663
    assert rows["oblique_v5"]["delta_executor_equivalence_rate"] == 0.33333333333333337

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "v6 at 11/18 exact and 12/18 executor-equivalent" in findings["code_guard_beats_v6"]
    assert "Argument hints remains the stronger executor-equivalence baseline" in findings[
        "argument_hints_still_best_executor"
    ]
    assert "better scoped repair than v6" in findings["promotion_decision"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1n_code_guard_vs_argument_hints_summary.csv").exists()
