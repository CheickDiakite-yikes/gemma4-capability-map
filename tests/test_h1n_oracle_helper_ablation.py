from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_h1n_oracle_helper_ablation.py"
SPEC = importlib.util.spec_from_file_location("analyze_h1n_oracle_helper_ablation_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_h1n_oracle_helper_ablation_reports_no_observed_helper_dependence(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_h1n_oracle_helper_ablation(output_dir=tmp_path)

    assert payload["findings"]["helper_count"] == 3
    assert payload["findings"]["all_helpers_preserve_exact_rate"] is True
    assert payload["findings"]["all_helpers_preserve_executor_equivalence_rate"] is True
    assert payload["findings"]["strict_rate"] == 0.8333333333333334
    assert payload["findings"]["executor_equivalence_rate"] == 1.0
    by_helper = {row["helper_removed"]: row for row in payload["summary_rows"]}
    assert by_helper["no_controller_repair"]["classification"] == "no_observed_helper_dependence"
    assert by_helper["no_controller_fallback"]["delta_exact_rate"] == 0.0
    assert by_helper["no_argument_repair"]["delta_executor_equivalence_rate"] == 0.0

    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "h1n_oracle_helper_ablation_summary.csv").exists()
    assert (tmp_path / "tables" / "h1n_oracle_helper_ablation_case_deltas.csv").exists()
