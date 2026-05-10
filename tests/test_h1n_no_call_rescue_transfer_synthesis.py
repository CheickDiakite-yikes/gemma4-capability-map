from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1n_no_call_rescue_transfer_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1n_no_call_rescue_transfer_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1n_no_call_rescue_transfer_synthesis_marks_scoped_gain(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1n_no_call_rescue_transfer_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["comparison_count"] == 4
    assert payload["manifest"]["total_case_count"] == 30
    assert payload["manifest"]["no_directive_exact_success_count"] == 11
    assert payload["manifest"]["v10_exact_success_count"] == 22
    assert payload["manifest"]["incumbent_exact_success_count"] == 25
    assert payload["manifest"]["no_directive_executor_success_count"] == 12
    assert payload["manifest"]["v10_executor_success_count"] == 25
    assert payload["manifest"]["incumbent_executor_success_count"] == 26

    rows = {row["label"]: row for row in payload["summary_rows"]}
    assert rows["component_value_v10"]["delta_executor_vs_incumbent"] == 0.125
    assert rows["residual_v8"]["delta_executor_vs_incumbent"] == -0.125
    assert rows["post_repair_v7"]["delta_executor_vs_incumbent"] == 0.0
    assert rows["oblique_v7"]["delta_executor_vs_incumbent"] == -0.16666666666666663

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "22/30 exact versus 11/30 no-directive" in findings["large_no_directive_lift"]
    assert "25/30 executor-equivalent versus incumbents at 26/30" in findings[
        "not_universal_replacement"
    ]
    assert "positive on component_value_v10" in findings["transfer_pattern"]
    assert "negative on residual_v8, oblique_v7" in findings["transfer_pattern"]
    assert "current-image/no-call activation guard" in findings["promotion_decision"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1n_no_call_rescue_transfer_summary.csv").exists()
