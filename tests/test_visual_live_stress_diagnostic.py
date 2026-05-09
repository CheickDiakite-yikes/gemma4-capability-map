from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_visual_live_stress_matrix.py"
SPEC = importlib.util.spec_from_file_location("analyze_visual_live_stress_matrix_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_visual_live_stress_diagnostic_writes_findings(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_visual_live_stress_matrix(output_dir=tmp_path)

    assert payload["manifest"]["comparison_count"] == 5
    assert payload["manifest"]["case_count"] == 4
    summary = {row["label"]: row for row in payload["summary_rows"]}
    assert summary["contracted"]["candidate_exact_rate"] == 1.0
    assert summary["schema_field_hints_v4"]["candidate_executor_equivalence_rate"] == 1.0
    assert summary["role_catalog_v1"]["delta_exact_rate"] == -0.25
    transitions = {
        (row["label"], row["case_id"]): row for row in payload["case_rows"]
    }
    assert transitions[
        ("schema_field_hints_v4", "stress_metric_panel_with_chart_table_decoys")
    ]["transition"] == "executor_gain_without_strict"
    assert transitions[
        ("role_catalog_v1", "stress_form_error_stale_selection_warning_decoy")
    ]["transition"] == "regression"
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "contracted" in findings["strict_upper_bound"]
    assert "schema_field_hints_v4" in findings["executor_without_strict"]
    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "stress_matrix_case_transitions.csv").exists()


def test_visual_alias_repeat_diagnostic_writes_findings(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_visual_live_stress_matrix(
        output_dir=tmp_path,
        comparisons=SCRIPT.DEFAULT_ALIAS_REPEAT_COMPARISONS,
        matrix_name="alias-repeat",
        table_prefix="alias_repeat_matrix",
    )

    assert payload["manifest"]["comparison_count"] == 5
    assert payload["manifest"]["case_count"] == 8
    assert payload["manifest"]["matrix_name"] == "alias-repeat"
    summary = {row["label"]: row for row in payload["summary_rows"]}
    assert summary["contracted"]["candidate_exact_rate"] == 0.875
    assert summary["schema_literal_targets_v5"]["candidate_executor_equivalence_rate"] == 1.0
    assert summary["schema_field_hints_v4"]["delta_exact_rate"] == 0.0
    assert summary["schema_field_hints_v4"]["delta_executor_equivalence_rate"] == 0.25
    transitions = {(row["label"], row["case_id"]): row for row in payload["case_rows"]}
    assert transitions[
        ("schema_literal_targets_v5", "stress_callout_warning_risk_note_decoy")
    ]["transition"] == "strict_gain"
    assert transitions[
        ("role_catalog_v1", "stress_form_error_stale_selection_warning_decoy")
    ]["transition"] == "regression"
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "contracted" in findings["strict_upper_bound"]
    assert "schema_literal_targets_v5" in findings["executor_equivalence_set"]
    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "alias_repeat_matrix_case_transitions.csv").exists()
