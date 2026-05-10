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


def test_visual_alias_transfer_diagnostic_writes_findings(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_visual_live_stress_matrix(
        output_dir=tmp_path,
        comparisons=SCRIPT.DEFAULT_ALIAS_TRANSFER_COMPARISONS,
        matrix_name="alias-transfer",
        table_prefix="alias_transfer_matrix",
    )

    assert payload["manifest"]["comparison_count"] == 5
    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["matrix_name"] == "alias-transfer"
    summary = {row["label"]: row for row in payload["summary_rows"]}
    assert summary["contracted"]["candidate_exact_rate"] == 0.8333333333333334
    assert summary["contracted"]["delta_executor_equivalence_rate"] == -0.16666666666666666
    assert summary["argument_hints_v2"]["candidate_executor_equivalence_rate"] == 1.0
    assert summary["schema_literal_targets_v5"]["delta_executor_equivalence_rate"] == 0.3333333333333333
    transitions = {(row["label"], row["case_id"]): row for row in payload["case_rows"]}
    assert transitions[
        ("argument_hints_v2", "transfer_review_tile_notice_table_decoy")
    ]["transition"] == "executor_gain_without_strict"
    assert transitions[
        ("contracted", "transfer_status_pill_chart_decoy")
    ]["transition"] == "unchanged"
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "contracted" in findings["strict_upper_bound"]
    assert "argument_hints_v2" in findings["executor_equivalence_set"]
    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "alias_transfer_matrix_case_transitions.csv").exists()


def test_visual_alias_transfer_oracle_diagnostic_writes_findings(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_visual_live_stress_matrix(
        output_dir=tmp_path,
        comparisons=SCRIPT.DEFAULT_ALIAS_TRANSFER_ORACLE_COMPARISONS,
        matrix_name="alias-transfer-oracle",
        table_prefix="alias_transfer_oracle_matrix",
    )

    assert payload["manifest"]["comparison_count"] == 5
    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["matrix_name"] == "alias-transfer-oracle"
    summary = {row["label"]: row for row in payload["summary_rows"]}
    assert summary["contracted"]["candidate_exact_rate"] == 0.16666666666666666
    assert summary["contracted"]["delta_executor_equivalence_rate"] == -0.16666666666666666
    assert summary["argument_hints_v2"]["candidate_exact_rate"] == 0.8333333333333334
    assert summary["argument_hints_v2"]["candidate_executor_equivalence_rate"] == 1.0
    assert summary["schema_literal_targets_v5"]["candidate_executor_equivalence_rate"] == 0.6666666666666666
    transitions = {(row["label"], row["case_id"]): row for row in payload["case_rows"]}
    assert transitions[
        ("argument_hints_v2", "transfer_review_tile_notice_table_decoy")
    ]["transition"] == "strict_gain"
    assert transitions[
        ("contracted", "transfer_error_banner_note_decoy")
    ]["transition"] == "regression"
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "argument_hints_v2" in findings["strict_upper_bound"]
    assert "argument_hints_v2" in findings["executor_equivalence_set"]
    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "alias_transfer_oracle_matrix_case_transitions.csv").exists()


def test_visual_alias_transfer_repeat_diagnostic_writes_findings(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_visual_live_stress_matrix(
        output_dir=tmp_path,
        comparisons=SCRIPT.DEFAULT_ALIAS_TRANSFER_REPEAT_COMPARISONS,
        matrix_name="alias-transfer-repeat",
        table_prefix="alias_transfer_repeat_matrix",
    )

    assert payload["manifest"]["comparison_count"] == 5
    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["matrix_name"] == "alias-transfer-repeat"
    summary = {row["label"]: row for row in payload["summary_rows"]}
    assert summary["contracted"]["candidate_exact_rate"] == 0.0
    assert summary["contracted"]["delta_executor_equivalence_rate"] == -0.3333333333333333
    assert summary["argument_hints_v2"]["candidate_exact_rate"] == 0.8333333333333334
    assert summary["argument_hints_v2"]["candidate_executor_equivalence_rate"] == 1.0
    assert summary["schema_literal_targets_v5"]["candidate_executor_equivalence_rate"] == 1.0
    transitions = {(row["label"], row["case_id"]): row for row in payload["case_rows"]}
    assert transitions[
        ("argument_hints_v2", "transfer_repeat_latency_chip_person_decoy")
    ]["transition"] == "strict_gain"
    assert transitions[
        ("contracted", "transfer_repeat_audit_card_email_decoy")
    ]["transition"] == "regression"
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "argument_hints_v2" in findings["strict_upper_bound"]
    assert "schema_literal_targets_v5" in findings["executor_equivalence_set"]
    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "alias_transfer_repeat_matrix_case_transitions.csv").exists()


def test_visual_alias_transfer_oblique_diagnostic_writes_findings(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_visual_live_stress_matrix(
        output_dir=tmp_path,
        comparisons=SCRIPT.DEFAULT_ALIAS_TRANSFER_OBLIQUE_COMPARISONS,
        matrix_name="alias-transfer-oblique",
        table_prefix="alias_transfer_oblique_matrix",
    )

    assert payload["manifest"]["comparison_count"] == 7
    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["matrix_name"] == "alias-transfer-oblique"
    summary = {row["label"]: row for row in payload["summary_rows"]}
    assert summary["contracted"]["candidate_exact_rate"] == 0.16666666666666666
    assert summary["argument_hints_v2"]["candidate_exact_rate"] == 0.6666666666666666
    assert summary["argument_hints_v2"]["candidate_executor_equivalence_rate"] == 0.6666666666666666
    assert summary["schema_field_hints_v4"]["candidate_exact_rate"] == 0.5
    assert summary["schema_literal_targets_v5"]["candidate_executor_equivalence_rate"] == 0.0
    assert summary["oblique_code_hints_v6"]["candidate_exact_rate"] == 0.8333333333333334
    assert summary["oblique_code_hints_v6"]["candidate_executor_equivalence_rate"] == 0.8333333333333334
    assert summary["oblique_code_guard_v7"]["candidate_exact_rate"] == 1.0
    assert summary["oblique_code_guard_v7"]["candidate_executor_equivalence_rate"] == 1.0
    transitions = {(row["label"], row["case_id"]): row for row in payload["case_rows"]}
    assert transitions[
        ("argument_hints_v2", "transfer_oblique_badge_m88_chart_decoy")
    ]["transition"] == "strict_gain"
    assert transitions[
        ("schema_literal_targets_v5", "transfer_oblique_node_q17_table_decoy")
    ]["transition"] == "unchanged"
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "oblique_code_guard_v7" in findings["strict_upper_bound"]
    assert "oblique_code_guard_v7" in findings["executor_equivalence_set"]
    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "alias_transfer_oblique_matrix_case_transitions.csv").exists()


def test_visual_alias_transfer_post_repair_diagnostic_writes_findings(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_visual_live_stress_matrix(
        output_dir=tmp_path,
        comparisons=SCRIPT.DEFAULT_ALIAS_TRANSFER_POST_REPAIR_COMPARISONS,
        matrix_name="alias-transfer-post-repair",
        table_prefix="alias_transfer_post_repair_matrix",
    )

    assert payload["manifest"]["comparison_count"] == 4
    assert payload["manifest"]["case_count"] == 8
    assert payload["manifest"]["matrix_name"] == "alias-transfer-post-repair"
    summary = {row["label"]: row for row in payload["summary_rows"]}
    assert summary["contracted"]["candidate_exact_rate"] == 0.375
    assert summary["argument_hints_v2"]["candidate_exact_rate"] == 0.625
    assert summary["oblique_code_hints_v6"]["candidate_exact_rate"] == 0.625
    assert summary["oblique_code_guard_v7"]["candidate_exact_rate"] == 0.75
    assert summary["oblique_code_guard_v7"]["delta_executor_equivalence_rate"] == 0.5
    transitions = {(row["label"], row["case_id"]): row for row in payload["case_rows"]}
    assert transitions[
        ("oblique_code_guard_v7", "post_repair_badge_t64_notice_decoy")
    ]["transition"] == "strict_gain"
    assert transitions[
        ("argument_hints_v2", "post_repair_status_pill_note_decoy")
    ]["transition"] == "strict_gain"
    assert transitions[
        ("oblique_code_hints_v6", "post_repair_review_tile_table_decoy")
    ]["transition"] == "regression"
    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "oblique_code_guard_v7" in findings["strict_upper_bound"]
    assert "Executor-equivalent full-success rows: none." in findings["executor_equivalence_set"]
    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "alias_transfer_post_repair_matrix_case_transitions.csv").exists()
