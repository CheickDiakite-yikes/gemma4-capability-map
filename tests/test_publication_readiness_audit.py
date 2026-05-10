from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_publication_readiness.py"
SPEC = importlib.util.spec_from_file_location("audit_publication_readiness_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_publication_readiness_audit_writes_blocking_checks(tmp_path: Path) -> None:
    payload = SCRIPT.audit_publication_readiness(output_dir=tmp_path)

    summary = payload["summary"]
    assert summary["blocking_passed"] is True
    assert summary["readiness_level"] == "paper_draft_ready"
    assert summary["blocking_failed_count"] == 0

    checks = {row["check_id"]: row for row in payload["checks"]}
    assert checks["ledger_has_no_missing_sources"]["passed"] is True
    assert checks["ledger_includes_negative_results"]["passed"] is True
    assert checks["visual_hard_slice_design_exists"]["passed"] is True
    assert checks["visual_hard_slice_execute_packet_exists"]["passed"] is True
    assert checks["visual_hard_slice_v5_vs_v4_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_exactness_diagnostic_exists"]["passed"] is True
    assert checks["h1l_visual_executor_equivalence_packet_exists"]["passed"] is True
    assert checks["h1m_visual_alias_repeat_packet_exists"]["passed"] is True
    assert checks["visual_hard_slice_live_replay_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_live_contracted_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_live_role_catalog_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_live_argument_hints_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_live_schema_literals_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_stress_packet_exists"]["passed"] is True
    assert checks["visual_hard_slice_stress_schema_field_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_stress_report_table_exists"]["passed"] is True
    assert checks["visual_live_stress_diagnostic_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_repeat_packet_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_repeat_schema_field_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_repeat_contracted_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_repeat_role_catalog_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_repeat_argument_hints_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_repeat_schema_literals_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_repeat_report_table_exists"]["passed"] is True
    assert checks["visual_alias_repeat_diagnostic_exists"]["passed"] is True
    assert checks["h1m_visual_alias_repeat_report_table_exists"]["passed"] is True
    assert checks["packaged_replay_gap_diagnostic_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_transfer_packet_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_transfer_argument_hints_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_transfer_contracted_comparison_exists"]["passed"] is True
    assert checks["visual_hard_slice_alias_transfer_report_table_exists"]["passed"] is True
    assert checks["visual_alias_transfer_diagnostic_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_contract_split_diagnostic_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_oracle_packet_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_oracle_argument_hints_comparison_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_oracle_diagnostic_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_oracle_report_table_exists"]["passed"] is True
    assert checks["h1n_oracle_helper_ablation_diagnostic_exists"]["passed"] is True
    assert checks["h1n_oracle_helper_ablation_no_repair_comparison_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_repeat_packet_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_repeat_diagnostic_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_repeat_argument_hints_comparison_exists"]["passed"] is True
    assert checks["h1n_oracle_transfer_synthesis_report_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_oblique_packet_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_oblique_diagnostic_exists"]["passed"] is True
    assert checks["h1n_alias_transfer_oblique_argument_hints_comparison_exists"]["passed"] is True
    assert checks["h1n_oblique_miss_analysis_exists"]["passed"] is True
    assert checks["h1n_oblique_code_hints_packet_exists"]["passed"] is True
    assert checks["h1n_oblique_code_hints_vs_argument_hints_comparison_exists"]["passed"] is True
    assert checks["h1n_oblique_code_hints_delta_diagnostic_exists"]["passed"] is True
    assert checks["h1n_code_hints_transfer_packet_exists"]["passed"] is True
    assert checks["h1n_code_hints_repeat_transfer_packet_exists"]["passed"] is True
    assert checks["h1n_code_hints_transfer_synthesis_exists"]["passed"] is True
    assert checks["script_analyze_visual_hard_slice_exactness.py_exists"]["passed"] is True
    assert checks["script_analyze_packaged_replay_gap.py_exists"]["passed"] is True
    assert checks["script_analyze_h1n_alias_transfer_contract_split.py_exists"]["passed"] is True
    assert checks["script_analyze_h1n_oracle_helper_ablation.py_exists"]["passed"] is True
    assert checks["script_analyze_h1n_oblique_misses.py_exists"]["passed"] is True
    assert checks["script_analyze_h1n_oblique_code_hints_delta.py_exists"]["passed"] is True
    assert checks["script_build_h1n_code_hints_transfer_synthesis.py_exists"]["passed"] is True
    assert checks["script_build_h1n_oracle_transfer_synthesis.py_exists"]["passed"] is True
    assert checks["script_run_visual_hard_slice_probe_packet.py_exists"]["passed"] is True
    assert checks["script_build_visual_hard_slice_replay_packet.py_exists"]["passed"] is True
    assert checks["v3_skipped_live_decision_exists"]["passed"] is True
    assert checks["paper_outline_exists"]["severity"] == "recommended"

    assert (tmp_path / "publication_readiness_audit.md").exists()
    assert (tmp_path / "publication_readiness_audit.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "publication_readiness_checks.csv").exists()
