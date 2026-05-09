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
    assert checks["script_analyze_visual_hard_slice_exactness.py_exists"]["passed"] is True
    assert checks["script_run_visual_hard_slice_probe_packet.py_exists"]["passed"] is True
    assert checks["script_build_visual_hard_slice_replay_packet.py_exists"]["passed"] is True
    assert checks["v3_skipped_live_decision_exists"]["passed"] is True
    assert checks["paper_outline_exists"]["severity"] == "recommended"

    assert (tmp_path / "publication_readiness_audit.md").exists()
    assert (tmp_path / "publication_readiness_audit.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "publication_readiness_checks.csv").exists()
