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
    assert checks["v3_skipped_live_decision_exists"]["passed"] is True
    assert checks["paper_outline_exists"]["severity"] == "recommended"

    assert (tmp_path / "publication_readiness_audit.md").exists()
    assert (tmp_path / "publication_readiness_audit.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "publication_readiness_checks.csv").exists()
