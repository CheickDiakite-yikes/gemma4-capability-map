from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_DIR = ROOT / "results" / "reports" / "publication_evidence_ledger"
DEFAULT_TOOL_CONTRACT_REPORT_DIR = ROOT / "results" / "reports" / "mlx_tool_contract_harnessing"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "publication_readiness_audit"


def audit_publication_readiness(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ledger_dir: str | Path = DEFAULT_LEDGER_DIR,
    tool_contract_report_dir: str | Path = DEFAULT_TOOL_CONTRACT_REPORT_DIR,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ledger_dir = Path(ledger_dir)
    tool_contract_report_dir = Path(tool_contract_report_dir)
    ledger_payload = _read_json(ledger_dir / "ledger.json")
    ledger_manifest = ledger_payload.get("manifest", {}) if isinstance(ledger_payload, dict) else {}
    report_manifest = _read_json(tool_contract_report_dir / "manifest.json")

    checks = [
        _check_path(
            check_id="ledger_manifest_exists",
            severity="blocking",
            path=ledger_dir / "manifest.json",
            detail="Publication evidence ledger manifest exists.",
        ),
        _check_bool(
            check_id="ledger_has_no_missing_sources",
            severity="blocking",
            passed=int(ledger_manifest.get("missing_source_count", 1) or 0) == 0,
            detail=f"missing_source_count={ledger_manifest.get('missing_source_count', '')}",
        ),
        _check_bool(
            check_id="ledger_has_claims",
            severity="blocking",
            passed=int(ledger_manifest.get("claim_count", 0) or 0) >= 6,
            detail=f"claim_count={ledger_manifest.get('claim_count', '')}",
        ),
        _check_bool(
            check_id="ledger_includes_negative_results",
            severity="blocking",
            passed=_has_negative_result_claim(ledger_payload),
            detail="At least one claim is explicitly labeled as negative-result evidence.",
        ),
        _check_bool(
            check_id="tool_contract_report_has_current_tables",
            severity="blocking",
            passed=int(report_manifest.get("table_count", 0) or 0) >= 54,
            detail=f"table_count={report_manifest.get('table_count', '')}",
        ),
        _check_bool(
            check_id="tool_contract_report_has_current_figures",
            severity="blocking",
            passed=int(report_manifest.get("figure_count", 0) or 0) >= 26,
            detail=f"figure_count={report_manifest.get('figure_count', '')}",
        ),
        _check_path(
            check_id="v3_negative_probe_packet_exists",
            severity="blocking",
            path=ROOT / "results" / "tool_catalog_profile_probe_packets" / "20260508T_visual_role_catalog_split_selector_hints_v3_probe",
            detail="Negative v3 catalog-profile probe is preserved.",
        ),
        _check_path(
            check_id="v3_skipped_live_decision_exists",
            severity="blocking",
            path=ROOT / "results" / "tool_probe_replay_live" / "20260508T_visual_split_selector_hints_live_replay_skipped_v1" / "decision.md",
            detail="Skipped-live decision is preserved as an auditable packet.",
        ),
        _check_path(
            check_id="v4_negative_probe_packet_exists",
            severity="blocking",
            path=ROOT / "results" / "tool_catalog_profile_probe_packets" / "20260509T_visual_role_catalog_schema_field_hints_v4_probe",
            detail="Negative v4 schema-field probe is preserved.",
        ),
        _check_path(
            check_id="v4_skipped_live_decision_exists",
            severity="blocking",
            path=ROOT / "results" / "tool_probe_replay_live" / "20260509T_visual_schema_field_hints_live_replay_skipped_v1" / "decision.md",
            detail="Skipped-live decision is preserved as an auditable packet.",
        ),
        _check_path(
            check_id="visual_hard_slice_design_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_hard_slice_design" / "design.json",
            detail="Fresh visual hard-slice design packet exists before new benchmark execution.",
        ),
        _check_path(
            check_id="visual_hard_slice_execute_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "visual_hard_slice_probe_packets"
            / "20260509T_visual_hard_slice_execute_v1"
            / "candidate_gate_summary.md",
            detail="Executed visual hard-slice packet exists with candidate gate summary.",
        ),
        _check_path(
            check_id="current_state_doc_exists",
            severity="blocking",
            path=ROOT / "docs" / "continuity" / "current-state.md",
            detail="Continuity current-state doc exists.",
        ),
        _check_path(
            check_id="next_steps_doc_exists",
            severity="blocking",
            path=ROOT / "docs" / "continuity" / "next-steps.md",
            detail="Continuity next-steps doc exists.",
        ),
        _check_path(
            check_id="research_log_exists",
            severity="blocking",
            path=ROOT / "docs" / "research-log.md",
            detail="Research log exists.",
        ),
        _check_path(
            check_id="paper_outline_exists",
            severity="recommended",
            path=ROOT / "docs" / "paper" / "moonie-gemma-harnessing-paper-outline.md",
            detail="Paper outline exists for publication drafting.",
        ),
        _check_path(
            check_id="methodology_doc_exists",
            severity="recommended",
            path=ROOT / "docs" / "methodology.md",
            detail="Methodology doc exists.",
        ),
    ]
    for script_name in [
        "build_mlx_tool_contract_report.py",
        "build_publication_evidence_ledger.py",
        "audit_publication_readiness.py",
        "run_tool_catalog_profile_probe_packet.py",
        "run_visual_hard_slice_probe.py",
        "run_visual_hard_slice_probe_packet.py",
        "compare_tool_directive_probes.py",
        "build_visual_hard_slice_design.py",
    ]:
        checks.append(
            _check_path(
                check_id=f"script_{script_name}_exists",
                severity="blocking",
                path=ROOT / "scripts" / script_name,
                detail=f"Reproduction script `{script_name}` exists.",
            )
        )

    blocking_checks = [row for row in checks if row["severity"] == "blocking"]
    recommended_checks = [row for row in checks if row["severity"] == "recommended"]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "check_count": len(checks),
        "blocking_check_count": len(blocking_checks),
        "blocking_failed_count": sum(1 for row in blocking_checks if not row["passed"]),
        "recommended_check_count": len(recommended_checks),
        "recommended_failed_count": sum(1 for row in recommended_checks if not row["passed"]),
    }
    summary["blocking_passed"] = summary["blocking_failed_count"] == 0
    summary["readiness_level"] = "paper_draft_ready" if summary["blocking_passed"] else "not_ready"

    _write_csv(tables_dir / "publication_readiness_checks.csv", checks)
    payload = {"summary": summary, "checks": checks}
    (output / "publication_readiness_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "publication_readiness_audit.md").write_text(_markdown(summary, checks), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether Moonie has enough packet-backed evidence for a paper draft.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ledger-dir", default=str(DEFAULT_LEDGER_DIR))
    parser.add_argument("--tool-contract-report-dir", default=str(DEFAULT_TOOL_CONTRACT_REPORT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_publication_readiness(
        output_dir=args.output_dir,
        ledger_dir=args.ledger_dir,
        tool_contract_report_dir=args.tool_contract_report_dir,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _has_negative_result_claim(ledger_payload: dict[str, Any]) -> bool:
    claims = ledger_payload.get("claims", []) if isinstance(ledger_payload, dict) else []
    return any(str(row.get("status", "")).startswith("negative_result") for row in claims if isinstance(row, dict))


def _check_path(*, check_id: str, severity: str, path: Path, detail: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "passed": path.exists(),
        "detail": detail,
        "path": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
    }


def _check_bool(*, check_id: str, severity: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "passed": passed,
        "detail": detail,
        "path": "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any], checks: list[dict[str, Any]]) -> str:
    lines = [
        "# Publication Readiness Audit",
        "",
        f"- readiness_level: `{summary['readiness_level']}`",
        f"- blocking_passed: `{summary['blocking_passed']}`",
        f"- blocking_failed_count: `{summary['blocking_failed_count']}`",
        f"- recommended_failed_count: `{summary['recommended_failed_count']}`",
        "",
        "| Check | Severity | Passed | Detail | Path |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in checks:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["check_id"]),
                    str(row["severity"]),
                    str(row["passed"]),
                    str(row["detail"]),
                    str(row["path"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
