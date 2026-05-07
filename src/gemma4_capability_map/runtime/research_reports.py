from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIRS = {
    "mlx-tool-contract": ROOT / "results" / "reports" / "mlx_tool_contract_harnessing",
    "mlx_tool_contract_harnessing": ROOT / "results" / "reports" / "mlx_tool_contract_harnessing",
}


def research_report_payload(*, report_id: str = "mlx-tool-contract", report_dir: str | Path | None = None) -> dict[str, Any]:
    target = _resolve_report_dir(report_id=report_id, report_dir=report_dir)
    manifest = _read_json(target / "manifest.json")
    report = _read_json(target / "report.json")
    tables = _artifact_rows(target / "tables")
    figures = _artifact_rows(target / "figures")
    prompt_candidates = list(report.get("prompt_contract_candidates") or [])
    packet_summary = list(report.get("packet_summary") or [])
    gemini = dict(report.get("gemini") or {})
    return {
        "report_id": report_id,
        "report_dir": str(target.resolve()),
        "exists": target.exists(),
        "report_md": _file_payload(target / "report.md"),
        "manifest": manifest,
        "packet_count": len(packet_summary),
        "prompt_contract_candidate_count": len(prompt_candidates),
        "prompt_contract_candidate_ids": [row.get("tool_prompt_contract_id", "") for row in prompt_candidates],
        "gemini_packet_run_id": gemini.get("packet_run_id", ""),
        "gemini_dry_run": gemini.get("dry_run", None),
        "tables": tables,
        "figures": figures,
    }


def print_research_report(payload: dict[str, Any], *, console: Console | None = None) -> None:
    target_console = console or Console()
    target_console.print(_research_report_renderable(payload))


def _resolve_report_dir(*, report_id: str, report_dir: str | Path | None) -> Path:
    if report_dir:
        return Path(report_dir)
    if report_id not in DEFAULT_REPORT_DIRS:
        known = ", ".join(sorted(DEFAULT_REPORT_DIRS))
        raise ValueError(f"Unknown report id `{report_id}`. Known reports: {known}.")
    return DEFAULT_REPORT_DIRS[report_id]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _file_payload(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _artifact_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [_file_payload(child) for child in sorted(path.iterdir()) if child.is_file()]


def _research_report_renderable(payload: dict[str, Any]) -> Group:
    manifest = payload.get("manifest") or {}
    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("Report", str(payload.get("report_id", "")))
    header.add_row("Path", str(payload.get("report_dir", "")))
    header.add_row("Generated", str(manifest.get("generated_at", "")))
    header.add_row("Packets", str(payload.get("packet_count", 0)))
    header.add_row("Prompt contracts", str(payload.get("prompt_contract_candidate_count", 0)))
    if payload.get("gemini_packet_run_id"):
        header.add_row("Gemini baseline", f"{payload['gemini_packet_run_id']} (dry_run={payload.get('gemini_dry_run')})")

    files = Table(title="Artifacts")
    files.add_column("Kind")
    files.add_column("Count", justify="right")
    files.add_column("Primary")
    report_md = payload.get("report_md") or {}
    files.add_row("report", "1" if report_md.get("exists") else "0", str(report_md.get("path", "")))
    files.add_row("tables", str(len(payload.get("tables") or [])), _first_path(payload.get("tables") or []))
    files.add_row("figures", str(len(payload.get("figures") or [])), _first_path(payload.get("figures") or []))

    candidate_ids = ", ".join(str(value) for value in payload.get("prompt_contract_candidate_ids") or [] if value)
    candidate_panel = Panel(candidate_ids or "No prompt-contract candidates recorded.", title="Prompt-Contract Candidates")
    return Group(Panel(header, title="Moonie Research Report"), files, candidate_panel)


def _first_path(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return str(rows[0].get("path", ""))
