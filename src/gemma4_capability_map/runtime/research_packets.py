from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PACKET_ROOTS = {
    "prompt-contract-probe": ROOT / "results" / "tool_prompt_contract_probe_packets",
    "tool-probe-replay": ROOT / "results" / "tool_probe_replay_packets",
    "tool-probe-replay-live": ROOT / "results" / "tool_probe_replay_live",
}


def research_packet_payload(
    *,
    packet_kind: str = "prompt-contract-probe",
    packet_id: str = "latest",
    packet_dir: str | Path | None = None,
) -> dict[str, Any]:
    target = _resolve_packet_dir(packet_kind=packet_kind, packet_id=packet_id, packet_dir=packet_dir)
    if packet_kind == "tool-probe-replay":
        return _tool_probe_replay_payload(packet_kind=packet_kind, target=target)
    if packet_kind == "tool-probe-replay-live":
        return _tool_probe_replay_live_payload(packet_kind=packet_kind, target=target)
    manifest = _read_json(target / "manifest.json")
    commands = _read_json(target / "commands.json")
    results = _read_json(target / "results.json")
    candidates = _read_csv(target / "candidate_summary.csv")
    return {
        "packet_kind": packet_kind,
        "packet_id": target.name,
        "packet_dir": str(target.resolve()),
        "exists": target.exists(),
        "manifest": manifest,
        "candidate_count": _count_or_default(results, "candidate_count", len(candidates)),
        "executed_count": _count_or_default(results, "executed_count", 0),
        "dry_run_count": _count_or_default(results, "dry_run_count", len(candidates)),
        "command_count": len(commands if isinstance(commands, list) else []),
        "candidate_rows": candidates,
        "files": [_file_payload(child) for child in sorted(target.iterdir()) if child.is_file()] if target.exists() else [],
}


def _tool_probe_replay_payload(*, packet_kind: str, target: Path) -> dict[str, Any]:
    manifest = _read_json(target / "manifest.json")
    summary = _read_json(target / "summary.json")
    commands = _read_json(target / "commands.json")
    replay_rows = _read_csv(target / "replay_cases.csv")
    next_action_rows = _read_csv(target / "replay_next_actions.csv")
    replay_result_rows = _read_csv(target / "replay_results.csv")
    return {
        "packet_kind": packet_kind,
        "packet_id": target.name,
        "packet_dir": str(target.resolve()),
        "exists": target.exists(),
        "manifest": manifest,
        "summary": summary,
        "case_count": _count_or_default(summary, "case_count", len(replay_rows)),
        "dry_run": bool(summary.get("dry_run", False)) if isinstance(summary, dict) else False,
        "command_count": len(commands if isinstance(commands, list) else []),
        "failure_mode_counts": summary.get("failure_mode_counts", {}) if isinstance(summary, dict) else {},
        "family_counts": summary.get("family_counts", {}) if isinstance(summary, dict) else {},
        "replay_case_rows": replay_rows,
        "next_action_rows": next_action_rows,
        "replay_result_rows": replay_result_rows,
        "files": [_file_payload(child) for child in sorted(target.iterdir()) if child.is_file()] if target.exists() else [],
    }


def _tool_probe_replay_live_payload(*, packet_kind: str, target: Path) -> dict[str, Any]:
    manifest = _read_json(target / "manifest.json")
    summary = _read_json(target / "summary.json")
    commands = _read_json(target / "commands.json")
    case_state_rows = _read_csv(target / "live_case_states.csv")
    result_rows = _read_csv(target / "live_replay_results.csv")
    return {
        "packet_kind": packet_kind,
        "packet_id": target.name,
        "packet_dir": str(target.resolve()),
        "exists": target.exists(),
        "manifest": manifest,
        "summary": summary,
        "case_count": _count_or_default(summary, "case_count", len(case_state_rows)),
        "execute": bool(summary.get("execute", False)) if isinstance(summary, dict) else False,
        "executed_count": _count_or_default(summary, "executed_count", len(result_rows)),
        "exact_count": _count_or_default(summary, "exact_count", 0),
        "exact_rate": float(summary.get("exact_rate") or 0.0) if isinstance(summary, dict) else 0.0,
        "command_count": len(commands if isinstance(commands, list) else []),
        "failure_mode_counts": summary.get("failure_mode_counts", {}) if isinstance(summary, dict) else {},
        "case_state_rows": case_state_rows,
        "result_rows": result_rows,
        "files": [_file_payload(child) for child in sorted(target.iterdir()) if child.is_file()] if target.exists() else [],
    }


def print_research_packet(payload: dict[str, Any], *, console: Console | None = None) -> None:
    target_console = console or Console()
    target_console.print(_research_packet_renderable(payload))


def _resolve_packet_dir(*, packet_kind: str, packet_id: str, packet_dir: str | Path | None) -> Path:
    if packet_dir:
        return Path(packet_dir)
    if packet_kind not in DEFAULT_PACKET_ROOTS:
        known = ", ".join(sorted(DEFAULT_PACKET_ROOTS))
        raise ValueError(f"Unknown packet kind `{packet_kind}`. Known packet kinds: {known}.")
    root = DEFAULT_PACKET_ROOTS[packet_kind]
    if packet_id != "latest":
        return root / packet_id
    candidates = sorted(child for child in root.iterdir() if child.is_dir()) if root.exists() else []
    if not candidates:
        raise ValueError(f"No packet directories found under `{root}`.")
    return candidates[-1]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {} if path.name == "manifest.json" else []
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_payload(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _count_or_default(payload: Any, key: str, default: int) -> int:
    if not isinstance(payload, dict):
        return default
    return int(payload.get(key, default) or 0)


def _research_packet_renderable(payload: dict[str, Any]) -> Group:
    manifest = payload.get("manifest") or {}
    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("Kind", str(payload.get("packet_kind", "")))
    header.add_row("Packet", str(payload.get("packet_id", "")))
    header.add_row("Path", str(payload.get("packet_dir", "")))
    header.add_row("Created", str(manifest.get("created_at", "")))
    if payload.get("packet_kind") == "tool-probe-replay":
        header.add_row("Dry run", str(payload.get("dry_run", "")))
        header.add_row("Cases", str(payload.get("case_count", 0)))
        header.add_row("Commands", str(payload.get("command_count", 0)))
    elif payload.get("packet_kind") == "tool-probe-replay-live":
        header.add_row("Execute", str(payload.get("execute", "")))
        header.add_row("Cases", str(payload.get("case_count", 0)))
        header.add_row("Executed", str(payload.get("executed_count", 0)))
        header.add_row("Exact rate", str(payload.get("exact_rate", 0.0)))
    else:
        header.add_row("Execute", str(manifest.get("execute", "")))
        header.add_row("Candidates", str(payload.get("candidate_count", 0)))
        header.add_row("Executed", str(payload.get("executed_count", 0)))
        header.add_row("Dry run", str(payload.get("dry_run_count", 0)))

    candidates = Table(title="Candidates")
    candidates.add_column("System")
    candidates.add_column("Contract")
    candidates.add_column("Execute")
    candidates.add_column("Exact")
    candidates.add_column("Executable")
    for row in payload.get("candidate_rows") or []:
        candidates.add_row(
            str(row.get("system_id", "")),
            str(row.get("tool_prompt_contract_id", "")),
            str(row.get("execute", "")),
            str(row.get("exact_match_rate", "")),
            str(row.get("executable_match_rate", "")),
        )

    replay_cases = Table(title="Replay Cases")
    replay_cases.add_column("Case")
    replay_cases.add_column("Family")
    replay_cases.add_column("Failure")
    replay_cases.add_column("Baseline exact")
    for row in payload.get("replay_case_rows") or []:
        replay_cases.add_row(
            str(row.get("case_id", "")),
            str(row.get("family", "")),
            str(row.get("source_failure_mode", "")),
            str(row.get("baseline_exact_match", "")),
        )

    next_actions = Table(title="Next Actions")
    next_actions.add_column("Case")
    next_actions.add_column("Priority")
    next_actions.add_column("Action")
    for row in payload.get("next_action_rows") or []:
        next_actions.add_row(
            str(row.get("case_id", "")),
            str(row.get("priority", "")),
            str(row.get("next_action", "")),
        )

    replay_results = Table(title="Replay Results")
    replay_results.add_column("Case")
    replay_results.add_column("Replay failure")
    replay_results.add_column("Exact")
    replay_results.add_column("Executable")
    for row in payload.get("replay_result_rows") or []:
        replay_results.add_row(
            str(row.get("case_id", "")),
            str(row.get("replay_failure_mode", "")),
            str(row.get("replay_exact_match", "")),
            str(row.get("replay_executable_match", "")),
        )

    live_cases = Table(title="Live Replay Cases")
    live_cases.add_column("Case")
    live_cases.add_column("Family")
    live_cases.add_column("Source failure")
    live_cases.add_column("Status")
    live_cases.add_column("Replay failure")
    live_cases.add_column("Exact")
    for row in payload.get("case_state_rows") or []:
        live_cases.add_row(
            str(row.get("case_id", "")),
            str(row.get("family", "")),
            str(row.get("source_failure_mode", "")),
            str(row.get("status", "")),
            str(row.get("replay_failure_mode", "")),
            str(row.get("replay_exact_match", "")),
        )

    files = Table(title="Files")
    files.add_column("File")
    files.add_column("Bytes", justify="right")
    for file_row in payload.get("files") or []:
        files.add_row(Path(str(file_row.get("path", ""))).name, str(file_row.get("size_bytes", 0)))
    body = [Panel(header, title="Moonie Research Packet")]
    if payload.get("packet_kind") == "tool-probe-replay":
        body.append(replay_cases)
        if payload.get("next_action_rows"):
            body.append(next_actions)
        if payload.get("replay_result_rows"):
            body.append(replay_results)
    elif payload.get("packet_kind") == "tool-probe-replay-live":
        body.append(live_cases)
    else:
        body.append(candidates)
    body.append(files)
    return Group(*body)
