from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH
from gemma4_capability_map.runtime.tool_directive_probe import (
    ToolDirectiveProbeCase,
    build_tool_directive_probe_cases,
    run_tool_directive_probe,
)
from gemma4_capability_map.schemas import Message


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLAY_PACKET_ROOT = ROOT / "results" / "tool_probe_replay_packets"
DEFAULT_LIVE_REPLAY_ROOT = ROOT / "results" / "tool_probe_replay_live"


def run_tool_probe_replay_live(
    *,
    packet_id: str = "latest",
    packet_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    system_id: str = "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    case_ids: list[str] | None = None,
    execute: bool = False,
    render: bool = True,
    refresh_s: float = 0.5,
    console: Console | None = None,
) -> dict[str, Any]:
    source_packet = _resolve_packet_dir(packet_id=packet_id, packet_dir=packet_dir)
    source_manifest = _read_json(source_packet / "manifest.json")
    source_rows = _read_csv(source_packet / "replay_cases.csv")
    selected_rows = _select_rows(source_rows, case_ids or [])
    cases_by_id = _load_cases_by_id(source_packet)
    missing = [row["case_id"] for row in selected_rows if row["case_id"] not in cases_by_id]
    if missing:
        raise ValueError(f"Replay packet references unknown case id(s): {', '.join(missing)}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = Path(output_dir) if output_dir else DEFAULT_LIVE_REPLAY_ROOT / f"{timestamp}_{source_packet.name}"
    target.mkdir(parents=True, exist_ok=True)

    case_states = [_case_state(row, status="queued") for row in selected_rows]
    commands = [
        {
            "case_id": row["case_id"],
            "command": _case_command(
                source_packet=source_packet,
                output_dir=target,
                system_id=system_id,
                registry_path=Path(registry_path),
                case_id=row["case_id"],
            ),
        }
        for row in selected_rows
    ]

    target_console = console or Console()
    results: list[dict[str, Any]] = []
    if execute:
        if render:
            with Live(
                _render_live_replay(
                    source_packet=source_packet,
                    target=target,
                    system_id=system_id,
                    execute=execute,
                    case_states=case_states,
                ),
                console=target_console,
                refresh_per_second=max(1, int(1 / refresh_s)) if refresh_s > 0 else 4,
            ) as live:
                for index, state in enumerate(case_states):
                    state["status"] = "running"
                    live.update(
                        _render_live_replay(
                            source_packet=source_packet,
                            target=target,
                            system_id=system_id,
                            execute=execute,
                            case_states=case_states,
                        )
                    )
                    result = _execute_case(
                        case=cases_by_id[state["case_id"]],
                        output_dir=target / "runs" / state["case_id"],
                        system_id=system_id,
                        registry_path=Path(registry_path),
                        source_failure_mode=state["source_failure_mode"],
                    )
                    results.append(result)
                    case_states[index].update(
                        {
                            "status": "exact" if result["replay_exact_match"] else "non_exact",
                            "replay_failure_mode": result["replay_failure_mode"],
                            "replay_exact_match": result["replay_exact_match"],
                            "replay_executable_match": result["replay_executable_match"],
                            "replay_executor_equivalence_match": result["replay_executor_equivalence_match"],
                        }
                    )
                    _write_outputs(
                        target=target,
                        source_packet=source_packet,
                        source_manifest=source_manifest,
                        system_id=system_id,
                        registry_path=Path(registry_path),
                        execute=execute,
                        case_states=case_states,
                        commands=commands,
                        results=results,
                    )
                    live.update(
                        _render_live_replay(
                            source_packet=source_packet,
                            target=target,
                            system_id=system_id,
                            execute=execute,
                            case_states=case_states,
                        )
                    )
        else:
            for index, state in enumerate(case_states):
                case_states[index]["status"] = "running"
                result = _execute_case(
                    case=cases_by_id[state["case_id"]],
                    output_dir=target / "runs" / state["case_id"],
                    system_id=system_id,
                    registry_path=Path(registry_path),
                    source_failure_mode=state["source_failure_mode"],
                )
                results.append(result)
                case_states[index].update(
                    {
                        "status": "exact" if result["replay_exact_match"] else "non_exact",
                        "replay_failure_mode": result["replay_failure_mode"],
                        "replay_exact_match": result["replay_exact_match"],
                        "replay_executable_match": result["replay_executable_match"],
                        "replay_executor_equivalence_match": result["replay_executor_equivalence_match"],
                    }
                )
    else:
        for state in case_states:
            state["status"] = "dry_run"
        if render:
            target_console.print(
                _render_live_replay(
                    source_packet=source_packet,
                    target=target,
                    system_id=system_id,
                    execute=execute,
                    case_states=case_states,
                )
            )

    payload = _write_outputs(
        target=target,
        source_packet=source_packet,
        source_manifest=source_manifest,
        system_id=system_id,
        registry_path=Path(registry_path),
        execute=execute,
        case_states=case_states,
        commands=commands,
        results=results,
    )
    return payload


def _execute_case(
    *,
    case: ToolDirectiveProbeCase,
    output_dir: Path,
    system_id: str,
    registry_path: Path,
    source_failure_mode: str,
) -> dict[str, Any]:
    probe = run_tool_directive_probe(
        system_id=system_id,
        output_dir=output_dir,
        registry_path=registry_path,
        cases=[case],
    )
    row = probe["rows"][0]
    return {
        "case_id": case.case_id,
        "family": case.family,
        "source_failure_mode": source_failure_mode,
        "replay_failure_mode": _failure_mode(row),
        "replay_exact_match": bool(row.get("exact_match")),
        "replay_executable_match": row.get("executable_match"),
        "replay_executor_equivalence_match": row.get("executor_target_match"),
        "expected_call_count": int(row.get("expected_call_count") or 0),
        "replay_actual_call_count": int(row.get("actual_call_count") or 0),
        "output_dir": str(output_dir.resolve()),
    }


def _write_outputs(
    *,
    target: Path,
    source_packet: Path,
    source_manifest: dict[str, Any],
    system_id: str,
    registry_path: Path,
    execute: bool,
    case_states: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    executor_rows = [row for row in results if row.get("replay_executor_equivalence_match") is not None]
    executor_count = sum(1 for row in executor_rows if row.get("replay_executor_equivalence_match"))
    summary = {
        "packet_run_id": target.name,
        "packet_dir": str(target.resolve()),
        "source_packet_dir": str(source_packet.resolve()),
        "source_packet_run_id": source_manifest.get("packet_run_id", source_packet.name),
        "system_id": system_id,
        "case_count": len(case_states),
        "execute": execute,
        "executed_count": len(results),
        "exact_count": sum(1 for row in results if row["replay_exact_match"]),
        "exact_rate": _rate(sum(1 for row in results if row["replay_exact_match"]), len(results)),
        "executor_equivalence_evaluable_count": len(executor_rows),
        "executor_equivalence_count": executor_count,
        "executor_equivalence_rate": _optional_rate(executor_count, len(executor_rows)),
        "failure_mode_counts": _count_by(case_states, "source_failure_mode"),
    }
    manifest = {
        **summary,
        "created_at": datetime.now(UTC).isoformat(),
        "registry_path": str(registry_path.resolve()),
        "case_ids": [row["case_id"] for row in case_states],
        "operator_surface": "rich_cli_exact_probe_replay_v1",
        "entrypoint": "moonie-agent replay-live",
    }
    _write_json(target / "manifest.json", manifest)
    _write_json(target / "summary.json", summary)
    _write_json(target / "commands.json", commands)
    _write_json(target / "live_case_states.json", case_states)
    _write_json(target / "live_replay_results.json", results)
    _write_csv(target / "live_case_states.csv", case_states)
    _write_csv(target / "live_replay_results.csv", results)
    return {
        "packet_dir": str(target.resolve()),
        "manifest": manifest,
        "summary": summary,
        "case_states": case_states,
        "results": results,
        "commands": commands,
    }


def _render_live_replay(
    *,
    source_packet: Path,
    target: Path,
    system_id: str,
    execute: bool,
    case_states: list[dict[str, Any]],
) -> Group:
    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("Source", str(source_packet.resolve()))
    header.add_row("Output", str(target.resolve()))
    header.add_row("System", system_id)
    header.add_row("Mode", "execute" if execute else "dry-run")
    header.add_row("Cases", str(len(case_states)))

    table = Table(title="Exact Probe Replay")
    table.add_column("Case")
    table.add_column("Family")
    table.add_column("Source failure")
    table.add_column("Status")
    table.add_column("Replay failure")
    table.add_column("Exact")
    table.add_column("Executor Eq")
    for state in case_states:
        table.add_row(
            str(state.get("case_id", "")),
            str(state.get("family", "")),
            str(state.get("source_failure_mode", "")),
            str(state.get("status", "")),
            str(state.get("replay_failure_mode", "")),
            str(state.get("replay_exact_match", "")),
            str(state.get("replay_executor_equivalence_match", "")),
        )
    return Group(Panel(header, title="Moonie Exact Replay"), table)


def _resolve_packet_dir(*, packet_id: str, packet_dir: str | Path | None) -> Path:
    if packet_dir:
        return Path(packet_dir)
    if packet_id != "latest":
        return DEFAULT_REPLAY_PACKET_ROOT / packet_id
    candidates = sorted(child for child in DEFAULT_REPLAY_PACKET_ROOT.iterdir() if child.is_dir())
    if not candidates:
        raise ValueError(f"No replay packets found under `{DEFAULT_REPLAY_PACKET_ROOT}`.")
    return candidates[-1]


def _select_rows(rows: list[dict[str, str]], case_ids: list[str]) -> list[dict[str, str]]:
    if not case_ids:
        return rows
    wanted = set(case_ids)
    selected = [row for row in rows if row.get("case_id") in wanted]
    missing = sorted(wanted - {row.get("case_id", "") for row in selected})
    if missing:
        raise ValueError(f"Replay packet does not contain case id(s): {', '.join(missing)}")
    return selected


def _case_state(row: dict[str, str], *, status: str) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id", ""),
        "family": row.get("family", ""),
        "source_failure_mode": row.get("source_failure_mode", ""),
        "baseline_exact_match": row.get("baseline_exact_match", ""),
        "expected_call_count": row.get("expected_call_count", ""),
        "source_actual_call_count": row.get("source_actual_call_count", ""),
        "status": status,
        "replay_failure_mode": "",
        "replay_exact_match": "",
        "replay_executable_match": "",
        "replay_executor_equivalence_match": "",
    }


def _load_cases_by_id(source_packet: Path) -> dict[str, ToolDirectiveProbeCase]:
    cases_by_id = {case.case_id: case for case in build_tool_directive_probe_cases()}
    for case in _packet_replay_cases(source_packet):
        cases_by_id[case.case_id] = case
    return cases_by_id


def _packet_replay_cases(source_packet: Path) -> list[ToolDirectiveProbeCase]:
    payload = _read_json(source_packet / "replay_cases.json")
    if not isinstance(payload, list):
        return []

    cases: list[ToolDirectiveProbeCase] = []
    for row in payload:
        if not isinstance(row, dict) or not row.get("case_id"):
            continue
        cases.append(
            ToolDirectiveProbeCase(
                case_id=str(row["case_id"]),
                family=str(row.get("family", "")),
                messages=[Message.model_validate(message) for message in row.get("messages", [])],
                media=[str(item) for item in row.get("media", [])],
                tool_names=[str(item) for item in row.get("tool_names", [])],
                initial_state=row.get("initial_state") if isinstance(row.get("initial_state"), dict) else {},
                expected_execution=row.get("expected_execution") if isinstance(row.get("expected_execution"), dict) else {},
            )
        )
    return cases


def _case_command(
    *,
    source_packet: Path,
    output_dir: Path,
    system_id: str,
    registry_path: Path,
    case_id: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gemma4_capability_map.runtime.cli",
        "replay-live",
        "--packet-dir",
        str(source_packet.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--system-id",
        system_id,
        "--registry",
        str(registry_path.resolve()),
        "--case-id",
        case_id,
        "--execute",
    ]


def _failure_mode(row: dict[str, Any]) -> str:
    if bool(row.get("exact_match")):
        return "exact"
    if row.get("executable_match") is True:
        return "executable_paraphrase"
    expected_count = int(row.get("expected_call_count") or 0)
    actual_count = int(row.get("actual_call_count") or 0)
    if actual_count == 0:
        return "no_tool_call"
    if expected_count != actual_count:
        return "call_count_mismatch"
    expected_calls = row.get("expected_calls") or []
    actual_calls = row.get("actual_calls") or []
    if expected_calls and actual_calls and expected_calls[0].get("name") != actual_calls[0].get("name"):
        return "wrong_tool"
    return "argument_mismatch"


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
