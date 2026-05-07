from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH
from gemma4_capability_map.runtime.tool_directive_probe import build_tool_directive_probe_cases
from gemma4_capability_map.tools.planner import plan_tool_calls
from gemma4_capability_map.tools.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "tool_probe_replay_packets"
DEFAULT_SOURCE_PROBE = ROOT / "results" / "tool_directive_probe" / "20260507T_mlx_no_directive_probe_v1"
DEFAULT_BASELINE_PROBE = ROOT / "results" / "tool_directive_probe" / "20260506T_mlx_tool_directive_probe_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dry-run exact tool-probe replay packet.")
    parser.add_argument("--source-probe-dir", default=str(DEFAULT_SOURCE_PROBE))
    parser.add_argument("--baseline-probe-dir", default=str(DEFAULT_BASELINE_PROBE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--system-id", default="mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive")
    parser.add_argument("--case-id", action="append", dest="case_ids", default=[])
    parser.add_argument("--include-exact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_tool_probe_replay_packet(
        source_probe_dir=Path(args.source_probe_dir),
        baseline_probe_dir=Path(args.baseline_probe_dir),
        output_root=Path(args.output_root),
        run_group_id=args.run_group_id,
        registry_path=Path(args.registry),
        system_id=args.system_id,
        case_ids=args.case_ids,
        include_exact=args.include_exact,
    )
    print(json.dumps(packet["summary"], indent=2, ensure_ascii=False))


def build_tool_probe_replay_packet(
    *,
    source_probe_dir: Path = DEFAULT_SOURCE_PROBE,
    baseline_probe_dir: Path = DEFAULT_BASELINE_PROBE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_group_id: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    system_id: str = "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
    case_ids: list[str] | None = None,
    include_exact: bool = False,
) -> dict[str, Any]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_run_id = run_group_id or f"{timestamp}_tool_probe_replay_packet"
    packet_dir = output_root / packet_run_id
    case_dir = packet_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = _read_json(source_probe_dir / "manifest.json")
    source_rows = {str(row["case_id"]): row for row in _read_json(source_probe_dir / "probe_results.json")}
    baseline_rows = {}
    if baseline_probe_dir and (baseline_probe_dir / "probe_results.json").exists():
        baseline_rows = {str(row["case_id"]): row for row in _read_json(baseline_probe_dir / "probe_results.json")}
    cases_by_id = {case.case_id: case for case in build_tool_directive_probe_cases()}
    selected_ids = case_ids or sorted(source_rows)
    missing = [case_id for case_id in selected_ids if case_id not in cases_by_id or case_id not in source_rows]
    if missing:
        raise ValueError(f"Unknown replay probe case id(s): {', '.join(missing)}")

    registry = build_default_registry().specs
    rows: list[dict[str, Any]] = []
    replay_cases: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for case_id in selected_ids:
        source_row = source_rows[case_id]
        if bool(source_row.get("exact_match")) and not include_exact:
            continue
        case = cases_by_id[case_id]
        tool_specs = [registry[name] for name in case.tool_names]
        expected_calls = [
            {"name": call.name, "arguments": call.arguments}
            for call in plan_tool_calls(case.messages, case.media, tool_specs)
        ]
        failure_mode = _failure_mode(source_row)
        replay_case = {
            "case_id": case.case_id,
            "family": case.family,
            "messages": [message.model_dump(mode="json") for message in case.messages],
            "media": list(case.media),
            "tool_names": list(case.tool_names),
            "tool_specs": [tool.model_dump(mode="json", by_alias=True) for tool in tool_specs],
            "initial_state": case.initial_state,
            "expected_execution": case.expected_execution,
            "expected_calls": expected_calls,
            "source_actual_calls": source_row.get("actual_calls", []),
            "source_raw_model_output": source_row.get("raw_model_output", ""),
            "source_exact_match": bool(source_row.get("exact_match")),
            "source_executable_match": source_row.get("executable_match"),
            "source_failure_mode": failure_mode,
            "baseline_exact_match": baseline_rows.get(case_id, {}).get("exact_match", ""),
            "baseline_actual_calls": baseline_rows.get(case_id, {}).get("actual_calls", []),
            "live_entrypoint_status": "probe_replay_packet_only_v1",
            "live_entrypoint_note": "This is a raw exact-call replay artifact, not a packaged workflow session.",
        }
        replay_cases.append(replay_case)
        case_path = case_dir / f"{case_id}.json"
        _write_json(case_path, replay_case)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_tool_directive_probe.py"),
            "--system-id",
            system_id,
            "--registry",
            str(registry_path),
            "--output-dir",
            str((packet_dir / "runs" / case_id).resolve()),
            "--case-id",
            case_id,
        ]
        commands.append({"case_id": case_id, "family": case.family, "command": command})
        rows.append(
            {
                "case_id": case_id,
                "family": case.family,
                "source_failure_mode": failure_mode,
                "source_exact_match": bool(source_row.get("exact_match")),
                "source_executable_match": source_row.get("executable_match"),
                "baseline_exact_match": baseline_rows.get(case_id, {}).get("exact_match", ""),
                "expected_call_count": len(expected_calls),
                "source_actual_call_count": int(source_row.get("actual_call_count") or 0),
                "case_path": str(case_path.resolve()),
            }
        )

    summary = {
        "packet_run_id": packet_run_id,
        "packet_dir": str(packet_dir.resolve()),
        "source_probe_dir": str(source_probe_dir.resolve()),
        "baseline_probe_dir": str(baseline_probe_dir.resolve()),
        "source_system_id": source_manifest.get("system_id", ""),
        "replay_system_id": system_id,
        "include_exact": include_exact,
        "case_count": len(rows),
        "failure_mode_counts": _count_by(rows, "source_failure_mode"),
        "family_counts": _count_by(rows, "family"),
        "next_action_counts": _count_by(_next_action_rows(rows), "next_action"),
        "dry_run": True,
    }
    manifest = {
        **summary,
        "created_at": datetime.now(UTC).isoformat(),
        "registry_path": str(registry_path.resolve()),
        "case_ids": [row["case_id"] for row in rows],
    }
    _write_json(packet_dir / "manifest.json", manifest)
    _write_json(packet_dir / "summary.json", summary)
    _write_json(packet_dir / "commands.json", commands)
    _write_json(packet_dir / "replay_cases.json", replay_cases)
    _write_csv(packet_dir / "replay_cases.csv", rows)
    _write_csv(packet_dir / "replay_next_actions.csv", _next_action_rows(rows))
    return {
        "packet_dir": str(packet_dir.resolve()),
        "summary": summary,
        "manifest": manifest,
        "rows": rows,
        "commands": commands,
        "replay_cases": replay_cases,
        "next_actions": _next_action_rows(rows),
    }


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


def _next_action_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [_next_action_row(row) for row in rows]


def _next_action_row(row: dict[str, Any]) -> dict[str, str]:
    family = str(row.get("family", ""))
    failure_mode = str(row.get("source_failure_mode", ""))
    if family == "parallel_tool_calling":
        return {
            "case_id": str(row.get("case_id", "")),
            "family": family,
            "source_failure_mode": failure_mode,
            "priority": "high",
            "next_action": "build_parallel_array_replay_or_workflow",
            "why": "current packaged workflows do not faithfully test the two-call array contract",
        }
    if family.startswith("visual") and failure_mode == "no_tool_call":
        return {
            "case_id": str(row.get("case_id", "")),
            "family": family,
            "source_failure_mode": failure_mode,
            "priority": "high",
            "next_action": "build_visual_state_replay_executor",
            "why": "packaged visual workflows complete, but raw no-directive visual cases collapse to no call",
        }
    if family.startswith(("cli", "api")) and failure_mode == "argument_mismatch":
        return {
            "case_id": str(row.get("case_id", "")),
            "family": family,
            "source_failure_mode": failure_mode,
            "priority": "medium",
            "next_action": "build_canonical_argument_replay",
            "why": "model chooses the right tool family but drifts on canonical path/query/record arguments",
        }
    return {
        "case_id": str(row.get("case_id", "")),
        "family": family,
        "source_failure_mode": failure_mode,
        "priority": "low",
        "next_action": "inspect_case_manually",
        "why": "case does not match a known replay implementation family",
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


if __name__ == "__main__":
    main()
