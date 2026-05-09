from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH
from gemma4_capability_map.runtime.tool_directive_probe import _probe_failure_mode
from gemma4_capability_map.runtime.visual_hard_slice import build_visual_hard_slice_cases
from gemma4_capability_map.tools.planner import plan_tool_calls
from gemma4_capability_map.tools.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "tool_probe_replay_packets"
DEFAULT_VISUAL_HARD_SLICE_PACKET = ROOT / "results" / "visual_hard_slice_probe_packets" / "20260509T_visual_hard_slice_execute_v1"
DEFAULT_SOURCE_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"
DEFAULT_BASELINE_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only"
DEFAULT_REPLAY_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a live-replay packet from visual hard-slice probe results.")
    parser.add_argument("--visual-packet-dir", default=str(DEFAULT_VISUAL_HARD_SLICE_PACKET))
    parser.add_argument("--source-system-id", default=DEFAULT_SOURCE_SYSTEM_ID)
    parser.add_argument("--baseline-system-id", default=DEFAULT_BASELINE_SYSTEM_ID)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--replay-system-id", default=DEFAULT_REPLAY_SYSTEM_ID)
    parser.add_argument("--case-id", action="append", dest="case_ids", default=[])
    parser.add_argument("--family", action="append", dest="families", default=[])
    parser.add_argument("--failure-mode", action="append", dest="failure_modes", default=[])
    parser.add_argument("--include-exact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_visual_hard_slice_replay_packet(
        visual_packet_dir=Path(args.visual_packet_dir),
        source_system_id=args.source_system_id,
        baseline_system_id=args.baseline_system_id,
        output_root=Path(args.output_root),
        run_group_id=args.run_group_id,
        registry_path=Path(args.registry),
        replay_system_id=args.replay_system_id,
        case_ids=args.case_ids,
        families=args.families,
        failure_modes=args.failure_modes,
        include_exact=args.include_exact,
    )
    print(json.dumps(packet["summary"], indent=2, ensure_ascii=False))


def build_visual_hard_slice_replay_packet(
    *,
    visual_packet_dir: Path = DEFAULT_VISUAL_HARD_SLICE_PACKET,
    source_system_id: str = DEFAULT_SOURCE_SYSTEM_ID,
    baseline_system_id: str = DEFAULT_BASELINE_SYSTEM_ID,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_group_id: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    replay_system_id: str = DEFAULT_REPLAY_SYSTEM_ID,
    case_ids: list[str] | None = None,
    families: list[str] | None = None,
    failure_modes: list[str] | None = None,
    include_exact: bool = False,
) -> dict[str, Any]:
    visual_manifest = _read_json(visual_packet_dir / "manifest.json")
    source_probe_dir = visual_packet_dir / source_system_id
    baseline_probe_dir = visual_packet_dir / baseline_system_id
    source_rows = {str(row["case_id"]): row for row in _read_json(source_probe_dir / "probe_results.json")}
    baseline_rows: dict[str, dict[str, Any]] = {}
    if (baseline_probe_dir / "probe_results.json").exists():
        baseline_rows = {str(row["case_id"]): row for row in _read_json(baseline_probe_dir / "probe_results.json")}

    cases_by_id = {case.case_id: case for case in build_visual_hard_slice_cases()}
    selected_ids = case_ids or list(visual_manifest.get("case_ids") or sorted(source_rows))
    missing = [case_id for case_id in selected_ids if case_id not in cases_by_id or case_id not in source_rows]
    if missing:
        raise ValueError(f"Unknown visual hard-slice replay case id(s): {', '.join(missing)}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_run_id = run_group_id or f"{timestamp}_visual_hard_slice_replay_packet"
    packet_dir = output_root / packet_run_id
    case_dir = packet_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    registry = build_default_registry().specs
    rows: list[dict[str, Any]] = []
    replay_cases: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for case_id in selected_ids:
        source_row = source_rows[case_id]
        source_failure_mode = _probe_failure_mode(source_row)
        if bool(source_row.get("exact_match")) and not include_exact:
            continue
        case = cases_by_id[case_id]
        if families and case.family not in families:
            continue
        if failure_modes and source_failure_mode not in failure_modes:
            continue

        tool_specs = [registry[name] for name in case.tool_names]
        expected_calls = [
            {"name": call.name, "arguments": call.arguments}
            for call in plan_tool_calls(case.messages, case.media, tool_specs)
        ]
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
            "source_system_id": source_system_id,
            "source_actual_calls": source_row.get("actual_calls", []),
            "source_raw_model_output": source_row.get("raw_model_output", ""),
            "source_exact_match": bool(source_row.get("exact_match")),
            "source_executable_match": source_row.get("executable_match"),
            "source_failure_mode": source_failure_mode,
            "baseline_system_id": baseline_system_id,
            "baseline_exact_match": baseline_rows.get(case_id, {}).get("exact_match", ""),
            "baseline_actual_calls": baseline_rows.get(case_id, {}).get("actual_calls", []),
            "live_entrypoint_status": "visual_hard_slice_replay_packet_v1",
            "live_entrypoint_note": "This preserves visual hard-slice probe cases for moonie-agent replay-live.",
        }
        replay_cases.append(replay_case)
        case_path = case_dir / f"{case_id}.json"
        _write_json(case_path, replay_case)
        commands.append(
            {
                "case_id": case_id,
                "family": case.family,
                "command": _live_replay_command(
                    packet_dir=packet_dir,
                    registry_path=registry_path,
                    replay_system_id=replay_system_id,
                    case_id=case_id,
                ),
            }
        )
        rows.append(
            {
                "case_id": case_id,
                "family": case.family,
                "source_failure_mode": source_failure_mode,
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
        "visual_packet_dir": str(visual_packet_dir.resolve()),
        "source_probe_dir": str(source_probe_dir.resolve()),
        "baseline_probe_dir": str(baseline_probe_dir.resolve()),
        "source_system_id": source_system_id,
        "baseline_system_id": baseline_system_id,
        "replay_system_id": replay_system_id,
        "include_exact": include_exact,
        "case_count": len(rows),
        "failure_mode_counts": _count_by(rows, "source_failure_mode"),
        "family_counts": _count_by(rows, "family"),
        "dry_run": True,
        "executed_count": 0,
    }
    manifest = {
        **summary,
        "created_at": datetime.now(UTC).isoformat(),
        "registry_path": str(registry_path.resolve()),
        "case_ids": [row["case_id"] for row in rows],
        "source_visual_packet_run_id": visual_manifest.get("packet_run_id", visual_packet_dir.name),
        "operator_surface": "rich_cli_visual_hard_slice_replay_v1",
        "entrypoint": "moonie-agent replay-live",
        "filters": {
            "families": families or [],
            "failure_modes": failure_modes or [],
        },
    }
    _write_json(packet_dir / "manifest.json", manifest)
    _write_json(packet_dir / "summary.json", summary)
    _write_json(packet_dir / "commands.json", commands)
    _write_json(packet_dir / "replay_cases.json", replay_cases)
    _write_json(packet_dir / "replay_results.json", [])
    _write_csv(packet_dir / "replay_cases.csv", rows)
    _write_csv(packet_dir / "replay_results.csv", [])
    return {
        "packet_dir": str(packet_dir.resolve()),
        "summary": summary,
        "manifest": manifest,
        "rows": rows,
        "commands": commands,
        "replay_cases": replay_cases,
    }


def _live_replay_command(*, packet_dir: Path, registry_path: Path, replay_system_id: str, case_id: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gemma4_capability_map.runtime.cli",
        "replay-live",
        "--packet-dir",
        str(packet_dir.resolve()),
        "--system-id",
        replay_system_id,
        "--registry",
        str(registry_path.resolve()),
        "--case-id",
        case_id,
        "--execute",
    ]


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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
