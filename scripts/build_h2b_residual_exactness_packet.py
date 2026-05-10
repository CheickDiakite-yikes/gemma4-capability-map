from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "tool_probe_replay_packets"
DEFAULT_RESIDUAL_TABLE = (
    ROOT
    / "results"
    / "reports"
    / "h2a_stale_selection_transfer_synthesis"
    / "tables"
    / "h2a_transfer_residual_rows.csv"
)
DEFAULT_REPLAY_SYSTEM_ID = (
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_"
    "visual_role_catalog_component_label_guard_visual_stale_selection_gate"
)


@dataclass(frozen=True)
class ResidualCaseSpec:
    case_id: str
    source_packet_dir: Path
    residual_class: str
    residual_axis: str


RESIDUAL_CASE_SPECS: tuple[ResidualCaseSpec, ...] = (
    ResidualCaseSpec(
        "component_value_result_pill_log_decoy",
        ROOT / "results" / "tool_probe_replay_packets" / "20260510T_visual_hard_slice_component_value_oracle_dry_run_v1",
        "result_pill_exact_label",
        "executor_equivalent_alias",
    ),
    ResidualCaseSpec(
        "h1o_code_alert_s92_negated_toggle_decoy",
        ROOT / "results" / "tool_probe_replay_packets" / "20260510T_h1o_control_factorial_oracle_dry_run_v1",
        "alert_s92_code_label",
        "executor_equivalent_alias",
    ),
    ResidualCaseSpec(
        "h1o_code_badge_c08_note_decoy",
        ROOT / "results" / "tool_probe_replay_packets" / "20260510T_h1o_control_factorial_oracle_dry_run_v1",
        "badge_c08_code_label",
        "executor_equivalent_alias",
    ),
    ResidualCaseSpec(
        "h1p_compact_state_tag_log_value_decoy",
        ROOT / "results" / "tool_probe_replay_packets" / "20260510T_h1p_component_value_holdout_oracle_dry_run_v1",
        "state_tag_component_class",
        "non_executor_argument_mismatch",
    ),
    ResidualCaseSpec(
        "h1p_surface_mode_toggle_note_value_decoy",
        ROOT / "results" / "tool_probe_replay_packets" / "20260510T_h1p_component_value_holdout_oracle_dry_run_v1",
        "mode_toggle_component_class",
        "non_executor_argument_mismatch",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2b exact-alias residual replay packet.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--replay-system-id", default=DEFAULT_REPLAY_SYSTEM_ID)
    parser.add_argument("--residual-table", default=str(DEFAULT_RESIDUAL_TABLE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_h2b_residual_exactness_packet(
        output_root=Path(args.output_root),
        run_group_id=args.run_group_id,
        registry_path=Path(args.registry),
        replay_system_id=args.replay_system_id,
        residual_table=Path(args.residual_table),
    )
    print(json.dumps(packet["summary"], indent=2, ensure_ascii=False))


def build_h2b_residual_exactness_packet(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_group_id: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    replay_system_id: str = DEFAULT_REPLAY_SYSTEM_ID,
    residual_table: Path = DEFAULT_RESIDUAL_TABLE,
) -> dict[str, Any]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_run_id = run_group_id or f"{timestamp}_h2b_residual_exactness_packet"
    packet_dir = output_root / packet_run_id
    case_dir = packet_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    residual_rows = _residual_rows(residual_table)
    residual_by_case = {row["case_id"]: row for row in residual_rows}
    replay_cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for spec in RESIDUAL_CASE_SPECS:
        source_cases = {case["case_id"]: case for case in _read_json(spec.source_packet_dir / "replay_cases.json")}
        if spec.case_id not in source_cases:
            raise ValueError(f"Missing residual case {spec.case_id} in {spec.source_packet_dir}")
        if spec.case_id not in residual_by_case:
            raise ValueError(f"Missing H2a residual metadata for {spec.case_id} in {residual_table}")
        source_case = dict(source_cases[spec.case_id])
        residual = residual_by_case[spec.case_id]
        source_case.update(
            {
                "source_system_id": "h2a_stale_selection_transfer_residuals_v1",
                "source_failure_mode": residual["failure_mode"],
                "source_exact_match": False,
                "source_executable_match": residual["executor_equivalence_match"] == "True",
                "h2a_residual_axis": spec.residual_axis,
                "h2a_residual_class": spec.residual_class,
                "h2a_actual_tool": residual["actual_tool"],
                "h2a_actual_arguments": _json_load_or_string(residual["actual_arguments"]),
                "h2a_expected_tool": residual["expected_tool"],
                "h2a_expected_arguments": _json_load_or_string(residual["expected_arguments"]),
                "source_packet_dir": str(spec.source_packet_dir.relative_to(ROOT)),
                "live_entrypoint_status": "h2b_residual_exactness_packet_v1",
                "live_entrypoint_note": (
                    "Composed H2a transfer residual packet for exact alias/code-label testing. "
                    "This packet must not expose expected calls beyond the normal replay contract."
                ),
            }
        )
        replay_cases.append(source_case)
        case_path = case_dir / f"{spec.case_id}.json"
        _write_json(case_path, source_case)
        rows.append(
            {
                "case_id": spec.case_id,
                "family": source_case["family"],
                "source_failure_mode": residual["failure_mode"],
                "source_exact_match": False,
                "source_executable_match": residual["executor_equivalence_match"] == "True",
                "baseline_exact_match": "",
                "expected_call_count": len(source_case.get("expected_calls") or []),
                "source_actual_call_count": 1 if residual["actual_tool"] else 0,
                "residual_axis": spec.residual_axis,
                "residual_class": spec.residual_class,
                "h2a_failure_mode": residual["failure_mode"],
                "h2a_executor_equivalence_match": residual["executor_equivalence_match"],
                "expected_tool": residual["expected_tool"],
                "expected_arguments": residual["expected_arguments"],
                "h2a_actual_tool": residual["actual_tool"],
                "h2a_actual_arguments": residual["actual_arguments"],
                "source_packet_dir": str(spec.source_packet_dir.relative_to(ROOT)),
                "case_path": str(case_path.resolve()),
            }
        )
        commands.append(
            {
                "case_id": spec.case_id,
                "family": source_case["family"],
                "command": _live_replay_command(
                    packet_dir=packet_dir,
                    registry_path=registry_path,
                    replay_system_id=replay_system_id,
                    case_id=spec.case_id,
                ),
            }
        )

    summary = {
        "packet_run_id": packet_run_id,
        "packet_dir": str(packet_dir.resolve()),
        "case_count": len(rows),
        "residual_axis_counts": _count_by(rows, "residual_axis"),
        "h2a_failure_mode_counts": _count_by(rows, "h2a_failure_mode"),
        "h2a_executor_equivalent_count": sum(
            1 for row in rows if row["h2a_executor_equivalence_match"] == "True"
        ),
        "h2a_non_executor_count": sum(
            1 for row in rows if row["h2a_executor_equivalence_match"] != "True"
        ),
        "dry_run": True,
        "executed_count": 0,
    }
    manifest = {
        **summary,
        "created_at": datetime.now(UTC).isoformat(),
        "registry_path": str(registry_path.resolve()),
        "residual_table": str(residual_table.resolve()),
        "case_ids": [row["case_id"] for row in rows],
        "source_packet_dirs": sorted({row["source_packet_dir"] for row in rows}),
        "replay_system_id": replay_system_id,
        "purpose": (
            "H2b residual exactness packet: isolate the exact alias/code-label failures left by H2a transfer."
        ),
        "operator_surface": "rich_cli_tool_probe_replay_live_v1",
        "entrypoint": "moonie-agent replay-live",
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


def _residual_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_load_or_string(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


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
