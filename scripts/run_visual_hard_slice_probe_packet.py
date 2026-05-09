from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry
from gemma4_capability_map.runtime.tool_directive_probe import (
    _probe_failure_mode,
    run_tool_directive_probe,
    write_tool_directive_probe_comparison,
)
from gemma4_capability_map.runtime.visual_hard_slice import build_visual_hard_slice_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "visual_hard_slice_probe_packets"
CONTRACTED_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only"
NO_DIRECTIVE_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"
DEFAULT_SYSTEM_IDS = [
    CONTRACTED_SYSTEM_ID,
    NO_DIRECTIVE_SYSTEM_ID,
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_literal_guard",
]


def build_visual_hard_slice_probe_packet(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_group_id: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    system_ids: list[str] | None = None,
    case_ids: list[str] | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    registry_path = Path(registry_path)
    registry = load_model_registry(registry_path)
    systems = registry.get("systems", {})
    selected_system_ids = system_ids or list(DEFAULT_SYSTEM_IDS)
    missing_systems = [system_id for system_id in selected_system_ids if system_id not in systems]
    if missing_systems:
        raise ValueError(f"Unknown system profile(s): {', '.join(missing_systems)}")

    cases = build_visual_hard_slice_cases()
    cases_by_id = {case.case_id: case for case in cases}
    selected_case_ids = case_ids or [case.case_id for case in cases]
    missing_cases = [case_id for case_id in selected_case_ids if case_id not in cases_by_id]
    if missing_cases:
        raise ValueError(f"Unknown visual hard-slice case id(s): {', '.join(missing_cases)}")
    selected_cases = [cases_by_id[case_id] for case_id in selected_case_ids]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_run_id = run_group_id or f"{timestamp}_visual_hard_slice_probe_packet"
    packet_dir = Path(output_root) / packet_run_id
    packet_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "packet_run_id": packet_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "registry_path": str(registry_path.resolve()),
        "execute": execute,
        "system_ids": selected_system_ids,
        "case_ids": selected_case_ids,
        "case_count": len(selected_cases),
        "contracted_system_id": CONTRACTED_SYSTEM_ID,
        "no_directive_system_id": NO_DIRECTIVE_SYSTEM_ID,
        "purpose": "Fresh executable visual hard slice for measuring whether catalog/profile harnesses generalize beyond saturated top-line readiness.",
    }

    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    probe_dirs: dict[str, Path] = {}
    comparison_payloads: dict[tuple[str, str], dict[str, Any]] = {}

    for system_id in selected_system_ids:
        output_dir = packet_dir / system_id
        command = _probe_command(
            system_id=system_id,
            output_dir=output_dir,
            registry_path=registry_path,
            case_ids=selected_case_ids,
        )
        commands.append(
            {
                "system_id": system_id,
                "output_dir": str(output_dir.resolve()),
                "case_ids": selected_case_ids,
                "command": command,
            }
        )
        row = _empty_row(system_id=system_id, execute=execute, output_dir=output_dir)
        if execute:
            result = run_tool_directive_probe(
                system_id=system_id,
                output_dir=output_dir,
                registry_path=registry_path,
                cases=selected_cases,
            )
            probe_dirs[system_id] = output_dir
            row.update(_summary_fields(result["summary"], result["rows"]))
            results.append(
                {
                    "system_id": system_id,
                    "probe_output_dir": result["output_dir"],
                    "summary": result["summary"],
                }
            )
        rows.append(row)

    if execute:
        _attach_comparisons(rows=rows, results=results, probe_dirs=probe_dirs, comparison_payloads=comparison_payloads)

    summary = {
        "packet_dir": str(packet_dir.resolve()),
        "manifest": manifest,
        "candidate_count": len(selected_system_ids),
        "system_count": len(selected_system_ids),
        "case_count": len(selected_cases),
        "executed_count": sum(1 for row in rows if row["execute"]),
        "dry_run_count": sum(1 for row in rows if not row["execute"]),
        "rows": rows,
        "commands": commands,
        "results": results,
    }
    (packet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (packet_dir / "commands.json").write_text(json.dumps(commands, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (packet_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(packet_dir / "candidate_summary.csv", rows)
    _write_csv(packet_dir / "system_summary.csv", rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run the executable visual hard-slice probe packet.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--system-id", action="append", dest="system_ids", default=[])
    parser.add_argument("--case-id", action="append", dest="case_ids", default=[])
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_visual_hard_slice_probe_packet(
        output_root=args.output_root,
        run_group_id=args.run_group_id,
        registry_path=args.registry,
        system_ids=args.system_ids or None,
        case_ids=args.case_ids or None,
        execute=args.execute,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _probe_command(*, system_id: str, output_dir: Path, registry_path: Path, case_ids: list[str]) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_visual_hard_slice_probe.py"),
        "--system-id",
        system_id,
        "--registry",
        str(registry_path),
        "--output-dir",
        str(output_dir),
    ]
    for case_id in case_ids:
        command.extend(["--case-id", case_id])
    return command


def _empty_row(*, system_id: str, execute: bool, output_dir: Path) -> dict[str, Any]:
    return {
        "system_id": system_id,
        "execute": execute,
        "output_dir": str(output_dir.resolve()),
        "exact_match_count": "",
        "exact_match_rate": "",
        "executable_match_count": "",
        "executable_match_rate": "",
        "dominant_failure_mode": "",
        "failure_mode_counts": "",
        "comparison_path": "",
        "comparison_vs_contracted": "",
        "comparison_vs_no_directive": "",
        "delta_exact_vs_contracted": "",
        "delta_exact_vs_no_directive": "",
        "delta_executable_vs_contracted": "",
        "delta_executable_vs_no_directive": "",
        "hard_slice_gate": "",
    }


def _summary_fields(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counts = Counter(_probe_failure_mode(row) for row in rows)
    dominant = ""
    if failure_counts:
        dominant = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return {
        "exact_match_count": summary.get("exact_match_count", 0),
        "exact_match_rate": summary.get("exact_match_rate", 0.0),
        "executable_match_count": summary.get("executable_match_count", 0),
        "executable_match_rate": summary.get("executable_match_rate", ""),
        "dominant_failure_mode": dominant,
        "failure_mode_counts": dict(sorted(failure_counts.items())),
    }


def _attach_comparisons(
    *,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    probe_dirs: dict[str, Path],
    comparison_payloads: dict[tuple[str, str], dict[str, Any]],
) -> None:
    contracted_dir = probe_dirs.get(CONTRACTED_SYSTEM_ID)
    no_directive_dir = probe_dirs.get(NO_DIRECTIVE_SYSTEM_ID)
    result_by_system = {str(item["system_id"]): item for item in results}
    for row in rows:
        system_id = str(row["system_id"])
        output_dir = probe_dirs.get(system_id)
        if output_dir is None:
            continue
        contracted_comparison: dict[str, Any] = {}
        no_directive_comparison: dict[str, Any] = {}
        contracted_outputs: dict[str, str] = {}
        no_directive_outputs: dict[str, str] = {}
        if contracted_dir is not None:
            contracted_outputs = write_tool_directive_probe_comparison(
                contracted_dir,
                output_dir,
                output_dir=output_dir / "comparison_vs_contracted",
            )
            contracted_comparison = _read_json(Path(contracted_outputs["summary"]))
            comparison_payloads[(CONTRACTED_SYSTEM_ID, system_id)] = contracted_comparison
        if no_directive_dir is not None:
            no_directive_outputs = write_tool_directive_probe_comparison(
                no_directive_dir,
                output_dir,
                output_dir=output_dir / "comparison_vs_no_directive",
            )
            no_directive_comparison = _read_json(Path(no_directive_outputs["summary"]))
            comparison_payloads[(NO_DIRECTIVE_SYSTEM_ID, system_id)] = no_directive_comparison

        row["comparison_path"] = contracted_outputs.get("summary", "")
        row["comparison_vs_contracted"] = contracted_outputs.get("summary", "")
        row["comparison_vs_no_directive"] = no_directive_outputs.get("summary", "")
        row["delta_exact_vs_contracted"] = contracted_comparison.get("delta_exact_match_rate", "")
        row["delta_exact_vs_no_directive"] = no_directive_comparison.get("delta_exact_match_rate", "")
        row["delta_executable_vs_contracted"] = _executable_delta(contracted_comparison)
        row["delta_executable_vs_no_directive"] = _executable_delta(no_directive_comparison)
        row["hard_slice_gate"] = _hard_slice_gate(system_id=system_id, comparison_vs_no_directive=no_directive_comparison)

        result_by_system.get(system_id, {}).setdefault("comparison_outputs", {})
        result_by_system.get(system_id, {})["comparison_outputs"] = {
            "comparison_vs_contracted": contracted_outputs,
            "comparison_vs_no_directive": no_directive_outputs,
        }


def _hard_slice_gate(*, system_id: str, comparison_vs_no_directive: dict[str, Any]) -> str:
    if system_id == CONTRACTED_SYSTEM_ID:
        return "contracted_reference"
    if system_id == NO_DIRECTIVE_SYSTEM_ID:
        return "no_directive_reference"
    exact_delta = float(comparison_vs_no_directive.get("delta_exact_match_rate") or 0.0)
    executable_delta = _executable_delta(comparison_vs_no_directive)
    if exact_delta > 0.0 or (executable_delta not in ("", None) and float(executable_delta) > 0.0):
        return "hard_slice_improved_vs_no_directive"
    return "no_hard_slice_gain"


def _executable_delta(comparison: dict[str, Any]) -> float | str:
    candidate = comparison.get("candidate_executable_match_rate")
    baseline = comparison.get("baseline_executable_match_rate")
    if candidate is None or baseline is None:
        return ""
    return float(candidate or 0.0) - float(baseline or 0.0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field, "")) for field in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
