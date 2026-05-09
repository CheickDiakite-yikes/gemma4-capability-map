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
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets",
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

    gate_rows = _gate_rows(rows)
    failure_rows = _failure_mode_rows(rows)
    family_rows = _family_rows(results)
    case_deltas_vs_contracted = _comparison_case_rows(comparison_payloads, baseline_system_id=CONTRACTED_SYSTEM_ID)
    case_deltas_vs_no_directive = _comparison_case_rows(comparison_payloads, baseline_system_id=NO_DIRECTIVE_SYSTEM_ID)

    summary = {
        "packet_dir": str(packet_dir.resolve()),
        "manifest": manifest,
        "candidate_count": len(selected_system_ids),
        "system_count": len(selected_system_ids),
        "case_count": len(selected_cases),
        "executed_count": sum(1 for row in rows if row["execute"]),
        "dry_run_count": sum(1 for row in rows if not row["execute"]),
        "rows": rows,
        "gate_rows": gate_rows,
        "failure_mode_rows": failure_rows,
        "family_rows": family_rows,
        "commands": commands,
        "results": results,
    }
    (packet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (packet_dir / "commands.json").write_text(json.dumps(commands, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (packet_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(packet_dir / "candidate_summary.csv", rows)
    _write_csv(packet_dir / "system_summary.csv", rows)
    _write_csv(packet_dir / "candidate_gate_summary.csv", gate_rows)
    _write_csv(packet_dir / "candidate_failure_mode_counts.csv", failure_rows)
    _write_csv(packet_dir / "family_summary.csv", family_rows)
    _write_csv(packet_dir / "case_deltas_vs_contracted.csv", case_deltas_vs_contracted)
    _write_csv(packet_dir / "case_deltas_vs_no_directive.csv", case_deltas_vs_no_directive)
    (packet_dir / "candidate_gate_summary.md").write_text(_gate_markdown(manifest, gate_rows), encoding="utf-8")
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
        "executor_equivalence_match_count": "",
        "executor_equivalence_match_rate": "",
        "dominant_failure_mode": "",
        "failure_mode_counts": "",
        "comparison_path": "",
        "comparison_vs_contracted": "",
        "comparison_vs_no_directive": "",
        "delta_exact_vs_contracted": "",
        "delta_exact_vs_no_directive": "",
        "delta_executable_vs_contracted": "",
        "delta_executable_vs_no_directive": "",
        "delta_executor_equivalence_vs_contracted": "",
        "delta_executor_equivalence_vs_no_directive": "",
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
        "executor_equivalence_match_count": summary.get("executor_equivalence_match_count", 0),
        "executor_equivalence_match_rate": summary.get("executor_equivalence_match_rate", ""),
        "dominant_failure_mode": dominant,
        "failure_mode_counts": dict(sorted(failure_counts.items())),
    }


def _gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "system_id": row.get("system_id", ""),
            "exact_match_count": row.get("exact_match_count", ""),
            "exact_match_rate": row.get("exact_match_rate", ""),
            "executable_match_count": row.get("executable_match_count", ""),
            "executable_match_rate": row.get("executable_match_rate", ""),
            "executor_equivalence_match_count": row.get("executor_equivalence_match_count", ""),
            "executor_equivalence_match_rate": row.get("executor_equivalence_match_rate", ""),
            "delta_exact_vs_contracted": row.get("delta_exact_vs_contracted", ""),
            "delta_exact_vs_no_directive": row.get("delta_exact_vs_no_directive", ""),
            "delta_executable_vs_contracted": row.get("delta_executable_vs_contracted", ""),
            "delta_executable_vs_no_directive": row.get("delta_executable_vs_no_directive", ""),
            "delta_executor_equivalence_vs_contracted": row.get("delta_executor_equivalence_vs_contracted", ""),
            "delta_executor_equivalence_vs_no_directive": row.get("delta_executor_equivalence_vs_no_directive", ""),
            "dominant_failure_mode": row.get("dominant_failure_mode", ""),
            "hard_slice_gate": row.get("hard_slice_gate", ""),
            "output_dir": row.get("output_dir", ""),
        }
        for row in rows
    ]


def _failure_mode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        counts = row.get("failure_mode_counts", {})
        if not isinstance(counts, dict):
            continue
        for failure_mode, count in sorted(counts.items()):
            output.append(
                {
                    "system_id": row.get("system_id", ""),
                    "failure_mode": failure_mode,
                    "count": count,
                }
            )
    return output


def _family_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for result in results:
        system_id = str(result.get("system_id", ""))
        summary = result.get("summary", {})
        family_summary = summary.get("family_summary", {}) if isinstance(summary, dict) else {}
        if not isinstance(family_summary, dict):
            continue
        for family, bucket in sorted(family_summary.items()):
            if not isinstance(bucket, dict):
                continue
            output.append(
                {
                    "system_id": system_id,
                    "family": family,
                    "case_count": bucket.get("cases", ""),
                    "exact_count": bucket.get("exact", ""),
                    "exact_rate": bucket.get("exact_rate", ""),
                    "executable_case_count": bucket.get("executable_cases", ""),
                    "executable_count": bucket.get("executable", ""),
                    "executable_rate": bucket.get("executable_rate", ""),
                    "executor_equivalence_case_count": bucket.get("executor_equivalence_cases", ""),
                    "executor_equivalence_count": bucket.get("executor_equivalence", ""),
                    "executor_equivalence_rate": bucket.get("executor_equivalence_rate", ""),
                }
            )
    return output


def _comparison_case_rows(
    comparison_payloads: dict[tuple[str, str], dict[str, Any]],
    *,
    baseline_system_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (baseline_id, system_id), comparison in sorted(comparison_payloads.items()):
        if baseline_id != baseline_system_id:
            continue
        case_deltas = comparison.get("case_deltas", []) if isinstance(comparison, dict) else []
        if not isinstance(case_deltas, list):
            continue
        for case_delta in case_deltas:
            if not isinstance(case_delta, dict):
                continue
            output.append({"system_id": system_id, **case_delta})
    return output


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
        row["delta_executor_equivalence_vs_contracted"] = _executor_equivalence_delta(contracted_comparison)
        row["delta_executor_equivalence_vs_no_directive"] = _executor_equivalence_delta(no_directive_comparison)
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
    executor_equivalence_delta = _executor_equivalence_delta(comparison_vs_no_directive)
    if (
        exact_delta > 0.0
        or (executable_delta not in ("", None) and float(executable_delta) > 0.0)
        or (executor_equivalence_delta not in ("", None) and float(executor_equivalence_delta) > 0.0)
    ):
        return "hard_slice_improved_vs_no_directive"
    return "no_hard_slice_gain"


def _executable_delta(comparison: dict[str, Any]) -> float | str:
    candidate = comparison.get("candidate_executable_match_rate")
    baseline = comparison.get("baseline_executable_match_rate")
    if candidate is None or baseline is None:
        return ""
    return float(candidate or 0.0) - float(baseline or 0.0)


def _executor_equivalence_delta(comparison: dict[str, Any]) -> float | str:
    candidate = comparison.get("candidate_executor_equivalence_match_rate")
    baseline = comparison.get("baseline_executor_equivalence_match_rate")
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


def _gate_markdown(manifest: dict[str, Any], gate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Visual Hard Slice Candidate Gates",
        "",
        f"- packet_run_id: `{manifest['packet_run_id']}`",
        f"- created_at: `{manifest['created_at']}`",
        f"- case_count: `{manifest['case_count']}`",
        f"- contracted_system_id: `{manifest['contracted_system_id']}`",
        f"- no_directive_system_id: `{manifest['no_directive_system_id']}`",
        "",
        "| System | Exact | Executable | Executor Eq | Delta Exact vs No Directive | Delta Exec Eq vs No Directive | Failure | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in gate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("system_id", "")),
                    _markdown_number(row.get("exact_match_rate", "")),
                    _markdown_number(row.get("executable_match_rate", "")),
                    _markdown_number(row.get("executor_equivalence_match_rate", "")),
                    _markdown_number(row.get("delta_exact_vs_no_directive", "")),
                    _markdown_number(row.get("delta_executor_equivalence_vs_no_directive", "")),
                    str(row.get("dominant_failure_mode", "")),
                    str(row.get("hard_slice_gate", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_number(value: Any) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
