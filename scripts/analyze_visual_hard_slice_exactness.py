from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.runtime.tool_directive_probe import _probe_failure_mode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET_DIR = (
    ROOT / "results" / "visual_hard_slice_probe_packets" / "20260509T_visual_hard_slice_executor_equivalence_v1"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_hard_slice_exactness_diagnostic"
DEFAULT_SYSTEM_IDS = [
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets",
]


SYSTEM_LABELS = {
    "mlx_gemma4_e2b_reasoner_only": "contracted",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive": "no directive",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog": "visual role catalog",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints": "catalog arg hints",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints": "catalog split selector",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints": "catalog schema fields",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets": "catalog schema target literals",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_literal_guard": "visual catalog literal",
}


def analyze_visual_hard_slice_exactness(
    *,
    packet_dir: str | Path = DEFAULT_PACKET_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    system_ids: list[str] | None = None,
) -> dict[str, Any]:
    packet_root = Path(packet_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tables_dir = target / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(packet_root / "manifest.json")
    selected_system_ids = system_ids or list(DEFAULT_SYSTEM_IDS)
    rows: list[dict[str, Any]] = []
    for system_id in selected_system_ids:
        probe_path = packet_root / system_id / "probe_results.json"
        for probe_row in _read_json(probe_path):
            rows.append(_diagnostic_row(system_id=system_id, probe_row=probe_row))

    gap_rows = [row for row in rows if row["exact_match"] is False]
    system_rows = _system_rows(rows)
    payload_manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(target.resolve()),
        "packet_dir": str(packet_root.resolve()),
        "packet_run_id": manifest.get("packet_run_id", packet_root.name),
        "system_ids": selected_system_ids,
        "system_count": len(selected_system_ids),
        "case_row_count": len(rows),
        "exactness_gap_count": len(gap_rows),
        "purpose": (
            "Separate strict benchmark-canonical visual argument exactness from executor-visible target success "
            "on the latest visual hard-slice packet."
        ),
    }
    payload = {
        "manifest": payload_manifest,
        "system_summary": system_rows,
        "case_rows": rows,
        "exactness_gap_rows": gap_rows,
    }

    _write_json(target / "manifest.json", payload_manifest)
    _write_json(target / "exactness_diagnostic.json", payload)
    _write_csv(tables_dir / "visual_hard_slice_exactness_summary.csv", system_rows)
    _write_csv(tables_dir / "visual_hard_slice_exactness_cases.csv", rows)
    _write_csv(tables_dir / "visual_hard_slice_exactness_gaps.csv", gap_rows)
    (target / "exactness_diagnostic.md").write_text(_markdown(payload_manifest, system_rows, gap_rows), encoding="utf-8")
    return payload


def _diagnostic_row(*, system_id: str, probe_row: dict[str, Any]) -> dict[str, Any]:
    expected_calls = _calls(probe_row.get("expected_calls"))
    actual_calls = _calls(probe_row.get("actual_calls"))
    expected_tools = [str(call.get("name", "")) for call in expected_calls]
    actual_tools = [str(call.get("name", "")) for call in actual_calls]
    expected_arguments = [call.get("arguments", {}) for call in expected_calls]
    actual_arguments = [call.get("arguments", {}) for call in actual_calls]
    expected_targets = _expected_region_ids(probe_row.get("expected_execution"))
    actual_targets = _actual_region_ids(probe_row.get("actual_execution"))
    exact_match = bool(probe_row.get("exact_match"))
    executable_match = _optional_bool(probe_row.get("executable_match"))
    strict_tool_match = expected_tools == actual_tools
    strict_argument_match = expected_arguments == actual_arguments
    executor_target_match = _executor_target_match(expected_targets, actual_targets, executable_match)
    failure_mode = _probe_failure_mode(probe_row)
    diagnosis = _diagnosis(
        exact_match=exact_match,
        executable_match=executable_match,
        strict_tool_match=strict_tool_match,
        strict_argument_match=strict_argument_match,
        executor_target_match=executor_target_match,
        actual_calls=actual_calls,
    )
    return {
        "system_id": system_id,
        "system_label": SYSTEM_LABELS.get(system_id, system_id),
        "case_id": str(probe_row.get("case_id", "")),
        "family": str(probe_row.get("family", "")),
        "exact_match": exact_match,
        "executable_match": executable_match,
        "executor_target_match": executor_target_match,
        "strict_tool_match": strict_tool_match,
        "strict_argument_match": strict_argument_match,
        "expected_tools": " -> ".join(expected_tools),
        "actual_tools": " -> ".join(actual_tools),
        "expected_arguments": json.dumps(expected_arguments, ensure_ascii=False, sort_keys=True),
        "actual_arguments": json.dumps(actual_arguments, ensure_ascii=False, sort_keys=True),
        "expected_target_region_ids": ",".join(expected_targets),
        "actual_target_region_ids": ",".join(actual_targets),
        "failure_mode": failure_mode,
        "exactness_diagnosis": diagnosis,
        "research_interpretation": _research_interpretation(diagnosis),
        "next_action": _next_action(diagnosis),
    }


def _system_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for system_id in sorted({str(row["system_id"]) for row in rows}):
        selected = [row for row in rows if row["system_id"] == system_id]
        case_count = len(selected)
        exact_count = sum(1 for row in selected if row["exact_match"] is True)
        executable_count = sum(1 for row in selected if row["executable_match"] is True)
        executor_success_non_exact = sum(
            1 for row in selected if row["exact_match"] is False and row["executor_target_match"] is True
        )
        benchmark_label_artifacts = sum(
            1 for row in selected if row["research_interpretation"] == "benchmark_label_artifact_candidate"
        )
        true_harness_failures = sum(1 for row in selected if row["research_interpretation"] == "true_harness_failure")
        diagnosis_counts = Counter(str(row["exactness_diagnosis"]) for row in selected)
        output.append(
            {
                "system_id": system_id,
                "system_label": SYSTEM_LABELS.get(system_id, system_id),
                "case_count": case_count,
                "exact_count": exact_count,
                "exact_rate": exact_count / case_count if case_count else 0.0,
                "executable_count": executable_count,
                "executable_rate": executable_count / case_count if case_count else 0.0,
                "executor_success_non_exact_count": executor_success_non_exact,
                "benchmark_label_artifact_candidate_count": benchmark_label_artifacts,
                "true_harness_failure_count": true_harness_failures,
                "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
            }
        )
    return output


def _diagnosis(
    *,
    exact_match: bool,
    executable_match: bool | None,
    strict_tool_match: bool,
    strict_argument_match: bool,
    executor_target_match: bool | None,
    actual_calls: list[dict[str, Any]],
) -> str:
    if exact_match:
        return "exact_contract_match"
    if not actual_calls:
        return "no_tool_call"
    if executable_match is True and strict_tool_match and not strict_argument_match:
        return "executable_selector_alias"
    if executable_match is True:
        return "executor_success_noncanonical_protocol"
    if not strict_tool_match:
        return "wrong_tool_executor_failure"
    if executor_target_match is False:
        return "executor_target_miss"
    return "non_exact_protocol_failure"


def _research_interpretation(diagnosis: str) -> str:
    if diagnosis == "exact_contract_match":
        return "strict_protocol_success"
    if diagnosis == "executable_selector_alias":
        return "benchmark_label_artifact_candidate"
    if diagnosis == "executor_success_noncanonical_protocol":
        return "executor_success_protocol_drift"
    return "true_harness_failure"


def _next_action(diagnosis: str) -> str:
    if diagnosis == "exact_contract_match":
        return "preserve as control"
    if diagnosis == "executable_selector_alias":
        return "use executor-equivalence score before tuning another target_query wording profile"
    if diagnosis == "executor_success_noncanonical_protocol":
        return "inspect protocol shape before treating the result as a model failure"
    if diagnosis == "wrong_tool_executor_failure":
        return "treat as routing failure; check stale selection and allowed-tool priority"
    if diagnosis == "no_tool_call":
        return "treat as visual tool-initiation failure"
    if diagnosis == "executor_target_miss":
        return "treat as true visual targeting failure"
    return "inspect row manually before promotion"


def _calls(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _expected_region_ids(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("region_ids"), list):
        return [str(item) for item in value["region_ids"]]
    if value.get("region_id") is not None:
        return [str(value["region_id"])]
    return []


def _actual_region_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    for result in reversed(value):
        output = result.get("output", {}) if isinstance(result, dict) else {}
        if not isinstance(output, dict):
            continue
        region_ids = output.get("region_ids")
        if isinstance(region_ids, list):
            return [str(item) for item in region_ids]
        if output.get("region_id") is not None:
            return [str(output["region_id"])]
    return []


def _executor_target_match(
    expected_targets: list[str],
    actual_targets: list[str],
    executable_match: bool | None,
) -> bool | None:
    if not expected_targets:
        return executable_match
    return actual_targets == expected_targets


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _markdown(manifest: dict[str, Any], system_rows: list[dict[str, Any]], gap_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Visual Hard-Slice Exactness Diagnostic",
        "",
        f"- packet_run_id: `{manifest['packet_run_id']}`",
        f"- system_count: `{manifest['system_count']}`",
        f"- case_row_count: `{manifest['case_row_count']}`",
        f"- exactness_gap_count: `{manifest['exactness_gap_count']}`",
        "",
        "## System Summary",
        "",
        "| System | Exact | Executable | Non-Exact Executor Success | Label-Artifact Candidates | True Harness Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in system_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["system_label"]),
                    f"{row['exact_count']} / {row['case_count']}",
                    f"{row['executable_count']} / {row['case_count']}",
                    str(row["executor_success_non_exact_count"]),
                    str(row["benchmark_label_artifact_candidate_count"]),
                    str(row["true_harness_failure_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Exactness Gaps",
            "",
            "| System | Case | Failure | Expected Target | Actual Target | Diagnosis | Interpretation |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in gap_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["system_label"]),
                    str(row["case_id"]),
                    str(row["failure_mode"]),
                    str(row["expected_target_region_ids"]),
                    str(row["actual_target_region_ids"]),
                    str(row["exactness_diagnosis"]),
                    str(row["research_interpretation"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate visual hard-slice strict exactness from executor target success.")
    parser.add_argument("--packet-dir", default=str(DEFAULT_PACKET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--system-id", action="append", dest="system_ids", default=[])
    parser.add_argument("--all-systems", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system_ids: list[str] | None = args.system_ids or None
    if args.all_systems and not args.system_ids:
        manifest = _read_json(Path(args.packet_dir) / "manifest.json")
        system_ids = [str(system_id) for system_id in manifest.get("system_ids", [])]
    payload = analyze_visual_hard_slice_exactness(
        packet_dir=args.packet_dir,
        output_dir=args.output_dir,
        system_ids=system_ids,
    )
    response = {
        "output_dir": str(Path(args.output_dir).resolve()),
        **payload["manifest"],
    }
    print(json.dumps(response, indent=2, ensure_ascii=False) if args.json else _markdown(payload["manifest"], payload["system_summary"], payload["exactness_gap_rows"]))


if __name__ == "__main__":
    main()
