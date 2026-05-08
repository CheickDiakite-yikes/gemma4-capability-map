from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "tool_probe_replay_live_diagnostics"
DEFAULT_PACKET_DIRS = [
    ROOT / "results" / "tool_probe_replay_live" / "20260507T_visual_state_visual_tool_initiation_live_execute_v1",
    ROOT / "results" / "tool_probe_replay_live" / "20260508T_visual_state_tool_selection_live_execute_v1",
    ROOT / "results" / "tool_probe_replay_live" / "20260508T_visual_role_catalog_live_execute_v1",
]


def analyze_visual_tool_choice_diagnostics(
    packet_dirs: list[str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    packet_payloads = [_packet_payload(Path(packet_dir)) for packet_dir in packet_dirs]
    for payload in packet_payloads:
        for result in payload["results"]:
            if not _is_visual_case(result):
                continue
            rows.append(_diagnostic_row(payload, result))

    diagnosis_counts = _count_rows(rows, "diagnosis")
    case_counts = _count_rows(rows, "case_id")
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "packet_count": len(packet_payloads),
        "case_count": len(rows),
        "diagnosis_counts": diagnosis_counts,
        "case_counts": case_counts,
        "case_diagnosis_transitions": _case_diagnosis_transitions(rows),
        "packet_dirs": [str(payload["packet_dir"]) for payload in packet_payloads],
    }
    payload = {
        "summary": summary,
        "rows": rows,
    }
    _write_json(target / "visual_tool_choice_diagnostics.json", payload)
    _write_csv(target / "visual_tool_choice_diagnostics.csv", rows)
    (target / "visual_tool_choice_diagnostics.md").write_text(_markdown(summary, rows), encoding="utf-8")
    return payload


def _packet_payload(packet_dir: Path) -> dict[str, Any]:
    manifest = _read_json(packet_dir / "manifest.json")
    results = _read_json(packet_dir / "live_replay_results.json")
    return {
        "packet_dir": packet_dir.resolve(),
        "manifest": manifest,
        "results": results,
    }


def _diagnostic_row(packet: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    probe_path = Path(result["output_dir"]) / "probe_results.json"
    probe_rows = _read_json(probe_path)
    probe = probe_rows[0] if probe_rows else {}
    expected_calls = probe.get("expected_calls") or []
    actual_calls = probe.get("actual_calls") or []
    expected_names = [str(call.get("name", "")) for call in expected_calls]
    actual_names = [str(call.get("name", "")) for call in actual_calls]
    failure_mode = str(result.get("replay_failure_mode") or "")
    diagnosis = _diagnosis(
        failure_mode=failure_mode,
        expected_names=expected_names,
        actual_names=actual_names,
        exact=bool(result.get("replay_exact_match")),
        executable=result.get("replay_executable_match"),
    )
    return {
        "packet_run_id": packet["manifest"].get("packet_run_id", ""),
        "packet_label": _packet_label(packet),
        "system_id": packet["manifest"].get("system_id", ""),
        "case_id": result.get("case_id", ""),
        "family": result.get("family", ""),
        "expected_tools": " -> ".join(expected_names),
        "actual_tools": " -> ".join(actual_names),
        "expected_arguments": json.dumps([call.get("arguments", {}) for call in expected_calls], sort_keys=True),
        "actual_arguments": json.dumps([call.get("arguments", {}) for call in actual_calls], sort_keys=True),
        "replay_failure_mode": failure_mode,
        "replay_exact_match": bool(result.get("replay_exact_match")),
        "replay_executable_match": _none_to_blank(result.get("replay_executable_match")),
        "expected_call_count": int(result.get("expected_call_count") or 0),
        "actual_call_count": int(result.get("replay_actual_call_count") or 0),
        "diagnosis": diagnosis,
        "next_diagnostic": _next_diagnostic(diagnosis, expected_names, actual_names),
        "raw_model_output": str(probe.get("raw_model_output", "")),
    }


def _diagnosis(
    *,
    failure_mode: str,
    expected_names: list[str],
    actual_names: list[str],
    exact: bool,
    executable: Any,
) -> str:
    if exact:
        return "exact"
    if executable is True:
        return "tool_ok_argument_alias_executable"
    if not actual_names or failure_mode == "no_tool_call":
        return "visual_tool_initiation_missing"
    if expected_names and actual_names and expected_names[0] != actual_names[0]:
        return "wrong_visual_tool_selection"
    if failure_mode == "argument_mismatch":
        return "visual_literal_argument_mismatch"
    if failure_mode == "call_count_mismatch":
        return "visual_call_count_mismatch"
    if failure_mode in {"wrong_tool", "executable_paraphrase"}:
        return "visual_argument_or_selector_mismatch"
    return failure_mode or "non_exact"


def _next_diagnostic(diagnosis: str, expected_names: list[str], actual_names: list[str]) -> str:
    expected_first = expected_names[0] if expected_names else ""
    actual_first = actual_names[0] if actual_names else ""
    if diagnosis == "visual_tool_initiation_missing":
        return "preserve visual tool initiation before tuning selectors"
    if diagnosis == "wrong_visual_tool_selection" and expected_first == "refine_selection":
        return f"separate latest-selection filtering from locating/readback; actual first tool was {actual_first}"
    if diagnosis == "wrong_visual_tool_selection":
        return f"inspect tool catalog/routing priority for expected {expected_first}"
    if diagnosis == "tool_ok_argument_alias_executable":
        return "tighten canonical visual argument copy without losing executable aliases"
    if diagnosis == "visual_literal_argument_mismatch":
        return "preserve literal visual selector arguments after correct routing"
    if diagnosis == "visual_argument_or_selector_mismatch":
        return "compare expected and actual visual ids/queries for canonical selector drift"
    return "no further diagnostic needed"


def _is_visual_case(result: dict[str, Any]) -> bool:
    family = str(result.get("family", ""))
    case_id = str(result.get("case_id", ""))
    return family.startswith("visual") or case_id.startswith("visual_")


def _count_rows(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def _case_diagnosis_transitions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    transitions: dict[str, list[str]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        packet_label = str(row.get("packet_label") or row.get("packet_run_id") or "")
        diagnosis = str(row.get("diagnosis", ""))
        transitions.setdefault(case_id, []).append(f"{packet_label}:{diagnosis}")
    return dict(sorted(transitions.items()))


def _packet_label(packet: dict[str, Any]) -> str:
    manifest = packet["manifest"]
    explicit = str(manifest.get("label", "")).strip()
    if explicit:
        return explicit
    system_id = str(manifest.get("system_id", ""))
    labels = {
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation": "visual_tool_initiation_v3",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection": "visual_state_tool_selection_v4",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog": "visual_role_catalog_v1",
    }
    return labels.get(system_id, str(manifest.get("packet_run_id", "")))


def _markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Visual Tool-Choice Diagnostics",
        "",
        f"- Packet count: `{summary['packet_count']}`",
        f"- Visual case rows: `{summary['case_count']}`",
        f"- Diagnosis counts: `{summary['diagnosis_counts']}`",
        "",
        "| packet | label | system | case | expected | actual | failure | diagnosis | next diagnostic |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["packet_run_id"]),
                    str(row["packet_label"]),
                    str(row["system_id"]),
                    str(row["case_id"]),
                    str(row["expected_tools"]),
                    str(row["actual_tools"]),
                    str(row["replay_failure_mode"]),
                    str(row["diagnosis"]),
                    str(row["next_diagnostic"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _none_to_blank(value: Any) -> Any:
    return "" if value is None else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze visual tool choices from CLI-live replay packets.")
    parser.add_argument(
        "packet_dirs",
        nargs="*",
        help="One or more tool_probe_replay_live packet directories. Defaults to wave three, wave four, and visual-role catalog packets.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_dirs = args.packet_dirs or [str(path) for path in DEFAULT_PACKET_DIRS]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}_visual_tool_choice_diagnostics"
    payload = analyze_visual_tool_choice_diagnostics(packet_dirs, output_dir=output_dir)
    response = {
        "output_dir": str(Path(output_dir).resolve()),
        **payload["summary"],
    }
    print(json.dumps(response, indent=2, ensure_ascii=False) if args.json else _markdown(payload["summary"], payload["rows"]))


if __name__ == "__main__":
    main()
