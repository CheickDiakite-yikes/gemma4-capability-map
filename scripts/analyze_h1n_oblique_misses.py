from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1n_oblique_miss_analysis"
ARGUMENT_HINTS_PACKET = (
    ROOT / "results" / "tool_probe_replay_live" / "20260509T_h1n_oracle_oblique_argument_hints_execute_v1"
)
SCHEMA_FIELD_PACKET = (
    ROOT / "results" / "tool_probe_replay_live" / "20260509T_h1n_oracle_oblique_schema_field_hints_execute_v1"
)


def analyze_h1n_oblique_misses(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    argument_rows = _packet_rows(ARGUMENT_HINTS_PACKET, "argument_hints_v2")
    schema_rows = _packet_rows(SCHEMA_FIELD_PACKET, "schema_field_hints_v4")
    miss_rows = [row for row in [*argument_rows, *schema_rows] if not row["executor_target_match"]]
    finding_rows = _findings(argument_rows, schema_rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "argument_hints_packet": str(ARGUMENT_HINTS_PACKET.resolve()),
        "schema_field_packet": str(SCHEMA_FIELD_PACKET.resolve()),
        "row_count": len(argument_rows) + len(schema_rows),
        "miss_count": len(miss_rows),
        "finding_count": len(finding_rows),
    }
    payload = {
        "manifest": manifest,
        "miss_rows": miss_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1n_oblique_misses.csv", miss_rows)
    _write_csv(tables_dir / "h1n_oblique_miss_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "diagnostic.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "diagnostic.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _packet_rows(packet_dir: Path, label: str) -> list[dict[str, Any]]:
    live_results = _read_json(packet_dir / "live_replay_results.json")
    rows = []
    for row in live_results:
        case_id = str(row["case_id"])
        result_path = packet_dir / "runs" / case_id / "probe_results.json"
        result = _read_json(result_path)[0]
        expected_call = _first_call(result.get("expected_calls", []))
        actual_call = _first_call(result.get("actual_calls", []))
        actual_execution = result.get("actual_execution", [])
        actual_output = actual_execution[-1].get("output", {}) if actual_execution else {}
        expected_regions = [str(region_id) for region_id in result.get("expected_execution", {}).get("region_ids", [])]
        actual_regions = [str(region_id) for region_id in actual_output.get("region_ids", [])]
        rows.append(
            {
                "label": label,
                "case_id": case_id,
                "family": row.get("family", ""),
                "failure_mode": row.get("replay_failure_mode", ""),
                "expected_tool": expected_call.get("name", ""),
                "actual_tool": actual_call.get("name", ""),
                "expected_target_query": expected_call.get("arguments", {}).get("target_query", ""),
                "actual_target_query": actual_call.get("arguments", {}).get("target_query", ""),
                "expected_region_ids": json.dumps(expected_regions),
                "actual_region_ids": json.dumps(actual_regions),
                "exact_match": bool(row.get("replay_exact_match")),
                "executor_target_match": bool(row.get("replay_executor_equivalence_match")),
                "classification": _classify(
                    expected_query=str(expected_call.get("arguments", {}).get("target_query", "")),
                    actual_query=str(actual_call.get("arguments", {}).get("target_query", "")),
                    failure_mode=str(row.get("replay_failure_mode", "")),
                    expected_regions=expected_regions,
                    actual_regions=actual_regions,
                ),
                "result_path": str(result_path.resolve()),
            }
        )
    return rows


def _classify(
    *,
    expected_query: str,
    actual_query: str,
    failure_mode: str,
    expected_regions: list[str],
    actual_regions: list[str],
) -> str:
    if failure_mode == "no_tool_call":
        return "tool_entry_failure"
    if not actual_query:
        return "missing_visual_query"
    expected_tokens = expected_query.split()
    actual_tokens = actual_query.split()
    if expected_tokens and actual_tokens and actual_tokens == expected_tokens[: len(actual_tokens)]:
        return "code_suffix_truncation"
    if expected_regions and set(expected_regions).issubset(set(actual_regions)) and len(actual_regions) > len(expected_regions):
        return "semantic_broad_selection"
    if actual_query and actual_query not in expected_query:
        return "negated_or_semantic_decoy_selected"
    return "argument_mismatch_other"


def _findings(argument_rows: list[dict[str, Any]], schema_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    argument_misses = [row for row in argument_rows if not row["executor_target_match"]]
    schema_misses = [row for row in schema_rows if not row["executor_target_match"]]
    argument_classes = ", ".join(sorted({str(row["classification"]) for row in argument_misses}))
    schema_classes = ", ".join(sorted({str(row["classification"]) for row in schema_misses}))
    return [
        {
            "finding_id": "argument_hints_miss_count",
            "finding": f"Argument hints has {len(argument_misses)} misses: {argument_classes}.",
        },
        {
            "finding_id": "schema_field_miss_count",
            "finding": f"Schema-field hints has {len(schema_misses)} misses: {schema_classes}.",
        },
        {
            "finding_id": "next_intervention_target",
            "finding": (
                "Next target should preserve short code suffixes and negated visible-target instructions, "
                "not revive broad schema-target-literal wording."
            ),
        },
    ]


def _first_call(calls: Any) -> dict[str, Any]:
    if isinstance(calls, list) and calls and isinstance(calls[0], dict):
        return calls[0]
    return {"name": "", "arguments": {}}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# H1n Oblique Miss Analysis",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Findings",
        "",
    ]
    for row in payload["finding_rows"]:
        lines.append(f"- `{row['finding_id']}`: {row['finding']}")
    lines.extend(["", "## Misses", "", _markdown_table(payload["miss_rows"]), ""])
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No misses._"
    headers = [
        "label",
        "case_id",
        "expected_target_query",
        "actual_target_query",
        "actual_region_ids",
        "classification",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze H1n oblique-label replay misses.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_h1n_oblique_misses(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
