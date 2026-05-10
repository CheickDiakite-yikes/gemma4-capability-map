from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1n_oblique_code_hints_delta"
ARGUMENT_HINTS_PACKET = (
    ROOT / "results" / "tool_probe_replay_live" / "20260509T_h1n_oracle_oblique_argument_hints_execute_v1"
)
CODE_HINTS_PACKET = (
    ROOT / "results" / "tool_probe_replay_live" / "20260509T_h1n_oracle_oblique_code_hints_execute_v1"
)
COMPARISON_PACKET = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1"
)


def analyze_h1n_oblique_code_hints_delta(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    comparison = _read_json(COMPARISON_PACKET / "live_replay_comparison.json")
    baseline_results = _result_index(ARGUMENT_HINTS_PACKET)
    candidate_results = _result_index(CODE_HINTS_PACKET)
    case_rows = [
        _case_row(delta, baseline_results[str(delta["case_id"])], candidate_results[str(delta["case_id"])])
        for delta in comparison["case_deltas"]
    ]
    gain_rows = [row for row in case_rows if row["transition"] == "repair_gain"]
    loss_rows = [row for row in case_rows if row["transition"] == "regression"]
    preserved_rows = [row for row in case_rows if row["transition"] == "preserved_success"]
    finding_rows = _findings(gain_rows, loss_rows, preserved_rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "baseline_packet": str(ARGUMENT_HINTS_PACKET.resolve()),
        "candidate_packet": str(CODE_HINTS_PACKET.resolve()),
        "comparison_packet": str(COMPARISON_PACKET.resolve()),
        "case_count": len(case_rows),
        "gain_count": len(gain_rows),
        "loss_count": len(loss_rows),
        "preserved_success_count": len(preserved_rows),
        "net_executor_equivalence_gain": len(gain_rows) - len(loss_rows),
        "finding_count": len(finding_rows),
    }
    payload = {
        "manifest": manifest,
        "case_rows": case_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1n_oblique_code_hints_case_deltas.csv", case_rows)
    _write_csv(tables_dir / "h1n_oblique_code_hints_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "diagnostic.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "diagnostic.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _case_row(delta: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    expected_call = _first_call(candidate.get("expected_calls", []))
    baseline_call = _first_call(baseline.get("actual_calls", []))
    candidate_call = _first_call(candidate.get("actual_calls", []))
    baseline_args = baseline_call.get("arguments", {})
    candidate_args = candidate_call.get("arguments", {})
    transition = _transition(delta)
    return {
        "case_id": delta["case_id"],
        "family": delta.get("family", ""),
        "transition": transition,
        "classification": _classification(transition, expected_call, baseline_call, candidate_call),
        "expected_tool": expected_call.get("name", ""),
        "baseline_tool": baseline_call.get("name", ""),
        "candidate_tool": candidate_call.get("name", ""),
        "expected_target_query": expected_call.get("arguments", {}).get("target_query", ""),
        "baseline_target_query": baseline_args.get("target_query", ""),
        "candidate_target_query": candidate_args.get("target_query", ""),
        "candidate_selection_id": candidate_args.get("selection_id", ""),
        "candidate_filter_query": candidate_args.get("filter_query", ""),
        "baseline_executor_equivalence": bool(delta.get("baseline_replay_executor_equivalence_match")),
        "candidate_executor_equivalence": bool(delta.get("candidate_replay_executor_equivalence_match")),
        "delta_executor_equivalence": int(delta.get("delta_executor_equivalence_match") or 0),
        "candidate_failure_mode": delta.get("candidate_replay_failure_mode", ""),
    }


def _transition(delta: dict[str, Any]) -> str:
    executor_delta = int(delta.get("delta_executor_equivalence_match") or 0)
    if executor_delta > 0:
        return "repair_gain"
    if executor_delta < 0:
        return "regression"
    if bool(delta.get("candidate_replay_executor_equivalence_match")):
        return "preserved_success"
    return "preserved_miss"


def _classification(
    transition: str,
    expected_call: dict[str, Any],
    baseline_call: dict[str, Any],
    candidate_call: dict[str, Any],
) -> str:
    candidate_args = candidate_call.get("arguments", {})
    if transition == "repair_gain":
        return "code_suffix_or_negated_decoy_repaired"
    if transition == "regression" and candidate_call.get("name") != expected_call.get("name"):
        selection_id = str(candidate_args.get("selection_id", ""))
        filter_query = str(candidate_args.get("filter_query", ""))
        if selection_id or filter_query:
            return "stale_selection_tool_attraction"
        return "wrong_tool_regression"
    if transition == "preserved_success":
        return "preserved_argument_hints_win"
    if baseline_call.get("name") != candidate_call.get("name"):
        return "preserved_miss_with_tool_shift"
    return "preserved_miss"


def _findings(
    gain_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    preserved_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    gain_cases = ", ".join(row["case_id"] for row in gain_rows) or "none"
    loss_cases = ", ".join(row["case_id"] for row in loss_rows) or "none"
    preserved_cases = ", ".join(row["case_id"] for row in preserved_rows) or "none"
    regression_detail = "No regression observed."
    if loss_rows:
        row = loss_rows[0]
        regression_detail = (
            f"{row['case_id']} regresses from the argument-hints exact call to "
            f"{row['candidate_tool']} selection_id={row['candidate_selection_id']} "
            f"filter_query={row['candidate_filter_query']}."
        )
    return [
        {
            "finding_id": "net_gain_with_regression",
            "finding": (
                f"Oblique code hints repairs {len(gain_rows)} cases and regresses {len(loss_rows)} case "
                f"for a net executor-equivalence gain of {len(gain_rows) - len(loss_rows)} case."
            ),
        },
        {
            "finding_id": "repair_cases",
            "finding": f"Repair gains: {gain_cases}.",
        },
        {
            "finding_id": "regression_case",
            "finding": regression_detail,
        },
        {
            "finding_id": "preserved_argument_hints_wins",
            "finding": f"Preserved argument-hints successes: {preserved_cases}.",
        },
        {
            "finding_id": "next_test",
            "finding": (
                "Before broad promotion, run the code-hints profile on earlier oracle/repeat packets and "
                "either constrain stale-selection routing or build a fresh post-repair holdout."
            ),
        },
    ]


def _result_index(packet_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for result_path in sorted(packet_dir.glob("runs/*/probe_results.json")):
        result = _read_json(result_path)[0]
        rows[str(result["case_id"])] = result
    return rows


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
        "# H1n Oblique Code-Hints Delta",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Findings",
        "",
    ]
    for row in payload["finding_rows"]:
        lines.append(f"- `{row['finding_id']}`: {row['finding']}")
    lines.extend(["", "## Case Deltas", "", _markdown_table(payload["case_rows"]), ""])
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "case_id",
        "transition",
        "classification",
        "expected_tool",
        "candidate_tool",
        "expected_target_query",
        "candidate_selection_id",
        "candidate_filter_query",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the H1n oblique code-hints gains and regression.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_h1n_oblique_code_hints_delta(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
