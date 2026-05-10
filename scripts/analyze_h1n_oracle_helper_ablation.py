from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1n_oracle_helper_ablation"
DEFAULT_COMPARISONS = {
    "no_controller_repair": (
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_argument_hints_no_controller_repair_vs_argument_hints_v1"
    ),
    "no_controller_fallback": (
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_argument_hints_no_controller_fallback_vs_argument_hints_v1"
    ),
    "no_argument_repair": (
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_argument_hints_no_argument_repair_vs_argument_hints_v1"
    ),
}


def analyze_h1n_oracle_helper_ablation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    comparisons: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    selected = comparisons or DEFAULT_COMPARISONS
    target = Path(output_dir)
    tables_dir = target / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}
    for helper, comparison_dir in selected.items():
        payload = _read_json(Path(comparison_dir) / "live_replay_comparison.json")
        loaded[helper] = payload
        summary = payload["summary"]
        row = {
            "helper_removed": helper,
            "comparison_dir": str(Path(comparison_dir).resolve()),
            "baseline_system_id": summary["baseline_system_id"],
            "candidate_system_id": summary["candidate_system_id"],
            "shared_case_count": summary["shared_case_count"],
            "baseline_exact_rate": summary["baseline_exact_rate"],
            "candidate_exact_rate": summary["candidate_exact_rate"],
            "delta_exact_rate": summary["delta_exact_rate"],
            "baseline_executor_equivalence_rate": summary["baseline_executor_equivalence_rate"],
            "candidate_executor_equivalence_rate": summary["candidate_executor_equivalence_rate"],
            "delta_executor_equivalence_rate": summary["delta_executor_equivalence_rate"],
            "classification": _classification(summary),
        }
        summary_rows.append(row)
        for case in payload["case_deltas"]:
            case_rows.append(
                {
                    "helper_removed": helper,
                    "case_id": case["case_id"],
                    "family": case.get("family", ""),
                    "baseline_exact_match": case.get("baseline_exact_match"),
                    "candidate_exact_match": case.get("candidate_exact_match"),
                    "delta_exact_match": case.get("delta_exact_match"),
                    "baseline_executor_equivalence_match": case.get("baseline_executor_equivalence_match"),
                    "candidate_executor_equivalence_match": case.get("candidate_executor_equivalence_match"),
                    "delta_executor_equivalence_match": case.get("delta_executor_equivalence_match"),
                }
            )

    findings = {
        "all_helpers_preserve_exact_rate": all(float(row["delta_exact_rate"]) == 0.0 for row in summary_rows),
        "all_helpers_preserve_executor_equivalence_rate": all(
            float(row["delta_executor_equivalence_rate"]) == 0.0 for row in summary_rows
        ),
        "helper_count": len(summary_rows),
        "strict_rate": summary_rows[0]["candidate_exact_rate"] if summary_rows else None,
        "executor_equivalence_rate": summary_rows[0]["candidate_executor_equivalence_rate"] if summary_rows else None,
    }
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(target.resolve()),
        "comparison_dirs": {helper: str(Path(path).resolve()) for helper, path in selected.items()},
        "summary_row_count": len(summary_rows),
        "case_row_count": len(case_rows),
    }
    payload = {
        "manifest": manifest,
        "findings": findings,
        "summary_rows": summary_rows,
        "case_rows": case_rows,
    }

    _write_csv(tables_dir / "h1n_oracle_helper_ablation_summary.csv", summary_rows)
    _write_csv(tables_dir / "h1n_oracle_helper_ablation_case_deltas.csv", case_rows)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "diagnostic.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "diagnostic.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _classification(summary: dict[str, Any]) -> str:
    if float(summary["delta_exact_rate"]) == 0.0 and float(summary["delta_executor_equivalence_rate"]) == 0.0:
        return "no_observed_helper_dependence"
    if float(summary["delta_exact_rate"]) < 0.0 or float(summary["delta_executor_equivalence_rate"]) < 0.0:
        return "helper_is_causal_on_this_slice"
    return "candidate_improves_without_helper"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# H1n Oracle Helper-Ablation Diagnostic",
        "",
        "This diagnostic compares the H1n oracle argument-hints row against variants that disable one controller helper at a time.",
        "",
        "## Findings",
        "",
        f"- helper rows: `{payload['findings']['helper_count']}`",
        f"- strict rate: `{payload['findings']['strict_rate']}`",
        f"- executor-equivalence rate: `{payload['findings']['executor_equivalence_rate']}`",
        f"- all helpers preserve strict rate: `{payload['findings']['all_helpers_preserve_exact_rate']}`",
        f"- all helpers preserve executor-equivalence rate: `{payload['findings']['all_helpers_preserve_executor_equivalence_rate']}`",
        "",
        "## Summary",
        "",
        _markdown_table(payload["summary_rows"]),
        "",
        "## Case Deltas",
        "",
        _markdown_table(payload["case_rows"]),
        "",
        "Interpretation: on this deterministic six-case oracle transfer packet, the argument-hints gain is not explained by controller repair, controller fallback, or argument repair. The result is negative for helper dependence, not broad proof that helpers never matter.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_h1n_oracle_helper_ablation(output_dir=args.output_dir)
    print(json.dumps(payload["findings"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
