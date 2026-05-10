from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1n_code_hints_transfer_synthesis"


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    comparison_dir: Path
    interpretation: str


COMPARISONS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        label="oracle_transfer_v2",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_code_hints_vs_argument_hints_transfer_v1",
        interpretation="negative_transfer",
    ),
    ComparisonSpec(
        label="oracle_repeat_v1",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_repeat_code_hints_vs_argument_hints_transfer_v1",
        interpretation="negative_transfer",
    ),
    ComparisonSpec(
        label="oblique_v5",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1",
        interpretation="localized_repair",
    ),
)


def build_h1n_code_hints_transfer_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [_summary_row(spec) for spec in COMPARISONS]
    aggregate = _aggregate(summary_rows)
    finding_rows = _findings(summary_rows, aggregate)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "comparison_count": len(summary_rows),
        "total_case_count": aggregate["total_case_count"],
        "argument_hints_exact_success_count": aggregate["baseline_exact_success_count"],
        "code_hints_exact_success_count": aggregate["candidate_exact_success_count"],
        "argument_hints_executor_success_count": aggregate["baseline_executor_success_count"],
        "code_hints_executor_success_count": aggregate["candidate_executor_success_count"],
        "finding_count": len(finding_rows),
    }
    payload = {
        "manifest": manifest,
        "aggregate": aggregate,
        "summary_rows": summary_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1n_code_hints_transfer_summary.csv", summary_rows)
    _write_csv(tables_dir / "h1n_code_hints_transfer_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _summary_row(spec: ComparisonSpec) -> dict[str, Any]:
    payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
    summary = payload["summary"]
    shared_case_count = int(summary["shared_case_count"])
    baseline_exact_rate = float(summary["baseline_exact_rate"])
    candidate_exact_rate = float(summary["candidate_exact_rate"])
    baseline_executor_rate = float(summary["baseline_executor_equivalence_rate"])
    candidate_executor_rate = float(summary["candidate_executor_equivalence_rate"])
    return {
        "label": spec.label,
        "interpretation": spec.interpretation,
        "comparison_dir": str(spec.comparison_dir.relative_to(ROOT)),
        "shared_case_count": shared_case_count,
        "argument_hints_exact_rate": baseline_exact_rate,
        "code_hints_exact_rate": candidate_exact_rate,
        "delta_exact_rate": float(summary["delta_exact_rate"]),
        "argument_hints_executor_equivalence_rate": baseline_executor_rate,
        "code_hints_executor_equivalence_rate": candidate_executor_rate,
        "delta_executor_equivalence_rate": float(summary["delta_executor_equivalence_rate"]),
        "argument_hints_exact_success_count": round(baseline_exact_rate * shared_case_count),
        "code_hints_exact_success_count": round(candidate_exact_rate * shared_case_count),
        "argument_hints_executor_success_count": round(baseline_executor_rate * shared_case_count),
        "code_hints_executor_success_count": round(candidate_executor_rate * shared_case_count),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_case_count = sum(int(row["shared_case_count"]) for row in rows)
    baseline_exact = sum(int(row["argument_hints_exact_success_count"]) for row in rows)
    candidate_exact = sum(int(row["code_hints_exact_success_count"]) for row in rows)
    baseline_executor = sum(int(row["argument_hints_executor_success_count"]) for row in rows)
    candidate_executor = sum(int(row["code_hints_executor_success_count"]) for row in rows)
    return {
        "total_case_count": total_case_count,
        "baseline_exact_success_count": baseline_exact,
        "candidate_exact_success_count": candidate_exact,
        "delta_exact_success_count": candidate_exact - baseline_exact,
        "baseline_executor_success_count": baseline_executor,
        "candidate_executor_success_count": candidate_executor,
        "delta_executor_success_count": candidate_executor - baseline_executor,
    }


def _findings(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> list[dict[str, str]]:
    negative_labels = [
        str(row["label"])
        for row in rows
        if float(row["delta_executor_equivalence_rate"]) < 0.0
    ]
    positive_labels = [
        str(row["label"])
        for row in rows
        if float(row["delta_executor_equivalence_rate"]) > 0.0
    ]
    return [
        {
            "finding_id": "localized_oblique_repair",
            "finding": (
                "Code hints improves only the oblique code-label packet; "
                f"positive transfer labels: {', '.join(positive_labels) or 'none'}."
            ),
        },
        {
            "finding_id": "negative_transfer_elsewhere",
            "finding": (
                "Code hints regresses against argument hints on earlier oracle transfer surfaces; "
                f"negative transfer labels: {', '.join(negative_labels) or 'none'}."
            ),
        },
        {
            "finding_id": "aggregate_exactness",
            "finding": (
                "Across the three H1n oracle packets, argument hints has "
                f"{aggregate['baseline_exact_success_count']}/{aggregate['total_case_count']} exact successes "
                f"versus code hints at {aggregate['candidate_exact_success_count']}/{aggregate['total_case_count']}."
            ),
        },
        {
            "finding_id": "aggregate_executor_equivalence",
            "finding": (
                "Across the three H1n oracle packets, argument hints has "
                f"{aggregate['baseline_executor_success_count']}/{aggregate['total_case_count']} executor-equivalent successes "
                f"versus code hints at {aggregate['candidate_executor_success_count']}/{aggregate['total_case_count']}."
            ),
        },
        {
            "finding_id": "promotion_decision",
            "finding": (
                "Keep oblique code hints as a scoped repair candidate, not a replacement for argument hints, "
                "until a stale-selection guard or fresh holdout reverses the transfer loss."
            ),
        },
    ]


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
    aggregate = payload["aggregate"]
    lines = [
        "# H1n Code-Hints Transfer Synthesis",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Aggregate",
        "",
        f"- total_case_count: `{aggregate['total_case_count']}`",
        f"- argument_hints_exact_success_count: `{aggregate['baseline_exact_success_count']}`",
        f"- code_hints_exact_success_count: `{aggregate['candidate_exact_success_count']}`",
        f"- argument_hints_executor_success_count: `{aggregate['baseline_executor_success_count']}`",
        f"- code_hints_executor_success_count: `{aggregate['candidate_executor_success_count']}`",
        "",
        "## Findings",
        "",
    ]
    for row in payload["finding_rows"]:
        lines.append(f"- `{row['finding_id']}`: {row['finding']}")
    lines.extend(["", "## Comparison Summary", "", _markdown_table(payload["summary_rows"]), ""])
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "label",
        "interpretation",
        "argument_hints_exact_rate",
        "code_hints_exact_rate",
        "delta_exact_rate",
        "argument_hints_executor_equivalence_rate",
        "code_hints_executor_equivalence_rate",
        "delta_executor_equivalence_rate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize H1n oblique-code transfer across oracle packets.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1n_code_hints_transfer_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
