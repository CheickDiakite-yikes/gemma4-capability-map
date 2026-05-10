from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1n_code_guard_transfer_synthesis"


@dataclass(frozen=True)
class ProfileComparison:
    label: str
    profile: str
    comparison_dir: Path
    baseline_profile: str


ARGUMENT_HINT_COMPARISONS: tuple[ProfileComparison, ...] = (
    ProfileComparison(
        label="oracle_transfer_v2",
        profile="code_guard_v7",
        baseline_profile="argument_hints_v2",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_code_guard_vs_argument_hints_transfer_v1",
    ),
    ProfileComparison(
        label="oracle_repeat_v1",
        profile="code_guard_v7",
        baseline_profile="argument_hints_v2",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_repeat_code_guard_vs_argument_hints_transfer_v1",
    ),
    ProfileComparison(
        label="oblique_v5",
        profile="code_guard_v7",
        baseline_profile="argument_hints_v2",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_oblique_code_guard_vs_argument_hints_v1",
    ),
)

CODE_HINT_COMPARISONS: tuple[ProfileComparison, ...] = (
    ProfileComparison(
        label="oracle_transfer_v2",
        profile="code_guard_v7",
        baseline_profile="code_hints_v6",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_code_guard_vs_code_hints_transfer_v1",
    ),
    ProfileComparison(
        label="oracle_repeat_v1",
        profile="code_guard_v7",
        baseline_profile="code_hints_v6",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_repeat_code_guard_vs_code_hints_transfer_v1",
    ),
    ProfileComparison(
        label="oblique_v5",
        profile="code_guard_v7",
        baseline_profile="code_hints_v6",
        comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_oblique_code_guard_vs_code_hints_v1",
    ),
)


def build_h1n_code_guard_transfer_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    argument_rows = [_summary_row(spec) for spec in ARGUMENT_HINT_COMPARISONS]
    code_hint_rows = [_summary_row(spec) for spec in CODE_HINT_COMPARISONS]
    argument_aggregate = _aggregate(argument_rows)
    code_hint_aggregate = _aggregate(code_hint_rows)
    finding_rows = _findings(argument_rows, argument_aggregate, code_hint_aggregate)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "argument_hint_comparison_count": len(argument_rows),
        "code_hint_comparison_count": len(code_hint_rows),
        "total_case_count": argument_aggregate["total_case_count"],
        "argument_hints_exact_success_count": argument_aggregate["baseline_exact_success_count"],
        "code_guard_exact_success_count": argument_aggregate["candidate_exact_success_count"],
        "argument_hints_executor_success_count": argument_aggregate["baseline_executor_success_count"],
        "code_guard_executor_success_count": argument_aggregate["candidate_executor_success_count"],
        "code_hints_exact_success_count": code_hint_aggregate["baseline_exact_success_count"],
        "code_hints_executor_success_count": code_hint_aggregate["baseline_executor_success_count"],
        "finding_count": len(finding_rows),
    }
    payload = {
        "manifest": manifest,
        "argument_hint_aggregate": argument_aggregate,
        "code_hint_aggregate": code_hint_aggregate,
        "argument_hint_rows": argument_rows,
        "code_hint_rows": code_hint_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1n_code_guard_vs_argument_hints_summary.csv", argument_rows)
    _write_csv(tables_dir / "h1n_code_guard_vs_code_hints_summary.csv", code_hint_rows)
    _write_csv(tables_dir / "h1n_code_guard_transfer_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _summary_row(spec: ProfileComparison) -> dict[str, Any]:
    payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
    summary = payload["summary"]
    shared_case_count = int(summary["shared_case_count"])
    baseline_exact_rate = float(summary["baseline_exact_rate"])
    candidate_exact_rate = float(summary["candidate_exact_rate"])
    baseline_executor_rate = float(summary["baseline_executor_equivalence_rate"])
    candidate_executor_rate = float(summary["candidate_executor_equivalence_rate"])
    return {
        "label": spec.label,
        "profile": spec.profile,
        "baseline_profile": spec.baseline_profile,
        "comparison_dir": str(spec.comparison_dir.relative_to(ROOT)),
        "shared_case_count": shared_case_count,
        "baseline_exact_rate": baseline_exact_rate,
        "candidate_exact_rate": candidate_exact_rate,
        "delta_exact_rate": float(summary["delta_exact_rate"]),
        "baseline_executor_equivalence_rate": baseline_executor_rate,
        "candidate_executor_equivalence_rate": candidate_executor_rate,
        "delta_executor_equivalence_rate": float(summary["delta_executor_equivalence_rate"]),
        "baseline_exact_success_count": round(baseline_exact_rate * shared_case_count),
        "candidate_exact_success_count": round(candidate_exact_rate * shared_case_count),
        "baseline_executor_success_count": round(baseline_executor_rate * shared_case_count),
        "candidate_executor_success_count": round(candidate_executor_rate * shared_case_count),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_case_count = sum(int(row["shared_case_count"]) for row in rows)
    baseline_exact = sum(int(row["baseline_exact_success_count"]) for row in rows)
    candidate_exact = sum(int(row["candidate_exact_success_count"]) for row in rows)
    baseline_executor = sum(int(row["baseline_executor_success_count"]) for row in rows)
    candidate_executor = sum(int(row["candidate_executor_success_count"]) for row in rows)
    return {
        "total_case_count": total_case_count,
        "baseline_exact_success_count": baseline_exact,
        "candidate_exact_success_count": candidate_exact,
        "delta_exact_success_count": candidate_exact - baseline_exact,
        "baseline_executor_success_count": baseline_executor,
        "candidate_executor_success_count": candidate_executor,
        "delta_executor_success_count": candidate_executor - baseline_executor,
    }


def _findings(
    argument_rows: list[dict[str, Any]],
    argument_aggregate: dict[str, Any],
    code_hint_aggregate: dict[str, Any],
) -> list[dict[str, str]]:
    positive_vs_argument = [
        str(row["label"])
        for row in argument_rows
        if float(row["delta_executor_equivalence_rate"]) > 0.0
    ]
    negative_vs_argument = [
        str(row["label"])
        for row in argument_rows
        if float(row["delta_executor_equivalence_rate"]) < 0.0
    ]
    return [
        {
            "finding_id": "code_guard_beats_v6",
            "finding": (
                "Code guard improves on v6 across the three-packet aggregate: "
                f"{argument_aggregate['candidate_exact_success_count']}/18 exact and "
                f"{argument_aggregate['candidate_executor_success_count']}/18 executor-equivalent versus "
                f"v6 at {code_hint_aggregate['baseline_exact_success_count']}/18 exact and "
                f"{code_hint_aggregate['baseline_executor_success_count']}/18 executor-equivalent."
            ),
        },
        {
            "finding_id": "argument_hints_still_best_executor",
            "finding": (
                "Argument hints remains the stronger executor-equivalence baseline overall: "
                f"{argument_aggregate['baseline_executor_success_count']}/18 versus code guard at "
                f"{argument_aggregate['candidate_executor_success_count']}/18."
            ),
        },
        {
            "finding_id": "oblique_only_positive_vs_argument_hints",
            "finding": (
                "Code guard is positive versus argument hints only on "
                f"{', '.join(positive_vs_argument) or 'none'} and negative on "
                f"{', '.join(negative_vs_argument) or 'none'}."
            ),
        },
        {
            "finding_id": "promotion_decision",
            "finding": (
                "Treat code guard as a better scoped repair than v6, not a broad replacement for argument hints; "
                "the next proof point should be a fresh post-repair holdout."
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
    argument = payload["argument_hint_aggregate"]
    code_hint = payload["code_hint_aggregate"]
    lines = [
        "# H1n Code-Guard Transfer Synthesis",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Aggregate",
        "",
        f"- argument_hints_exact_success_count: `{argument['baseline_exact_success_count']}`",
        f"- code_guard_exact_success_count: `{argument['candidate_exact_success_count']}`",
        f"- argument_hints_executor_success_count: `{argument['baseline_executor_success_count']}`",
        f"- code_guard_executor_success_count: `{argument['candidate_executor_success_count']}`",
        f"- code_hints_exact_success_count: `{code_hint['baseline_exact_success_count']}`",
        f"- code_hints_executor_success_count: `{code_hint['baseline_executor_success_count']}`",
        "",
        "## Findings",
        "",
    ]
    for row in payload["finding_rows"]:
        lines.append(f"- `{row['finding_id']}`: {row['finding']}")
    lines.extend(["", "## Code Guard vs Argument Hints", "", _markdown_table(payload["argument_hint_rows"])])
    lines.extend(["", "## Code Guard vs Code Hints", "", _markdown_table(payload["code_hint_rows"]), ""])
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "label",
        "baseline_profile",
        "candidate_exact_rate",
        "delta_exact_rate",
        "candidate_executor_equivalence_rate",
        "delta_executor_equivalence_rate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize H1n code-guard transfer across oracle packets.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1n_code_guard_transfer_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
