from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1n_no_call_rescue_transfer_synthesis"


@dataclass(frozen=True)
class TransferSpec:
    label: str
    no_directive_comparison_dir: Path
    incumbent_label: str
    incumbent_comparison_dir: Path


TRANSFER_SPECS: tuple[TransferSpec, ...] = (
    TransferSpec(
        label="component_value_v10",
        no_directive_comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_component_value_no_call_control_rescue_vs_no_directive_v1",
        incumbent_label="hybrid_label_guard_v8",
        incumbent_comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_component_value_no_call_control_rescue_vs_hybrid_label_guard_v1",
    ),
    TransferSpec(
        label="residual_v8",
        no_directive_comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_residual_no_call_control_rescue_vs_no_directive_v1",
        incumbent_label="hybrid_label_guard_v8",
        incumbent_comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_residual_no_call_control_rescue_vs_hybrid_label_guard_v1",
    ),
    TransferSpec(
        label="post_repair_v7",
        no_directive_comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_post_repair_no_call_control_rescue_vs_no_directive_v1",
        incumbent_label="oblique_code_guard_v7",
        incumbent_comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_post_repair_no_call_control_rescue_vs_code_guard_v1",
    ),
    TransferSpec(
        label="oblique_v7",
        no_directive_comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_oblique_no_call_control_rescue_vs_no_directive_v1",
        incumbent_label="oblique_code_guard_v7",
        incumbent_comparison_dir=ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h1n_oracle_oblique_no_call_control_rescue_vs_code_guard_v1",
    ),
)


def build_h1n_no_call_rescue_transfer_synthesis(
    *, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = [_summary_row(spec) for spec in TRANSFER_SPECS]
    aggregate = _aggregate(rows)
    finding_rows = _findings(rows, aggregate)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "comparison_count": len(rows),
        "total_case_count": aggregate["total_case_count"],
        "no_directive_exact_success_count": aggregate["no_directive_exact_success_count"],
        "v10_exact_success_count": aggregate["v10_exact_success_count"],
        "incumbent_exact_success_count": aggregate["incumbent_exact_success_count"],
        "no_directive_executor_success_count": aggregate["no_directive_executor_success_count"],
        "v10_executor_success_count": aggregate["v10_executor_success_count"],
        "incumbent_executor_success_count": aggregate["incumbent_executor_success_count"],
        "finding_count": len(finding_rows),
    }
    payload = {
        "manifest": manifest,
        "aggregate": aggregate,
        "summary_rows": rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1n_no_call_rescue_transfer_summary.csv", rows)
    _write_csv(tables_dir / "h1n_no_call_rescue_transfer_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _summary_row(spec: TransferSpec) -> dict[str, Any]:
    no_directive = _read_summary(spec.no_directive_comparison_dir)
    incumbent = _read_summary(spec.incumbent_comparison_dir)
    case_count = int(no_directive["shared_case_count"])
    return {
        "label": spec.label,
        "incumbent_label": spec.incumbent_label,
        "case_count": case_count,
        "no_directive_exact_rate": float(no_directive["baseline_exact_rate"]),
        "v10_exact_rate": float(no_directive["candidate_exact_rate"]),
        "delta_exact_vs_no_directive": float(no_directive["delta_exact_rate"]),
        "incumbent_exact_rate": float(incumbent["baseline_exact_rate"]),
        "delta_exact_vs_incumbent": float(incumbent["delta_exact_rate"]),
        "no_directive_executor_rate": float(no_directive["baseline_executor_equivalence_rate"]),
        "v10_executor_rate": float(no_directive["candidate_executor_equivalence_rate"]),
        "delta_executor_vs_no_directive": float(no_directive["delta_executor_equivalence_rate"]),
        "incumbent_executor_rate": float(incumbent["baseline_executor_equivalence_rate"]),
        "delta_executor_vs_incumbent": float(incumbent["delta_executor_equivalence_rate"]),
        "no_directive_exact_success_count": round(float(no_directive["baseline_exact_rate"]) * case_count),
        "v10_exact_success_count": round(float(no_directive["candidate_exact_rate"]) * case_count),
        "incumbent_exact_success_count": round(float(incumbent["baseline_exact_rate"]) * case_count),
        "no_directive_executor_success_count": round(
            float(no_directive["baseline_executor_equivalence_rate"]) * case_count
        ),
        "v10_executor_success_count": round(
            float(no_directive["candidate_executor_equivalence_rate"]) * case_count
        ),
        "incumbent_executor_success_count": round(
            float(incumbent["baseline_executor_equivalence_rate"]) * case_count
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "case_count",
        "no_directive_exact_success_count",
        "v10_exact_success_count",
        "incumbent_exact_success_count",
        "no_directive_executor_success_count",
        "v10_executor_success_count",
        "incumbent_executor_success_count",
    ]
    aggregate = {f"total_{key}" if key == "case_count" else key: sum(int(row[key]) for row in rows) for key in keys}
    aggregate["delta_exact_vs_no_directive"] = (
        aggregate["v10_exact_success_count"] - aggregate["no_directive_exact_success_count"]
    )
    aggregate["delta_executor_vs_no_directive"] = (
        aggregate["v10_executor_success_count"] - aggregate["no_directive_executor_success_count"]
    )
    aggregate["delta_exact_vs_incumbent"] = (
        aggregate["v10_exact_success_count"] - aggregate["incumbent_exact_success_count"]
    )
    aggregate["delta_executor_vs_incumbent"] = (
        aggregate["v10_executor_success_count"] - aggregate["incumbent_executor_success_count"]
    )
    return aggregate


def _findings(rows: list[dict[str, Any]], aggregate: dict[str, int]) -> list[dict[str, str]]:
    positive_vs_incumbent = [
        str(row["label"]) for row in rows if float(row["delta_executor_vs_incumbent"]) > 0.0
    ]
    neutral_vs_incumbent = [
        str(row["label"]) for row in rows if float(row["delta_executor_vs_incumbent"]) == 0.0
    ]
    negative_vs_incumbent = [
        str(row["label"]) for row in rows if float(row["delta_executor_vs_incumbent"]) < 0.0
    ]
    return [
        {
            "finding_id": "large_no_directive_lift",
            "finding": (
                "v10 is a real no-directive harness improvement: "
                f"{aggregate['v10_exact_success_count']}/30 exact versus "
                f"{aggregate['no_directive_exact_success_count']}/30 no-directive, and "
                f"{aggregate['v10_executor_success_count']}/30 executor-equivalent versus "
                f"{aggregate['no_directive_executor_success_count']}/30 no-directive."
            ),
        },
        {
            "finding_id": "not_universal_replacement",
            "finding": (
                "v10 is not a universal replacement for the best specialized profiles: "
                f"{aggregate['v10_exact_success_count']}/30 exact versus incumbents at "
                f"{aggregate['incumbent_exact_success_count']}/30, and "
                f"{aggregate['v10_executor_success_count']}/30 executor-equivalent versus incumbents at "
                f"{aggregate['incumbent_executor_success_count']}/30."
            ),
        },
        {
            "finding_id": "transfer_pattern",
            "finding": (
                "Executor-equivalence versus incumbents is positive on "
                f"{', '.join(positive_vs_incumbent) or 'none'}, tied on "
                f"{', '.join(neutral_vs_incumbent) or 'none'}, and negative on "
                f"{', '.join(negative_vs_incumbent) or 'none'}."
            ),
        },
        {
            "finding_id": "promotion_decision",
            "finding": (
                "Treat v10 as a scoped current-image/no-call activation guard. The next H1o slice should factor "
                "activation rescue, code/negation preservation, and component-label/value disambiguation instead of "
                "stacking broad prose."
            ),
        },
    ]


def _read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "live_replay_comparison.json").read_text(encoding="utf-8"))
    return payload["summary"]


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
        "# H1n No-Call Rescue Transfer Synthesis",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Aggregate",
        "",
        f"- v10 exact successes: `{aggregate['v10_exact_success_count']} / 30`",
        f"- no-directive exact successes: `{aggregate['no_directive_exact_success_count']} / 30`",
        f"- incumbent exact successes: `{aggregate['incumbent_exact_success_count']} / 30`",
        f"- v10 executor-equivalent successes: `{aggregate['v10_executor_success_count']} / 30`",
        f"- no-directive executor-equivalent successes: `{aggregate['no_directive_executor_success_count']} / 30`",
        f"- incumbent executor-equivalent successes: `{aggregate['incumbent_executor_success_count']} / 30`",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- `{row['finding_id']}`: {row['finding']}" for row in payload["finding_rows"])
    lines.extend(
        [
            "",
            "## Packet Rows",
            "",
            _markdown_table(payload["summary_rows"]),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "label",
        "incumbent_label",
        "case_count",
        "v10_exact_rate",
        "delta_exact_vs_no_directive",
        "delta_exact_vs_incumbent",
        "v10_executor_rate",
        "delta_executor_vs_no_directive",
        "delta_executor_vs_incumbent",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build H1n no-call rescue transfer synthesis.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1n_no_call_rescue_transfer_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
