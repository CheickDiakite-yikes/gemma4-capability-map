from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1n_oracle_transfer_synthesis"
ORACLE_SUMMARY = (
    ROOT
    / "results"
    / "reports"
    / "visual_alias_transfer_oracle_diagnostic"
    / "tables"
    / "alias_transfer_oracle_matrix_summary.csv"
)
REPEAT_SUMMARY = (
    ROOT
    / "results"
    / "reports"
    / "visual_alias_transfer_repeat_diagnostic"
    / "tables"
    / "alias_transfer_repeat_matrix_summary.csv"
)
HELPER_SUMMARY = (
    ROOT
    / "results"
    / "reports"
    / "h1n_oracle_helper_ablation"
    / "tables"
    / "h1n_oracle_helper_ablation_summary.csv"
)


def build_h1n_oracle_transfer_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    target = Path(output_dir)
    tables_dir = target / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    oracle_rows = _csv_rows(ORACLE_SUMMARY)
    repeat_rows = _csv_rows(REPEAT_SUMMARY)
    helper_rows = _csv_rows(HELPER_SUMMARY)
    synthesis_rows = [
        *_surface_rows("oracle_v2", oracle_rows),
        *_surface_rows("repeat_v1", repeat_rows),
    ]
    findings = _findings(oracle_rows, repeat_rows, helper_rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(target.resolve()),
        "oracle_summary": str(ORACLE_SUMMARY.resolve()),
        "repeat_summary": str(REPEAT_SUMMARY.resolve()),
        "helper_summary": str(HELPER_SUMMARY.resolve()),
        "surface_row_count": len(synthesis_rows),
        "helper_row_count": len(helper_rows),
        "finding_count": len(findings),
    }
    payload = {
        "manifest": manifest,
        "synthesis_rows": synthesis_rows,
        "helper_rows": helper_rows,
        "findings": findings,
    }
    _write_csv(tables_dir / "h1n_oracle_transfer_synthesis.csv", synthesis_rows)
    _write_csv(tables_dir / "h1n_oracle_helper_synthesis.csv", helper_rows)
    _write_csv(tables_dir / "h1n_oracle_transfer_findings.csv", findings)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _surface_rows(surface: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "surface": surface,
            "label": row["label"],
            "shared_case_count": int(row["shared_case_count"]),
            "candidate_exact_rate": float(row["candidate_exact_rate"]),
            "candidate_executor_equivalence_rate": float(row["candidate_executor_equivalence_rate"]),
            "delta_exact_rate": float(row["delta_exact_rate"]),
            "delta_executor_equivalence_rate": float(row["delta_executor_equivalence_rate"]),
        }
        for row in rows
    ]


def _findings(
    oracle_rows: list[dict[str, str]],
    repeat_rows: list[dict[str, str]],
    helper_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    oracle_exact = _best_labels(oracle_rows, "candidate_exact_rate")
    oracle_executor = _best_labels(oracle_rows, "candidate_executor_equivalence_rate")
    repeat_exact = _best_labels(repeat_rows, "candidate_exact_rate")
    repeat_executor = _best_labels(repeat_rows, "candidate_executor_equivalence_rate")
    helper_preserved = all(float(row["delta_exact_rate"]) == 0.0 and float(row["delta_executor_equivalence_rate"]) == 0.0 for row in helper_rows)
    contracted_repeat = next(row for row in repeat_rows if row["label"] == "contracted")
    return [
        {
            "finding_id": "oracle_v2_winner",
            "finding": f"Oracle v2 winner set: exact={', '.join(oracle_exact)}, executor={', '.join(oracle_executor)}.",
        },
        {
            "finding_id": "repeat_winner_set",
            "finding": f"Repeat winner set: exact={', '.join(repeat_exact)}, executor={', '.join(repeat_executor)}.",
        },
        {
            "finding_id": "helper_dependence",
            "finding": f"Argument-hints helper ablations preserve both metrics: {helper_preserved}.",
        },
        {
            "finding_id": "contracted_not_upper_bound",
            "finding": (
                "Contracted repeat candidate exact/executor rates are "
                f"{contracted_repeat['candidate_exact_rate']} / {contracted_repeat['candidate_executor_equivalence_rate']}."
            ),
        },
    ]


def _best_labels(rows: list[dict[str, str]], field: str) -> list[str]:
    best = max(float(row[field]) for row in rows)
    return [row["label"] for row in rows if float(row[field]) == best]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        "# H1n Oracle Transfer Synthesis",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Findings",
        "",
    ]
    for row in payload["findings"]:
        lines.append(f"- `{row['finding_id']}`: {row['finding']}")
    lines.extend(
        [
            "",
            "## Transfer Surfaces",
            "",
            _markdown_table(payload["synthesis_rows"]),
            "",
            "## Helper Ablation",
            "",
            _markdown_table(payload["helper_rows"]),
            "",
            "Interpretation: H1n now has a two-packet oracle-backed transfer result. Argument hints wins the first oracle packet and ties schema target literals on the repeat. The argument-hints gain is not explained by the three tested controller helpers, while contracted prompting is not a reliable upper bound on these transfer packets.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact H1n oracle-transfer synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1n_oracle_transfer_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
