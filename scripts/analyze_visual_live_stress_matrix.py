from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_live_stress_diagnostic"
DEFAULT_ALIAS_REPEAT_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_alias_repeat_diagnostic"
DEFAULT_ALIAS_TRANSFER_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_alias_transfer_diagnostic"
DEFAULT_ALIAS_TRANSFER_ORACLE_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_alias_transfer_oracle_diagnostic"
DEFAULT_ALIAS_TRANSFER_REPEAT_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_alias_transfer_repeat_diagnostic"
DEFAULT_ALIAS_TRANSFER_OBLIQUE_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_alias_transfer_oblique_diagnostic"
DEFAULT_COMPARISONS: tuple[tuple[str, Path], ...] = (
    (
        "contracted",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_contracted_vs_no_directive_v1",
    ),
    (
        "role_catalog_v1",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_role_catalog_vs_no_directive_v1",
    ),
    (
        "argument_hints_v2",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_argument_hints_vs_no_directive_v1",
    ),
    (
        "schema_field_hints_v4",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_schema_field_hints_vs_no_directive_v1",
    ),
    (
        "schema_literal_targets_v5",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_schema_literal_targets_vs_no_directive_v1",
    ),
)
DEFAULT_ALIAS_REPEAT_COMPARISONS: tuple[tuple[str, Path], ...] = (
    (
        "contracted",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_repeat_contracted_vs_no_directive_v1",
    ),
    (
        "role_catalog_v1",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_repeat_role_catalog_vs_no_directive_v1",
    ),
    (
        "argument_hints_v2",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_repeat_argument_hints_vs_no_directive_v1",
    ),
    (
        "schema_field_hints_v4",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_vs_no_directive_v1",
    ),
    (
        "schema_literal_targets_v5",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_repeat_schema_literal_targets_vs_no_directive_v1",
    ),
)
DEFAULT_ALIAS_TRANSFER_COMPARISONS: tuple[tuple[str, Path], ...] = (
    (
        "contracted",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_contracted_vs_no_directive_v1",
    ),
    (
        "role_catalog_v1",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_role_catalog_vs_no_directive_v1",
    ),
    (
        "argument_hints_v2",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_argument_hints_vs_no_directive_v1",
    ),
    (
        "schema_field_hints_v4",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_schema_field_hints_vs_no_directive_v1",
    ),
    (
        "schema_literal_targets_v5",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_schema_literal_targets_vs_no_directive_v1",
    ),
)
DEFAULT_ALIAS_TRANSFER_ORACLE_COMPARISONS: tuple[tuple[str, Path], ...] = (
    (
        "contracted",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_contracted_vs_no_directive_v2",
    ),
    (
        "role_catalog_v1",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_role_catalog_vs_no_directive_v2",
    ),
    (
        "argument_hints_v2",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_argument_hints_vs_no_directive_v2",
    ),
    (
        "schema_field_hints_v4",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_schema_field_hints_vs_no_directive_v2",
    ),
    (
        "schema_literal_targets_v5",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_schema_literal_targets_vs_no_directive_v2",
    ),
)
DEFAULT_ALIAS_TRANSFER_REPEAT_COMPARISONS: tuple[tuple[str, Path], ...] = (
    (
        "contracted",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_repeat_contracted_vs_no_directive_v1",
    ),
    (
        "role_catalog_v1",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_repeat_role_catalog_vs_no_directive_v1",
    ),
    (
        "argument_hints_v2",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_repeat_argument_hints_vs_no_directive_v1",
    ),
    (
        "schema_field_hints_v4",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_repeat_schema_field_hints_vs_no_directive_v1",
    ),
    (
        "schema_literal_targets_v5",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_repeat_schema_literal_targets_vs_no_directive_v1",
    ),
)
DEFAULT_ALIAS_TRANSFER_OBLIQUE_COMPARISONS: tuple[tuple[str, Path], ...] = (
    (
        "contracted",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_oblique_contracted_vs_no_directive_v1",
    ),
    (
        "role_catalog_v1",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_oblique_role_catalog_vs_no_directive_v1",
    ),
    (
        "argument_hints_v2",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_oblique_argument_hints_vs_no_directive_v1",
    ),
    (
        "schema_field_hints_v4",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_oblique_schema_field_hints_vs_no_directive_v1",
    ),
    (
        "schema_literal_targets_v5",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_oblique_schema_literal_targets_vs_no_directive_v1",
    ),
    (
        "oblique_code_hints_v6",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260509T_h1n_oracle_oblique_code_hints_vs_no_directive_v1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze visual hard-slice live replay matrices.")
    parser.add_argument(
        "--matrix",
        choices=[
            "stress",
            "alias-repeat",
            "alias-transfer",
            "alias-transfer-oracle",
            "alias-transfer-repeat",
            "alias-transfer-oblique",
        ],
        default="stress",
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparisons = _default_comparisons(args.matrix)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.matrix)
    table_prefix = _table_prefix(args.matrix)
    payload = analyze_visual_live_stress_matrix(
        output_dir=output_dir,
        comparisons=comparisons,
        matrix_name=args.matrix,
        table_prefix=table_prefix,
    )
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


def analyze_visual_live_stress_matrix(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    comparisons: tuple[tuple[str, Path], ...] = DEFAULT_COMPARISONS,
    matrix_name: str = "stress",
    table_prefix: str = "stress_matrix",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    loaded = [(label, _read_comparison(path)) for label, path in comparisons]
    summary_rows = [_summary_row(label, payload) for label, payload in loaded]
    case_rows = [_case_row(label, row) for label, payload in loaded for row in payload["case_deltas"]]
    transition_rows = _transition_rows(case_rows)
    finding_rows = _finding_rows(summary_rows, case_rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "comparison_count": len(loaded),
        "case_count": len({row["case_id"] for row in case_rows}),
        "finding_count": len(finding_rows),
        "matrix_name": matrix_name,
        "purpose": f"Diagnose strict-vs-executor behavior in the visual hard-slice {matrix_name} live replay matrix.",
    }
    payload = {
        "manifest": manifest,
        "summary_rows": summary_rows,
        "case_rows": case_rows,
        "transition_rows": transition_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / f"{table_prefix}_summary.csv", summary_rows)
    _write_csv(tables_dir / f"{table_prefix}_case_transitions.csv", case_rows)
    _write_csv(tables_dir / f"{table_prefix}_transition_counts.csv", transition_rows)
    _write_csv(tables_dir / f"{table_prefix}_findings.csv", finding_rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "diagnostic.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "diagnostic.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _default_output_dir(matrix_name: str) -> Path:
    if matrix_name == "alias-repeat":
        return DEFAULT_ALIAS_REPEAT_OUTPUT_DIR
    if matrix_name == "alias-transfer":
        return DEFAULT_ALIAS_TRANSFER_OUTPUT_DIR
    if matrix_name == "alias-transfer-oracle":
        return DEFAULT_ALIAS_TRANSFER_ORACLE_OUTPUT_DIR
    if matrix_name == "alias-transfer-repeat":
        return DEFAULT_ALIAS_TRANSFER_REPEAT_OUTPUT_DIR
    if matrix_name == "alias-transfer-oblique":
        return DEFAULT_ALIAS_TRANSFER_OBLIQUE_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def _default_comparisons(matrix_name: str) -> tuple[tuple[str, Path], ...]:
    if matrix_name == "alias-repeat":
        return DEFAULT_ALIAS_REPEAT_COMPARISONS
    if matrix_name == "alias-transfer":
        return DEFAULT_ALIAS_TRANSFER_COMPARISONS
    if matrix_name == "alias-transfer-oracle":
        return DEFAULT_ALIAS_TRANSFER_ORACLE_COMPARISONS
    if matrix_name == "alias-transfer-repeat":
        return DEFAULT_ALIAS_TRANSFER_REPEAT_COMPARISONS
    if matrix_name == "alias-transfer-oblique":
        return DEFAULT_ALIAS_TRANSFER_OBLIQUE_COMPARISONS
    return DEFAULT_COMPARISONS


def _table_prefix(matrix_name: str) -> str:
    if matrix_name == "alias-repeat":
        return "alias_repeat_matrix"
    if matrix_name == "alias-transfer":
        return "alias_transfer_matrix"
    if matrix_name == "alias-transfer-oracle":
        return "alias_transfer_oracle_matrix"
    if matrix_name == "alias-transfer-repeat":
        return "alias_transfer_repeat_matrix"
    if matrix_name == "alias-transfer-oblique":
        return "alias_transfer_oblique_matrix"
    return "stress_matrix"


def _read_comparison(path: Path) -> dict[str, Any]:
    return json.loads((path / "live_replay_comparison.json").read_text(encoding="utf-8"))


def _summary_row(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    return {
        "label": label,
        "candidate_system_id": summary["candidate_system_id"],
        "shared_case_count": summary["shared_case_count"],
        "baseline_exact_rate": summary["baseline_exact_rate"],
        "candidate_exact_rate": summary["candidate_exact_rate"],
        "delta_exact_rate": summary["delta_exact_rate"],
        "baseline_executor_equivalence_rate": summary["baseline_executor_equivalence_rate"],
        "candidate_executor_equivalence_rate": summary["candidate_executor_equivalence_rate"],
        "delta_executor_equivalence_rate": summary["delta_executor_equivalence_rate"],
    }


def _case_row(label: str, row: dict[str, Any]) -> dict[str, Any]:
    delta_exact = int(row.get("delta_exact_match") or 0)
    delta_executor = int(row.get("delta_executor_equivalence_match") or 0)
    transition = "unchanged"
    if delta_exact > 0:
        transition = "strict_gain"
    elif delta_executor > 0:
        transition = "executor_gain_without_strict"
    elif delta_exact < 0 or delta_executor < 0:
        transition = "regression"
    return {
        "label": label,
        "case_id": row["case_id"],
        "family": row.get("family", ""),
        "baseline_failure_mode": row.get("baseline_replay_failure_mode", ""),
        "candidate_failure_mode": row.get("candidate_replay_failure_mode", ""),
        "delta_exact_match": delta_exact,
        "delta_executor_equivalence_match": delta_executor,
        "transition": transition,
    }


def _transition_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = {}
    for row in case_rows:
        counters.setdefault(str(row["label"]), Counter())[str(row["transition"])] += 1
    return [
        {"label": label, "transition": transition, "count": count}
        for label, counter in sorted(counters.items())
        for transition, count in sorted(counter.items())
    ]


def _finding_rows(summary_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    exact_best = max(summary_rows, key=lambda row: float(row["candidate_exact_rate"]))
    executor_best = [
        row for row in summary_rows if float(row["candidate_executor_equivalence_rate"]) == 1.0
    ]
    findings.append(
        {
            "finding_id": "strict_upper_bound",
            "finding": f"{exact_best['label']} is the strict upper bound at {exact_best['candidate_exact_rate']}.",
        }
    )
    findings.append(
        {
            "finding_id": "executor_equivalence_set",
            "finding": "Executor-equivalent full-success rows: " + ", ".join(row["label"] for row in executor_best) + ".",
        }
    )
    executor_without_strict = sorted(
        {
            row["label"]
            for row in case_rows
            if row["transition"] == "executor_gain_without_strict"
        }
    )
    findings.append(
        {
            "finding_id": "executor_without_strict",
            "finding": "Rows with executor gain without strict gain: " + ", ".join(executor_without_strict) + ".",
        }
    )
    regressions = sorted(
        f"{row['label']}:{row['case_id']}"
        for row in case_rows
        if row["transition"] == "regression"
    )
    findings.append(
        {
            "finding_id": "regressions",
            "finding": "Regression cases: " + (", ".join(regressions) if regressions else "none") + ".",
        }
    )
    return findings


def _markdown(payload: dict[str, Any]) -> str:
    matrix_name = str(payload["manifest"].get("matrix_name", "stress")).replace("-", " ").title()
    lines = [
        f"# Visual Live {matrix_name} Diagnostic",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Findings",
        "",
    ]
    for row in payload["finding_rows"]:
        lines.append(f"- `{row['finding_id']}`: {row['finding']}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            _markdown_table(payload["summary_rows"]),
            "",
            "## Case Transitions",
            "",
            _markdown_table(payload["case_rows"]),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
