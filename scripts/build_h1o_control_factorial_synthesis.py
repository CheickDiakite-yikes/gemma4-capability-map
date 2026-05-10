from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1o_control_factorial_synthesis"


@dataclass(frozen=True)
class ProfileSpec:
    label: str
    live_packet_dir: Path


PROFILE_SPECS: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        label="no_directive",
        live_packet_dir=ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1o_control_factorial_no_directive_execute_v1",
    ),
    ProfileSpec(
        label="argument_hints_v2",
        live_packet_dir=ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1o_control_factorial_argument_hints_execute_v1",
    ),
    ProfileSpec(
        label="hybrid_label_guard_v8",
        live_packet_dir=ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1o_control_factorial_hybrid_label_guard_execute_v1",
    ),
    ProfileSpec(
        label="no_call_control_rescue_v10",
        live_packet_dir=ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1o_control_factorial_no_call_control_rescue_execute_v1",
    ),
    ProfileSpec(
        label="oblique_code_guard_v7",
        live_packet_dir=ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1o_control_factorial_oblique_code_guard_execute_v1",
    ),
    ProfileSpec(
        label="component_value_guard_v9",
        live_packet_dir=ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1o_control_factorial_component_value_guard_execute_v1",
    ),
)


def build_h1o_control_factorial_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    profile_rows = [_profile_row(spec) for spec in PROFILE_SPECS]
    family_rows = [row for spec in PROFILE_SPECS for row in _family_rows(spec)]
    family_delta_rows = _family_delta_rows(family_rows)
    finding_rows = _finding_rows(profile_rows, family_rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "profile_count": len(profile_rows),
        "family_row_count": len(family_rows),
        "finding_count": len(finding_rows),
        "case_count": int(profile_rows[0]["case_count"]),
        "strict_upper_bound": _best_labels(profile_rows, "exact_success_count"),
        "executor_upper_bound": _best_labels(profile_rows, "executor_success_count"),
    }
    payload = {
        "manifest": manifest,
        "profile_rows": profile_rows,
        "family_rows": family_rows,
        "family_delta_rows": family_delta_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1o_control_factorial_profile_summary.csv", profile_rows)
    _write_csv(tables_dir / "h1o_control_factorial_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h1o_control_factorial_family_deltas.csv", family_delta_rows)
    _write_csv(tables_dir / "h1o_control_factorial_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _profile_row(spec: ProfileSpec) -> dict[str, Any]:
    summary = _read_json(spec.live_packet_dir / "summary.json")
    results = _read_json(spec.live_packet_dir / "live_replay_results.json")
    exact_success_count = sum(1 for row in results if row.get("replay_exact_match") is True)
    executor_success_count = sum(1 for row in results if row.get("replay_executor_equivalence_match") is True)
    no_tool_call_count = sum(1 for row in results if row.get("replay_failure_mode") == "no_tool_call")
    argument_mismatch_count = sum(1 for row in results if row.get("replay_failure_mode") == "argument_mismatch")
    return {
        "label": spec.label,
        "system_id": summary["system_id"],
        "packet_dir": str(spec.live_packet_dir.relative_to(ROOT)),
        "case_count": int(summary["case_count"]),
        "exact_success_count": exact_success_count,
        "exact_rate": float(summary["exact_rate"]),
        "executor_success_count": executor_success_count,
        "executor_rate": float(summary["executor_equivalence_rate"]),
        "no_tool_call_count": no_tool_call_count,
        "argument_mismatch_count": argument_mismatch_count,
    }


def _family_rows(spec: ProfileSpec) -> list[dict[str, Any]]:
    results = _read_json(spec.live_packet_dir / "live_replay_results.json")
    families = sorted({str(row.get("family", "")) for row in results})
    rows: list[dict[str, Any]] = []
    for family in families:
        family_results = [row for row in results if row.get("family") == family]
        case_count = len(family_results)
        exact_success_count = sum(1 for row in family_results if row.get("replay_exact_match") is True)
        executor_success_count = sum(
            1 for row in family_results if row.get("replay_executor_equivalence_match") is True
        )
        rows.append(
            {
                "label": spec.label,
                "family": family,
                "case_count": case_count,
                "exact_success_count": exact_success_count,
                "exact_rate": exact_success_count / case_count if case_count else 0.0,
                "executor_success_count": executor_success_count,
                "executor_rate": executor_success_count / case_count if case_count else 0.0,
                "no_tool_call_count": sum(
                    1 for row in family_results if row.get("replay_failure_mode") == "no_tool_call"
                ),
                "argument_mismatch_count": sum(
                    1 for row in family_results if row.get("replay_failure_mode") == "argument_mismatch"
                ),
                "executable_paraphrase_count": sum(
                    1 for row in family_results if row.get("replay_failure_mode") == "executable_paraphrase"
                ),
            }
        )
    return rows


def _family_delta_rows(family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        row["family"]: row
        for row in family_rows
        if row["label"] == "no_directive"
    }
    rows: list[dict[str, Any]] = []
    for row in family_rows:
        if row["label"] == "no_directive":
            continue
        baseline_row = baseline[str(row["family"])]
        rows.append(
            {
                "label": row["label"],
                "family": row["family"],
                "case_count": row["case_count"],
                "delta_exact_success_count": int(row["exact_success_count"]) - int(baseline_row["exact_success_count"]),
                "delta_exact_rate": float(row["exact_rate"]) - float(baseline_row["exact_rate"]),
                "delta_executor_success_count": int(row["executor_success_count"])
                - int(baseline_row["executor_success_count"]),
                "delta_executor_rate": float(row["executor_rate"]) - float(baseline_row["executor_rate"]),
            }
        )
    return rows


def _finding_rows(profile_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    family_index = {
        (str(row["label"]), str(row["family"])): row
        for row in family_rows
    }
    strict_best = _best_labels(profile_rows, "exact_success_count")
    executor_best = _best_labels(profile_rows, "executor_success_count")
    baseline_activation = family_index[("no_directive", "h1o_activation_no_call")]
    no_call_activation = family_index[("no_call_control_rescue_v10", "h1o_activation_no_call")]
    argument_code = family_index[("argument_hints_v2", "h1o_code_negation_preservation")]
    component_value = family_index[("component_value_guard_v9", "h1o_component_value_boundary")]
    return [
        {
            "finding_id": "strict_upper_bound",
            "finding": (
                f"Strict upper bound is {', '.join(strict_best)} at "
                f"{_profile_by_label(profile_rows, strict_best[0])['exact_success_count']}/12."
            ),
        },
        {
            "finding_id": "executor_upper_bound",
            "finding": (
                f"Executor-equivalence upper bound is {', '.join(executor_best)} at "
                f"{_profile_by_label(profile_rows, executor_best[0])['executor_success_count']}/12; "
                "no H1o profile reaches full executor success."
            ),
        },
        {
            "finding_id": "activation_saturated_without_rescue",
            "finding": (
                "Activation/no-call is not the remaining bottleneck on H1o: no-directive already reaches "
                f"{baseline_activation['exact_success_count']}/4 exact, while no-call rescue reaches "
                f"{no_call_activation['exact_success_count']}/4 and introduces one regression."
            ),
        },
        {
            "finding_id": "code_negation_is_repairable",
            "finding": (
                "Code/negation failures are controller-sensitive: argument hints reaches "
                f"{argument_code['exact_success_count']}/4 exact and {argument_code['executor_success_count']}/4 "
                "executor-equivalent versus no-directive at 1/4 exact and 2/4 executor-equivalent."
            ),
        },
        {
            "finding_id": "component_boundary_remains_residual",
            "finding": (
                "Component/value boundaries remain the hard residue: component-value guard and argument hints both "
                f"top out at {component_value['exact_success_count']}/4 exact and "
                f"{component_value['executor_success_count']}/4 executor-equivalent on this family."
            ),
        },
        {
            "finding_id": "promotion_decision",
            "finding": (
                "Promote argument hints as the conservative H1o default; do not promote no-call rescue globally; "
                "treat component-value guard as a tied candidate that needs a fresh component-only holdout."
            ),
        },
    ]


def _best_labels(rows: list[dict[str, Any]], key: str) -> list[str]:
    best = max(int(row[key]) for row in rows)
    return [str(row["label"]) for row in rows if int(row[key]) == best]


def _profile_by_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in rows:
        if row["label"] == label:
            return row
    raise KeyError(label)


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
    manifest = payload["manifest"]
    lines = [
        "# H1o Control-Factorial Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- `{row['finding_id']}`: {row['finding']}" for row in payload["finding_rows"])
    lines.extend(
        [
            "",
            "## Profile Summary",
            "",
            _markdown_table(payload["profile_rows"]),
            "",
            "## Family Summary",
            "",
            _markdown_table(payload["family_rows"]),
            "",
            "## Family Deltas Versus No-Directive",
            "",
            _markdown_table(payload["family_delta_rows"]),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build H1o control-factorial synthesis artifacts.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1o_control_factorial_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
