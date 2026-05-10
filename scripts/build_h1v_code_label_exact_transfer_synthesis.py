from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1v_code_label_exact_transfer_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    packet_label: str
    profile_label: str
    live_packet_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h1n_component_value",
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1n_component_value_no_directive_execute_v1",
    ),
    PacketSpec(
        "h1n_component_value",
        "component_label_guard_v11",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1",
    ),
    PacketSpec(
        "h1n_component_value",
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1",
    ),
    PacketSpec(
        "h1n_component_value",
        "code_label_exact_guard_v15",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1o_control_factorial_no_directive_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "component_label_guard_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "code_label_exact_guard_v15",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1v_code_label_exact_guard_on_h1o_control_factorial_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1p_component_value_no_directive_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "component_label_guard_v11",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "code_label_exact_guard_v15",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1",
    ),
)

COMPARISON_DIRS: tuple[Path, ...] = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1v_code_label_exact_guard_h1n_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1v_code_label_exact_guard_h1n_vs_component_residual_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1v_code_label_exact_guard_h1o_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1v_code_label_exact_guard_h1o_vs_component_residual_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1v_code_label_exact_guard_h1p_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1v_code_label_exact_guard_h1p_vs_component_residual_guard_v1",
)


def build_h1v_code_label_exact_transfer_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    aggregate_rows = _aggregate_rows(packet_rows)
    family_rows = _family_rows()
    comparison_rows = [_comparison_row(path) for path in COMPARISON_DIRS]
    v15_failure_rows = _failure_rows("code_label_exact_guard_v15")
    finding_rows = _finding_rows(aggregate_rows, comparison_rows, v15_failure_rows)
    v11 = _row_by_label(aggregate_rows, "component_label_guard_v11")
    v12 = _row_by_label(aggregate_rows, "component_residual_guard_v12")
    v15 = _row_by_label(aggregate_rows, "code_label_exact_guard_v15")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "transfer_case_count": int(v15["case_count"]),
        "v11_transfer_exact_success_count": int(v11["exact_success_count"]),
        "v11_transfer_executor_success_count": int(v11["executor_success_count"]),
        "v12_transfer_exact_success_count": int(v12["exact_success_count"]),
        "v12_transfer_executor_success_count": int(v12["executor_success_count"]),
        "v15_transfer_exact_success_count": int(v15["exact_success_count"]),
        "v15_transfer_executor_success_count": int(v15["executor_success_count"]),
        "v15_failure_count": len(v15_failure_rows),
        "comparison_count": len(comparison_rows),
        "finding_count": len(finding_rows),
        "promotion_decision": "reject_global_promotion_target_code_label_only",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "aggregate_rows": aggregate_rows,
        "family_rows": family_rows,
        "comparison_rows": comparison_rows,
        "v15_failure_rows": v15_failure_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1v_code_label_exact_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1v_code_label_exact_transfer_aggregate.csv", aggregate_rows)
    _write_csv(tables_dir / "h1v_code_label_exact_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h1v_code_label_exact_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h1v_code_label_exact_v15_failures.csv", v15_failure_rows)
    _write_csv(tables_dir / "h1v_code_label_exact_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _packet_row(spec: PacketSpec) -> dict[str, Any]:
    summary = _read_json(spec.live_packet_dir / "summary.json")
    results = _read_json(spec.live_packet_dir / "live_replay_results.json")
    case_count = int(summary["case_count"])
    exact_success_count = sum(1 for row in results if row.get("replay_exact_match") is True)
    executor_success_count = sum(1 for row in results if row.get("replay_executor_equivalence_match") is True)
    return {
        "packet_label": spec.packet_label,
        "profile_label": spec.profile_label,
        "system_id": summary["system_id"],
        "packet_dir": str(spec.live_packet_dir.relative_to(ROOT)),
        "case_count": case_count,
        "exact_success_count": exact_success_count,
        "exact_rate": exact_success_count / case_count if case_count else 0.0,
        "executor_success_count": executor_success_count,
        "executor_rate": executor_success_count / case_count if case_count else 0.0,
        "argument_mismatch_count": sum(1 for row in results if row.get("replay_failure_mode") == "argument_mismatch"),
        "wrong_tool_count": sum(1 for row in results if row.get("replay_failure_mode") == "wrong_tool"),
        "no_tool_call_count": sum(1 for row in results if row.get("replay_failure_mode") == "no_tool_call"),
        "executable_paraphrase_count": sum(
            1 for row in results if row.get("replay_failure_mode") == "executable_paraphrase"
        ),
    }


def _aggregate_rows(packet_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_label in sorted({str(row["profile_label"]) for row in packet_rows}):
        profile_rows = [row for row in packet_rows if row["profile_label"] == profile_label]
        case_count = sum(int(row["case_count"]) for row in profile_rows)
        exact_success_count = sum(int(row["exact_success_count"]) for row in profile_rows)
        executor_success_count = sum(int(row["executor_success_count"]) for row in profile_rows)
        rows.append(
            {
                "profile_label": profile_label,
                "case_count": case_count,
                "exact_success_count": exact_success_count,
                "exact_rate": exact_success_count / case_count if case_count else 0.0,
                "executor_success_count": executor_success_count,
                "executor_rate": executor_success_count / case_count if case_count else 0.0,
                "argument_mismatch_count": sum(int(row["argument_mismatch_count"]) for row in profile_rows),
                "wrong_tool_count": sum(int(row["wrong_tool_count"]) for row in profile_rows),
                "no_tool_call_count": sum(int(row["no_tool_call_count"]) for row in profile_rows),
                "executable_paraphrase_count": sum(
                    int(row["executable_paraphrase_count"]) for row in profile_rows
                ),
            }
        )
    return rows


def _family_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
        results = _read_json(spec.live_packet_dir / "live_replay_results.json")
        for family in sorted({str(row["family"]) for row in results}):
            family_results = [row for row in results if row["family"] == family]
            case_count = len(family_results)
            exact_success_count = sum(1 for row in family_results if row.get("replay_exact_match") is True)
            executor_success_count = sum(
                1 for row in family_results if row.get("replay_executor_equivalence_match") is True
            )
            rows.append(
                {
                    "packet_label": spec.packet_label,
                    "profile_label": spec.profile_label,
                    "family": family,
                    "case_count": case_count,
                    "exact_success_count": exact_success_count,
                    "exact_rate": exact_success_count / case_count if case_count else 0.0,
                    "executor_success_count": executor_success_count,
                    "executor_rate": executor_success_count / case_count if case_count else 0.0,
                }
            )
    return rows


def _comparison_row(path: Path) -> dict[str, Any]:
    payload = _read_json(path / "live_replay_comparison.json")
    summary = payload["summary"]
    return {
        "comparison_dir": str(path.relative_to(ROOT)),
        "baseline_system_id": summary["baseline_system_id"],
        "candidate_system_id": summary["candidate_system_id"],
        "shared_case_count": summary["shared_case_count"],
        "baseline_exact_rate": summary["baseline_exact_rate"],
        "candidate_exact_rate": summary["candidate_exact_rate"],
        "delta_exact_rate": summary["delta_exact_rate"],
        "baseline_executor_equivalence_rate": summary["baseline_executor_equivalence_rate"],
        "candidate_executor_equivalence_rate": summary["candidate_executor_equivalence_rate"],
        "delta_executor_equivalence_rate": summary["delta_executor_equivalence_rate"],
    }


def _failure_rows(profile_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
        if spec.profile_label != profile_label:
            continue
        results = _read_json(spec.live_packet_dir / "live_replay_results.json")
        for row in results:
            if row.get("replay_exact_match") is True:
                continue
            rows.append(
                {
                    "packet_label": spec.packet_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "failure_mode": row.get("replay_failure_mode", ""),
                    "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                    "output_dir": row.get("output_dir", ""),
                }
            )
    return rows


def _finding_rows(
    aggregate_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    v11 = _row_by_label(aggregate_rows, "component_label_guard_v11")
    v12 = _row_by_label(aggregate_rows, "component_residual_guard_v12")
    v15 = _row_by_label(aggregate_rows, "code_label_exact_guard_v15")
    h1p_vs_v12 = next(row for row in comparison_rows if "h1p_vs_component_residual_guard" in row["comparison_dir"])
    h1n_vs_v11 = next(row for row in comparison_rows if "h1n_vs_component_label_guard" in row["comparison_dir"])
    h1o_vs_v11 = next(row for row in comparison_rows if "h1o_vs_component_label_guard" in row["comparison_dir"])
    failed_cases = ", ".join(row["case_id"] for row in failure_rows)
    return [
        {
            "finding_id": "v15_not_global_promotion",
            "finding": (
                "V15 reaches "
                f"{v15['exact_success_count']}/{v15['case_count']} exact and "
                f"{v15['executor_success_count']}/{v15['case_count']} executor-equivalent transfer successes, "
                f"below v11's {v11['executor_success_count']}/{v11['case_count']} executor-equivalent and "
                f"v12's {v12['exact_success_count']}/{v12['case_count']} exact totals."
            ),
        },
        {
            "finding_id": "h1n_negative_transfer_persists",
            "finding": (
                "V15 ties v12 on H1n but remains below v11, with "
                f"{h1n_vs_v11['delta_exact_rate']:.3f} exact-rate and "
                f"{h1n_vs_v11['delta_executor_equivalence_rate']:.3f} executor-rate deltas versus v11."
            ),
        },
        {
            "finding_id": "h1o_code_gain_has_executor_cost",
            "finding": (
                "V15 improves H1o strict exactness versus v11 by "
                f"{h1o_vs_v11['delta_exact_rate']:.3f}, but loses executor-equivalence by "
                f"{h1o_vs_v11['delta_executor_equivalence_rate']:.3f}."
            ),
        },
        {
            "finding_id": "h1p_component_value_regression",
            "finding": (
                "V15 loses the H1p component-value holdout against v12 by "
                f"{h1p_vs_v12['delta_exact_rate']:.3f} exact-rate and "
                f"{h1p_vs_v12['delta_executor_equivalence_rate']:.3f} executor-rate."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "Keep v11 as the transfer-stable default. Treat v15 as a local code-label repair and design the next "
                f"slice around the remaining v15 failures: {failed_cases}."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H1v Code-Label Exact Transfer Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H1v rejects v15 as a global promotion. The code-label exact guard saturated H1r locally, "
            f"but transfers to only `{manifest['v15_transfer_exact_success_count']} / {manifest['transfer_case_count']}` "
            f"strict exact and `{manifest['v15_transfer_executor_success_count']} / {manifest['transfer_case_count']}` "
            "executor-equivalent successes across H1n/H1o/H1p."
        ),
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Aggregate Rows",
        "",
        _table(payload["aggregate_rows"]),
        "",
        "## Family Rows",
        "",
        _table(payload["family_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## V15 Non-Exact Rows",
        "",
        _table(payload["v15_failure_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_None._"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"`{value:.5f}`"
    if isinstance(value, (int, bool)):
        return f"`{str(value).lower() if isinstance(value, bool) else value}`"
    return str(value)


def _row_by_label(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["profile_label"] == profile_label:
            return row
    raise KeyError(profile_label)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H1v code-label exact transfer synthesis packet.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1v_code_label_exact_transfer_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
