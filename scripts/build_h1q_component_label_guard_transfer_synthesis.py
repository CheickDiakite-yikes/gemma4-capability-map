from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1q_component_label_guard_transfer_synthesis"


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
        "argument_hints_v2",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1n_component_value_argument_hints_execute_v1",
    ),
    PacketSpec(
        "h1n_component_value",
        "hybrid_label_guard_v8",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1n_component_value_hybrid_label_guard_execute_v1",
    ),
    PacketSpec(
        "h1n_component_value",
        "component_value_guard_v9",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1n_component_value_component_value_guard_execute_v1",
    ),
    PacketSpec(
        "h1n_component_value",
        "no_call_control_rescue_v10",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1n_component_value_no_call_control_rescue_execute_v1",
    ),
    PacketSpec(
        "h1n_component_value",
        "component_label_guard_v11",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1o_control_factorial_no_directive_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "argument_hints_v2",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1o_control_factorial_argument_hints_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "hybrid_label_guard_v8",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1o_control_factorial_hybrid_label_guard_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "component_value_guard_v9",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1o_control_factorial_component_value_guard_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "no_call_control_rescue_v10",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1o_control_factorial_no_call_control_rescue_execute_v1",
    ),
    PacketSpec(
        "h1o_control_factorial",
        "component_label_guard_v11",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1p_component_value_no_directive_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "argument_hints_v2",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1p_component_value_argument_hints_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "hybrid_label_guard_v8",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1p_component_value_hybrid_label_guard_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "component_value_guard_v9",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1p_component_value_component_value_guard_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "no_call_control_rescue_v10",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1p_component_value_no_call_control_rescue_execute_v1",
    ),
    PacketSpec(
        "h1p_component_value",
        "component_label_guard_v11",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1",
    ),
)


def build_h1q_component_label_guard_transfer_synthesis(
    *, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    aggregate_rows = _aggregate_rows(packet_rows)
    failure_rows = _v11_failure_rows()
    finding_rows = _finding_rows(packet_rows, aggregate_rows, failure_rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "profile_count": len(aggregate_rows),
        "total_case_count": sum(
            int(row["case_count"]) for row in packet_rows if row["profile_label"] == "no_directive"
        ),
        "v11_exact_success_count": _aggregate_by_label(aggregate_rows, "component_label_guard_v11")[
            "exact_success_count"
        ],
        "v11_executor_success_count": _aggregate_by_label(aggregate_rows, "component_label_guard_v11")[
            "executor_success_count"
        ],
        "finding_count": len(finding_rows),
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "aggregate_rows": aggregate_rows,
        "v11_failure_rows": failure_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1q_component_label_guard_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1q_component_label_guard_aggregate_summary.csv", aggregate_rows)
    _write_csv(tables_dir / "h1q_component_label_guard_v11_failures.csv", failure_rows)
    _write_csv(tables_dir / "h1q_component_label_guard_findings.csv", finding_rows)
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
        "executable_paraphrase_count": sum(
            1 for row in results if row.get("replay_failure_mode") == "executable_paraphrase"
        ),
    }


def _aggregate_rows(packet_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile_labels = sorted({str(row["profile_label"]) for row in packet_rows})
    rows: list[dict[str, Any]] = []
    for profile_label in profile_labels:
        rows_for_profile = [row for row in packet_rows if row["profile_label"] == profile_label]
        case_count = sum(int(row["case_count"]) for row in rows_for_profile)
        exact_success_count = sum(int(row["exact_success_count"]) for row in rows_for_profile)
        executor_success_count = sum(int(row["executor_success_count"]) for row in rows_for_profile)
        rows.append(
            {
                "profile_label": profile_label,
                "case_count": case_count,
                "exact_success_count": exact_success_count,
                "exact_rate": exact_success_count / case_count if case_count else 0.0,
                "executor_success_count": executor_success_count,
                "executor_rate": executor_success_count / case_count if case_count else 0.0,
                "argument_mismatch_count": sum(int(row["argument_mismatch_count"]) for row in rows_for_profile),
                "wrong_tool_count": sum(int(row["wrong_tool_count"]) for row in rows_for_profile),
                "executable_paraphrase_count": sum(
                    int(row["executable_paraphrase_count"]) for row in rows_for_profile
                ),
            }
        )
    return rows


def _v11_failure_rows() -> list[dict[str, Any]]:
    specs = [spec for spec in PACKET_SPECS if spec.profile_label == "component_label_guard_v11"]
    rows: list[dict[str, Any]] = []
    for spec in specs:
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
    packet_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]
) -> list[dict[str, str]]:
    strict_best = _best_labels(aggregate_rows, "exact_success_count")
    executor_best = _best_labels(aggregate_rows, "executor_success_count")
    v11 = _aggregate_by_label(aggregate_rows, "component_label_guard_v11")
    v9 = _aggregate_by_label(aggregate_rows, "component_value_guard_v9")
    h1n_v11 = _packet_by_label(packet_rows, "h1n_component_value", "component_label_guard_v11")
    h1n_v9 = _packet_by_label(packet_rows, "h1n_component_value", "component_value_guard_v9")
    h1o_v11 = _packet_by_label(packet_rows, "h1o_control_factorial", "component_label_guard_v11")
    h1p_v11 = _packet_by_label(packet_rows, "h1p_component_value", "component_label_guard_v11")
    h1p_v9 = _packet_by_label(packet_rows, "h1p_component_value", "component_value_guard_v9")
    non_executor_failures = [
        row for row in failure_rows if str(row["executor_equivalence_match"]).lower() != "true"
    ]
    return [
        {
            "finding_id": "aggregate_strict_upper_bound",
            "finding": (
                f"Aggregate strict upper bound across H1n/H1o/H1p is {', '.join(strict_best)} at "
                f"{_aggregate_by_label(aggregate_rows, strict_best[0])['exact_success_count']}/32."
            ),
        },
        {
            "finding_id": "aggregate_executor_upper_bound",
            "finding": (
                f"Aggregate executor-equivalence upper bound is {', '.join(executor_best)} at "
                f"{_aggregate_by_label(aggregate_rows, executor_best[0])['executor_success_count']}/32."
            ),
        },
        {
            "finding_id": "v11_repairs_v9_h1n_regressions",
            "finding": (
                "v11 repairs the broad v9 regression on H1n component-value: "
                f"{h1n_v11['exact_success_count']}/8 exact and {h1n_v11['executor_success_count']}/8 "
                f"executor-equivalent versus v9 at {h1n_v9['exact_success_count']}/8 exact and "
                f"{h1n_v9['executor_success_count']}/8 executor-equivalent."
            ),
        },
        {
            "finding_id": "v11_sets_h1o_executor_ceiling",
            "finding": (
                "v11 sets the current H1o transfer ceiling: "
                f"{h1o_v11['exact_success_count']}/12 exact and {h1o_v11['executor_success_count']}/12 "
                "executor-equivalent."
            ),
        },
        {
            "finding_id": "h1p_tradeoff_vs_v9",
            "finding": (
                "On H1p, v11 ties v9 strict exactness but loses one executor-equivalent case: "
                f"v11 is {h1p_v11['exact_success_count']}/12 exact and {h1p_v11['executor_success_count']}/12 "
                f"executor-equivalent, while v9 is {h1p_v9['exact_success_count']}/12 exact and "
                f"{h1p_v9['executor_success_count']}/12 executor-equivalent."
            ),
        },
        {
            "finding_id": "remaining_v11_failures",
            "finding": (
                "Remaining non-executor v11 failures are "
                + (
                    ", ".join(f"{row['packet_label']}:{row['case_id']}" for row in non_executor_failures)
                    if non_executor_failures
                    else "none"
                )
                + "."
            ),
        },
        {
            "finding_id": "promotion_decision",
            "finding": (
                f"Treat v11 as the current best transfer candidate ({v11['exact_success_count']}/32 exact and "
                f"{v11['executor_success_count']}/32 executor-equivalent versus v9 at "
                f"{v9['exact_success_count']}/32 and {v9['executor_success_count']}/32), but do not make it the "
                "global default until the remaining owner-field, state-tag, and mode-toggle failures are isolated."
            ),
        },
    ]


def _best_labels(rows: list[dict[str, Any]], key: str) -> list[str]:
    best = max(int(row[key]) for row in rows)
    return [str(row["profile_label"]) for row in rows if int(row[key]) == best]


def _aggregate_by_label(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    return next(row for row in rows if row["profile_label"] == profile_label)


def _packet_by_label(packet_rows: list[dict[str, Any]], packet_label: str, profile_label: str) -> dict[str, Any]:
    return next(
        row
        for row in packet_rows
        if row["packet_label"] == packet_label and row["profile_label"] == profile_label
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# H1q Component-Label Guard Transfer Synthesis",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Aggregate Profile Summary",
        "",
        _markdown_table(payload["aggregate_rows"]),
        "",
        "## Packet Summary",
        "",
        _markdown_table(payload["packet_rows"]),
        "",
        "## v11 Non-Exact Cases",
        "",
        _markdown_table(payload["v11_failure_rows"]),
        "",
        "## Findings",
        "",
        _markdown_table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"`{value:.5f}`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, int):
        return f"`{value}`"
    return str(value).replace("|", "\\|")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H1q component-label guard transfer synthesis.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1q_component_label_guard_transfer_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
