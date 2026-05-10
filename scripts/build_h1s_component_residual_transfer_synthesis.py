from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1s_component_residual_transfer_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    packet_label: str
    profile_label: str
    live_packet_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h1r_component_residual",
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1r_component_label_residual_no_directive_execute_v1",
    ),
    PacketSpec(
        "h1r_component_residual",
        "component_label_guard_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1r_component_label_residual_component_label_guard_execute_v1",
    ),
    PacketSpec(
        "h1r_component_residual",
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1r_component_label_residual_component_residual_guard_execute_v1",
    ),
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
)

COMPARISON_DIRS: tuple[Path, ...] = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1s_component_residual_guard_h1n_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1s_component_residual_guard_h1n_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1s_component_residual_guard_h1o_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1s_component_residual_guard_h1o_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1s_component_residual_guard_h1p_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1s_component_residual_guard_h1p_vs_no_directive_v1",
)

TRANSFER_PACKET_LABELS = {"h1n_component_value", "h1o_control_factorial", "h1p_component_value"}


def build_h1s_component_residual_transfer_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    transfer_aggregate_rows = _aggregate_rows(
        [row for row in packet_rows if row["packet_label"] in TRANSFER_PACKET_LABELS]
    )
    comparison_rows = [_comparison_row(path) for path in COMPARISON_DIRS]
    v12_failure_rows = _failure_rows("component_residual_guard_v12")
    finding_rows = _finding_rows(packet_rows, transfer_aggregate_rows, comparison_rows, v12_failure_rows)
    v12_transfer = _row_by_label(transfer_aggregate_rows, "component_residual_guard_v12")
    v11_transfer = _row_by_label(transfer_aggregate_rows, "component_label_guard_v11")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "transfer_case_count": int(v12_transfer["case_count"]),
        "v11_transfer_exact_success_count": int(v11_transfer["exact_success_count"]),
        "v11_transfer_executor_success_count": int(v11_transfer["executor_success_count"]),
        "v12_transfer_exact_success_count": int(v12_transfer["exact_success_count"]),
        "v12_transfer_executor_success_count": int(v12_transfer["executor_success_count"]),
        "v12_failure_count": len(v12_failure_rows),
        "comparison_count": len(comparison_rows),
        "finding_count": len(finding_rows),
        "promotion_decision": "targeted_patch_not_global_default",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "transfer_aggregate_rows": transfer_aggregate_rows,
        "comparison_rows": comparison_rows,
        "v12_failure_rows": v12_failure_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1s_component_residual_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1s_component_residual_transfer_aggregate.csv", transfer_aggregate_rows)
    _write_csv(tables_dir / "h1s_component_residual_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h1s_component_residual_v12_failures.csv", v12_failure_rows)
    _write_csv(tables_dir / "h1s_component_residual_findings.csv", finding_rows)
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
                "no_tool_call_count": sum(int(row["no_tool_call_count"]) for row in rows_for_profile),
                "executable_paraphrase_count": sum(
                    int(row["executable_paraphrase_count"]) for row in rows_for_profile
                ),
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
    specs = [spec for spec in PACKET_SPECS if spec.profile_label == profile_label]
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
    packet_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    v12_failure_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h1r_v12 = _packet_by_label(packet_rows, "h1r_component_residual", "component_residual_guard_v12")
    v11_transfer = _row_by_label(aggregate_rows, "component_label_guard_v11")
    v12_transfer = _row_by_label(aggregate_rows, "component_residual_guard_v12")
    h1n_v11_comparison = _comparison_by_dir(comparison_rows, "h1n_vs_component_label_guard")
    h1o_v11_comparison = _comparison_by_dir(comparison_rows, "h1o_vs_component_label_guard")
    h1p_v11_comparison = _comparison_by_dir(comparison_rows, "h1p_vs_component_label_guard")
    non_executor_failures = [
        row for row in v12_failure_rows if str(row["executor_equivalence_match"]).lower() != "true"
    ]
    return [
        {
            "finding_id": "v12_solves_local_h1r_residual",
            "finding": (
                "Component-residual guard v12 saturates the residual H1r slice at "
                f"{h1r_v12['exact_success_count']}/6 exact and "
                f"{h1r_v12['executor_success_count']}/6 executor-equivalent."
            ),
        },
        {
            "finding_id": "v12_transfers_strict_but_not_executor",
            "finding": (
                "Across H1n/H1o/H1p, v12 improves strict exactness from "
                f"{v11_transfer['exact_success_count']}/32 to {v12_transfer['exact_success_count']}/32, "
                "but lowers executor-equivalence from "
                f"{v11_transfer['executor_success_count']}/32 to {v12_transfer['executor_success_count']}/32."
            ),
        },
        {
            "finding_id": "negative_h1n_transfer",
            "finding": (
                "H1n is the clearest negative-transfer warning: v12 delta versus v11 is "
                f"{h1n_v11_comparison['delta_exact_rate']:.3f} exact-rate and "
                f"{h1n_v11_comparison['delta_executor_equivalence_rate']:.3f} executor-rate."
            ),
        },
        {
            "finding_id": "h1o_strict_executor_split",
            "finding": (
                "On H1o, v12 improves strict exactness but loses executor-equivalence versus v11: "
                f"delta exact-rate {h1o_v11_comparison['delta_exact_rate']:.3f}; "
                f"delta executor-rate {h1o_v11_comparison['delta_executor_equivalence_rate']:.3f}."
            ),
        },
        {
            "finding_id": "h1p_transfer_is_real_but_partial",
            "finding": (
                "On H1p, v12 improves both exact and executor-equivalence rates versus v11 by "
                f"{h1p_v11_comparison['delta_exact_rate']:.3f}, but still leaves one wrong-tool stale-selection miss."
            ),
        },
        {
            "finding_id": "remaining_v12_failures",
            "finding": (
                "Remaining non-executor v12 failures are "
                + (
                    ", ".join(f"{row['packet_label']}:{row['case_id']}:{row['failure_mode']}" for row in non_executor_failures)
                    if non_executor_failures
                    else "none"
                )
                + "."
            ),
        },
        {
            "finding_id": "promotion_decision",
            "finding": (
                "Do not promote v12 as the global visual-role catalog default yet. Treat it as a targeted residual "
                "patch or conditional route while v11 remains the more executor-robust general transfer profile."
            ),
        },
    ]


def _row_by_label(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    return next(row for row in rows if row["profile_label"] == profile_label)


def _packet_by_label(packet_rows: list[dict[str, Any]], packet_label: str, profile_label: str) -> dict[str, Any]:
    return next(
        row
        for row in packet_rows
        if row["packet_label"] == packet_label and row["profile_label"] == profile_label
    )


def _comparison_by_dir(rows: list[dict[str, Any]], needle: str) -> dict[str, Any]:
    return next(row for row in rows if needle in row["comparison_dir"])


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H1s Component-Residual Transfer Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H1s tests whether the H1r v12 residual prompt should become the global visual-role catalog default. "
            "The answer is no for now: v12 is a useful targeted patch, but it trades executor robustness for "
            "strict exactness on the broader H1n/H1o/H1p transfer surface."
        ),
        "",
        "## Transfer Aggregate",
        "",
        _markdown_table(payload["transfer_aggregate_rows"]),
        "",
        "## Packet Rows",
        "",
        _markdown_table(payload["packet_rows"]),
        "",
        "## Pairwise Comparisons",
        "",
        _markdown_table(payload["comparison_rows"]),
        "",
        "## v12 Non-Exact Cases",
        "",
        _markdown_table(payload["v12_failure_rows"]),
        "",
        "## Findings",
        "",
        _markdown_table(payload["finding_rows"]),
        "",
        "## Interpretation",
        "",
        (
            "This is the strongest evidence so far that prompt-contract improvements need transfer gates, not just "
            "local residual wins. v12 should feed the next conditional-routing or prompt-factorial slice, while v11 "
            "remains the safer general-purpose component-label guard until the H1n executor regressions are removed."
        ),
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
    parser = argparse.ArgumentParser(description="Build the H1s component-residual transfer synthesis.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1s_component_residual_transfer_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
