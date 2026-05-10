from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1u_split_factor_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    profile_label: str
    live_packet_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1r_component_label_residual_no_directive_execute_v1",
    ),
    PacketSpec(
        "component_label_guard_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1r_component_label_residual_component_label_guard_execute_v1",
    ),
    PacketSpec(
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1r_component_label_residual_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "conditional_residual_route_v13",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1",
    ),
    PacketSpec(
        "nonstandard_component_class_guard_v14",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1u_nonstandard_component_class_guard_on_h1r_component_residual_execute_v1",
    ),
    PacketSpec(
        "code_label_exact_guard_v15",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1u_code_label_exact_guard_on_h1r_component_residual_execute_v1",
    ),
)

COMPARISON_DIRS: tuple[Path, ...] = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1u_nonstandard_component_guard_h1r_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1u_nonstandard_component_guard_h1r_vs_component_residual_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1u_nonstandard_component_guard_h1r_vs_conditional_residual_route_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1u_code_label_exact_guard_h1r_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1u_code_label_exact_guard_h1r_vs_component_residual_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1u_code_label_exact_guard_h1r_vs_conditional_residual_route_v1",
)


def build_h1u_split_factor_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    family_rows = _family_rows()
    comparison_rows = [_comparison_row(path) for path in COMPARISON_DIRS]
    failure_rows = _failure_rows()
    finding_rows = _finding_rows(packet_rows, comparison_rows, failure_rows)
    v15 = _row_by_label(packet_rows, "code_label_exact_guard_v15")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "profile_count": len(packet_rows),
        "case_count": int(v15["case_count"]),
        "v15_exact_success_count": int(v15["exact_success_count"]),
        "v15_executor_success_count": int(v15["executor_success_count"]),
        "comparison_count": len(comparison_rows),
        "finding_count": len(finding_rows),
        "promotion_decision": "transfer_test_v15",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "family_rows": family_rows,
        "comparison_rows": comparison_rows,
        "non_exact_rows": failure_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1u_split_factor_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1u_split_factor_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h1u_split_factor_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h1u_split_factor_non_exact_rows.csv", failure_rows)
    _write_csv(tables_dir / "h1u_split_factor_findings.csv", finding_rows)
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
    }


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


def _failure_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
        results = _read_json(spec.live_packet_dir / "live_replay_results.json")
        for row in results:
            if row.get("replay_exact_match") is True:
                continue
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "failure_mode": row.get("replay_failure_mode", ""),
                    "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                }
            )
    return rows


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    v14 = _row_by_label(packet_rows, "nonstandard_component_class_guard_v14")
    v15 = _row_by_label(packet_rows, "code_label_exact_guard_v15")
    v15_vs_v11 = _comparison_by_dir(comparison_rows, "code_label_exact_guard_h1r_vs_component_label_guard")
    v15_vs_v12 = _comparison_by_dir(comparison_rows, "code_label_exact_guard_h1r_vs_component_residual_guard")
    v14_failure = next(
        row for row in failure_rows if row["profile_label"] == "nonstandard_component_class_guard_v14"
    )
    return [
        {
            "finding_id": "v14_is_factor_positive_but_incomplete",
            "finding": (
                f"Nonstandard component-class guard v14 reaches {v14['exact_success_count']}/6 and fixes "
                f"tag/toggle value-collapse, but still misses {v14_failure['case_id']}."
            ),
        },
        {
            "finding_id": "v15_saturates_h1r",
            "finding": (
                f"Code-label exact guard v15 reaches {v15['exact_success_count']}/6 exact and "
                f"{v15['executor_success_count']}/6 executor-equivalent, matching v12 on H1r."
            ),
        },
        {
            "finding_id": "v15_beats_v11_ties_v12",
            "finding": (
                f"v15 improves over v11 by {v15_vs_v11['delta_exact_rate']:.3f} exact-rate and ties v12 with "
                f"delta {v15_vs_v12['delta_exact_rate']:.3f}."
            ),
        },
        {
            "finding_id": "transfer_decision",
            "finding": (
                "Transfer-test v15 across H1n/H1o/H1p next. It may keep the H1r repair while using less broad "
                "component-residual prose than v12."
            ),
        },
    ]


def _row_by_label(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    return next(row for row in rows if row["profile_label"] == profile_label)


def _comparison_by_dir(rows: list[dict[str, Any]], needle: str) -> dict[str, Any]:
    return next(row for row in rows if needle in row["comparison_dir"])


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H1u Split-Factor Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H1u splits the failed H1t compact conditional route into independent factors. V14 confirms that "
            "nonstandard component-class wording repairs tag/toggle value collapse, but v15 is the stronger local "
            "candidate: it saturates H1r at `6 / 6` and should be transfer-tested next."
        ),
        "",
        "## Packet Rows",
        "",
        _markdown_table(payload["packet_rows"]),
        "",
        "## Family Rows",
        "",
        _markdown_table(payload["family_rows"]),
        "",
        "## Comparison Rows",
        "",
        _markdown_table(payload["comparison_rows"]),
        "",
        "## Non-Exact Rows",
        "",
        _markdown_table(payload["non_exact_rows"]),
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
    parser = argparse.ArgumentParser(description="Build the H1u split-factor synthesis.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1u_split_factor_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
