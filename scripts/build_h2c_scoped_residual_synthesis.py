from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2c_scoped_residual_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    profile_label: str
    packet_dir: Path


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_label: str
    comparison_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2b_residual_exactness_no_directive_execute_v1",
    ),
    PacketSpec(
        "component_value_guard_v9",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2b_residual_exactness_component_value_guard_execute_v1",
    ),
    PacketSpec(
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2b_residual_exactness_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "h2a_stale_selection_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2b_residual_exactness_h2a_execute_v1",
    ),
    PacketSpec(
        "h2c_scoped_residual_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1",
    ),
)

COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2c_vs_no_directive",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2c_scoped_residual_gate_vs_no_directive_on_h2b_v1",
    ),
    ComparisonSpec(
        "h2c_vs_h2a",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2c_scoped_residual_gate_vs_h2a_on_h2b_v1",
    ),
    ComparisonSpec(
        "h2c_vs_v9",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2c_scoped_residual_gate_vs_component_value_guard_on_h2b_v1",
    ),
    ComparisonSpec(
        "h2c_vs_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2c_scoped_residual_gate_vs_component_residual_guard_on_h2b_v1",
    ),
)


def build_h2c_scoped_residual_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    case_rows = _case_rows()
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    finding_rows = _finding_rows(packet_rows, comparison_rows)
    h2c = _row_by_label(packet_rows, "h2c_scoped_residual_gate")
    v12 = _row_by_label(packet_rows, "component_residual_guard_v12")
    h2a = _row_by_label(packet_rows, "h2a_stale_selection_gate")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "profile_count": len(packet_rows),
        "case_count": int(h2c["case_count"]),
        "h2c_exact_success_count": int(h2c["exact_success_count"]),
        "h2c_executor_success_count": int(h2c["executor_success_count"]),
        "v12_exact_success_count": int(v12["exact_success_count"]),
        "v12_executor_success_count": int(v12["executor_success_count"]),
        "h2a_exact_success_count": int(h2a["exact_success_count"]),
        "h2a_executor_success_count": int(h2a["executor_success_count"]),
        "strict_winner": "h2c_scoped_residual_gate",
        "executor_winner": "h2c_scoped_residual_gate",
        "promotion_decision": "transfer_gate_required_before_global_or_default_promotion",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "case_rows": case_rows,
        "comparison_rows": comparison_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h2c_scoped_residual_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2c_scoped_residual_case_matrix.csv", case_rows)
    _write_csv(tables_dir / "h2c_scoped_residual_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2c_scoped_residual_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _packet_row(spec: PacketSpec) -> dict[str, Any]:
    summary = _read_json(spec.packet_dir / "summary.json")
    results = _read_json(spec.packet_dir / "live_replay_results.json")
    case_count = int(summary["case_count"])
    exact_success_count = sum(1 for row in results if row.get("replay_exact_match") is True)
    executor_success_count = sum(1 for row in results if row.get("replay_executor_equivalence_match") is True)
    return {
        "profile_label": spec.profile_label,
        "system_id": summary["system_id"],
        "packet_dir": str(spec.packet_dir.relative_to(ROOT)),
        "case_count": case_count,
        "exact_success_count": exact_success_count,
        "exact_rate": exact_success_count / case_count if case_count else 0.0,
        "executor_success_count": executor_success_count,
        "executor_rate": executor_success_count / case_count if case_count else 0.0,
    }


def _case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
        for row in _read_json(spec.packet_dir / "live_replay_results.json"):
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "source_failure_mode": row.get("source_failure_mode", ""),
                    "replay_failure_mode": row.get("replay_failure_mode", ""),
                    "exact_match": row.get("replay_exact_match"),
                    "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                }
            )
    return rows


def _comparison_row(spec: ComparisonSpec) -> dict[str, Any]:
    payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
    summary = payload["summary"]
    return {
        "comparison_label": spec.comparison_label,
        "comparison_dir": str(spec.comparison_dir.relative_to(ROOT)),
        "baseline_packet_run_id": summary["baseline_packet_run_id"],
        "candidate_packet_run_id": summary["candidate_packet_run_id"],
        "shared_case_count": summary["shared_case_count"],
        "baseline_exact_rate": summary["baseline_exact_rate"],
        "candidate_exact_rate": summary["candidate_exact_rate"],
        "delta_exact_rate": summary["delta_exact_rate"],
        "baseline_executor_equivalence_rate": summary["baseline_executor_equivalence_rate"],
        "candidate_executor_equivalence_rate": summary["candidate_executor_equivalence_rate"],
        "delta_executor_equivalence_rate": summary["delta_executor_equivalence_rate"],
    }


def _finding_rows(packet_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    h2c = _row_by_label(packet_rows, "h2c_scoped_residual_gate")
    h2a = _row_by_label(packet_rows, "h2a_stale_selection_gate")
    v12 = _row_by_label(packet_rows, "component_residual_guard_v12")
    v9 = _row_by_label(packet_rows, "component_value_guard_v9")
    h2c_vs_v12 = _comparison_by_label(comparison_rows, "h2c_vs_v12")
    h2c_vs_h2a = _comparison_by_label(comparison_rows, "h2c_vs_h2a")
    return [
        {
            "finding_id": "h2c_saturates_h2b_residuals",
            "finding": (
                f"H2c reaches {h2c['exact_success_count']}/5 strict and "
                f"{h2c['executor_success_count']}/5 executor-equivalent on the H2b residual packet."
            ),
        },
        {
            "finding_id": "h2c_beats_v12_strict_and_executor",
            "finding": (
                f"H2c improves over v12 by {h2c_vs_v12['delta_exact_rate']:.1f} exact-rate and "
                f"{h2c_vs_v12['delta_executor_equivalence_rate']:.1f} executor-rate, fixing v12's remaining "
                "`result pill` miss while preserving its code-label and nonstandard-component wins."
            ),
        },
        {
            "finding_id": "h2c_separates_residual_exactness_from_h2a",
            "finding": (
                f"H2a is still {h2a['exact_success_count']}/5 strict and {h2a['executor_success_count']}/5 "
                f"executor-equivalent on H2b, while H2c gains {h2c_vs_h2a['delta_exact_rate']:.1f} exact-rate. "
                "This keeps stale-selection mediation and residual exactness as distinct mechanisms."
            ),
        },
        {
            "finding_id": "h2c_surpasses_v9_executor_tie",
            "finding": (
                f"V9 tied v12 on executor-equivalence at {v9['executor_success_count']}/5 but only reached "
                f"{v9['exact_success_count']}/5 strict. H2c reaches 5/5 on both metrics."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "Do not promote H2c globally from a five-case residual fit. The next gate is a minimal transfer "
                "check over H1n/H1o/H1p/H1x residual families to detect whether scoped residual exactness harms "
                "the broader H2a executor profile."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2c Scoped Residual Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2c combines scoped residual-exactness wording with the existing stale-selection controller gate. "
            "On the five-row H2b residual packet it reaches `5 / 5` strict exact and `5 / 5` executor-equivalent, "
            "beating v12's `4 / 5` and H2a's `0 / 5` strict result. This is a local residual win, not a global "
            "promotion: H1s and H2a transfer still require a held-out transfer gate before any default change."
        ),
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Case Matrix",
        "",
        _table(payload["case_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _row_by_label(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["profile_label"] == profile_label:
            return row
    raise KeyError(profile_label)


def _comparison_by_label(rows: list[dict[str, Any]], comparison_label: str) -> dict[str, Any]:
    for row in rows:
        if row["comparison_label"] == comparison_label:
            return row
    raise KeyError(comparison_label)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_None._"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2c scoped residual synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2c_scoped_residual_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
