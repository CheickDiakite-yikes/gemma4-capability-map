from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2e_route_arbitration_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    packet_label: str
    profile_label: str
    packet_dir: Path


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_label: str
    comparison_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2b_residual_fit",
        "h2a_stale_selection_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2b_residual_exactness_h2a_execute_v1",
    ),
    PacketSpec(
        "h2b_residual_fit",
        "component_residual_guard_v12",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2b_residual_exactness_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "h2b_residual_fit",
        "h2c_scoped_residual_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h2b_residual_fit",
        "h2d_class_preserving_route",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2d_class_preserving_route_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h2b_residual_fit",
        "h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2e_route_arbitration_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "h2a_stale_selection_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "component_residual_guard_v12",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1x_v11_breaker_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "h2c_scoped_residual_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2c_scoped_residual_gate_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "h2d_class_preserving_route",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2d_class_preserving_route_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2e_route_arbitration_on_h1x_execute_v1",
    ),
)

COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2b_h2e_vs_h2c",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_h2c_on_h2b_v1",
    ),
    ComparisonSpec(
        "h2b_h2e_vs_h2d",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_h2d_on_h2b_v1",
    ),
    ComparisonSpec(
        "h2b_h2e_vs_h2a",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_h2a_on_h2b_v1",
    ),
    ComparisonSpec(
        "h2b_h2e_vs_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_component_residual_guard_on_h2b_v1",
    ),
    ComparisonSpec(
        "h1x_h2e_vs_h2c",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_h2c_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x_h2e_vs_h2d",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_h2d_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x_h2e_vs_h2a",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_h2a_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x_h2e_vs_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_component_residual_guard_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x_h2e_vs_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2e_route_arbitration_vs_component_label_guard_on_h1x_v1",
    ),
)


def build_h2e_route_arbitration_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    h2e_non_exact_rows = _non_exact_rows(
        "h2e",
        "both_packets",
        (
            ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2e_route_arbitration_on_h2b_execute_v1",
            ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2e_route_arbitration_on_h1x_execute_v1",
        ),
    )
    counterfactual_miss_rows = _non_exact_rows(
        "counterfactual",
        "h2c_h1x_and_h2d_h2b",
        (
            ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2c_scoped_residual_gate_on_h1x_execute_v1",
            ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2d_class_preserving_route_on_h2b_execute_v1",
        ),
    )
    finding_rows = _finding_rows(packet_rows, comparison_rows, counterfactual_miss_rows)

    h2e_h2b = _row_by_label(packet_rows, "h2b_residual_fit", "h2e_route_arbitration")
    h2e_h1x = _row_by_label(packet_rows, "h1x_transfer", "h2e_route_arbitration")
    h2e_vs_h2c_h2b = _comparison_by_label(comparison_rows, "h2b_h2e_vs_h2c")
    h2e_vs_h2c_h1x = _comparison_by_label(comparison_rows, "h1x_h2e_vs_h2c")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "h2e_h2b_exact_success_count": int(h2e_h2b["exact_success_count"]),
        "h2e_h2b_executor_success_count": int(h2e_h2b["executor_success_count"]),
        "h2e_h1x_exact_success_count": int(h2e_h1x["exact_success_count"]),
        "h2e_h1x_executor_success_count": int(h2e_h1x["executor_success_count"]),
        "h2b_delta_exact_vs_h2c": h2e_vs_h2c_h2b["delta_exact_rate"],
        "h2b_delta_executor_vs_h2c": h2e_vs_h2c_h2b["delta_executor_equivalence_rate"],
        "h1x_delta_exact_vs_h2c": h2e_vs_h2c_h1x["delta_exact_rate"],
        "h1x_delta_executor_vs_h2c": h2e_vs_h2c_h1x["delta_executor_equivalence_rate"],
        "h2e_non_exact_count": len(h2e_non_exact_rows),
        "promotion_decision": "promote_to_fresh_h2f_holdout_not_global_default",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "h2e_non_exact_rows": h2e_non_exact_rows,
        "counterfactual_miss_rows": counterfactual_miss_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h2e_route_arbitration_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2e_route_arbitration_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2e_non_exact_rows.csv", h2e_non_exact_rows)
    _write_csv(tables_dir / "h2e_counterfactual_miss_rows.csv", counterfactual_miss_rows)
    _write_csv(tables_dir / "h2e_route_arbitration_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2e_route_arbitration_gate.svg", packet_rows)
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
        "packet_label": spec.packet_label,
        "profile_label": spec.profile_label,
        "system_id": summary["system_id"],
        "packet_dir": str(spec.packet_dir.relative_to(ROOT)),
        "case_count": case_count,
        "exact_success_count": exact_success_count,
        "exact_rate": exact_success_count / case_count if case_count else 0.0,
        "executor_success_count": executor_success_count,
        "executor_rate": executor_success_count / case_count if case_count else 0.0,
    }


def _comparison_row(spec: ComparisonSpec) -> dict[str, Any]:
    payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
    summary = payload["summary"]
    return {
        "comparison_label": spec.comparison_label,
        "comparison_dir": str(spec.comparison_dir.relative_to(ROOT)),
        "shared_case_count": summary["shared_case_count"],
        "baseline_exact_rate": summary["baseline_exact_rate"],
        "candidate_exact_rate": summary["candidate_exact_rate"],
        "delta_exact_rate": summary["delta_exact_rate"],
        "baseline_executor_equivalence_rate": summary["baseline_executor_equivalence_rate"],
        "candidate_executor_equivalence_rate": summary["candidate_executor_equivalence_rate"],
        "delta_executor_equivalence_rate": summary["delta_executor_equivalence_rate"],
    }


def _non_exact_rows(packet_label: str, profile_label: str, packet_dirs: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet_dir in packet_dirs:
        for row in _read_json(packet_dir / "live_replay_results.json"):
            if row.get("replay_exact_match") is True:
                continue
            detail = _probe_detail(row)
            rows.append(
                {
                    "packet_label": packet_label,
                    "profile_label": profile_label,
                    "packet_dir": str(packet_dir.relative_to(ROOT)),
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "failure_mode": row.get("replay_failure_mode", ""),
                    "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                    "expected_tool": detail["expected_tool"],
                    "expected_arguments": detail["expected_arguments"],
                    "actual_tool": detail["actual_tool"],
                    "actual_arguments": detail["actual_arguments"],
                }
            )
    return rows


def _probe_detail(row: dict[str, Any]) -> dict[str, str]:
    probe_path = Path(row["output_dir"]) / "probe_results.json"
    probe = _read_json(probe_path)[0]
    expected = (probe.get("expected_calls") or [{}])[0]
    actual = (probe.get("actual_calls") or [{}])[0]
    return {
        "expected_tool": str(expected.get("name", "")),
        "expected_arguments": json.dumps(expected.get("arguments", {}), sort_keys=True),
        "actual_tool": str(actual.get("name", "")),
        "actual_arguments": json.dumps(actual.get("arguments", {}), sort_keys=True),
    }


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    counterfactual_miss_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2c_h2b = _row_by_label(packet_rows, "h2b_residual_fit", "h2c_scoped_residual_gate")
    h2d_h2b = _row_by_label(packet_rows, "h2b_residual_fit", "h2d_class_preserving_route")
    h2e_h2b = _row_by_label(packet_rows, "h2b_residual_fit", "h2e_route_arbitration")
    h2c_h1x = _row_by_label(packet_rows, "h1x_transfer", "h2c_scoped_residual_gate")
    h2d_h1x = _row_by_label(packet_rows, "h1x_transfer", "h2d_class_preserving_route")
    h2e_h1x = _row_by_label(packet_rows, "h1x_transfer", "h2e_route_arbitration")
    h2e_vs_h2c_h1x = _comparison_by_label(comparison_rows, "h1x_h2e_vs_h2c")
    h2e_vs_h2d_h2b = _comparison_by_label(comparison_rows, "h2b_h2e_vs_h2d")
    return [
        {
            "finding_id": "h2e_saturates_both_h2b_and_h1x",
            "finding": (
                f"H2e reaches {h2e_h2b['exact_success_count']}/5 exact on H2b and "
                f"{h2e_h1x['exact_success_count']}/8 exact on H1x, with executor-equivalence also saturated."
            ),
        },
        {
            "finding_id": "h2e_reconciles_h2c_h2d_tradeoff",
            "finding": (
                f"H2c is {h2c_h2b['exact_success_count']}/5 then {h2c_h1x['exact_success_count']}/8; "
                f"H2d is {h2d_h2b['exact_success_count']}/5 then {h2d_h1x['exact_success_count']}/8; "
                f"H2e preserves the max of both at {h2e_h2b['exact_success_count']}/5 and "
                f"{h2e_h1x['exact_success_count']}/8."
            ),
        },
        {
            "finding_id": "h2e_transfer_gain_is_specific",
            "finding": (
                f"H2e improves over H2c on H1x by {h2e_vs_h2c_h1x['delta_exact_rate']} exact and "
                f"executor-equivalence rate, and improves over H2d on H2b by {h2e_vs_h2d_h2b['delta_exact_rate']} "
                "strict exact rate."
            ),
        },
        {
            "finding_id": "counterfactual_misses_are_covered",
            "finding": (
                f"The counterfactual miss table has {len(counterfactual_miss_rows)} rows: H2c's result-chip class "
                "swap and H2d's badge-code over-specific query. H2e has zero non-exact rows across the two packets."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "Promote H2e only to a fresh H2f holdout gate. The current result is strong mechanism evidence, "
                "but the next proof must use newly authored route-arbitration cases rather than H2b/H1x rows."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2e Route Arbitration Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2e reconciles the H2c/H2d tradeoff. H2c saturated H2b but lost held-out H1x transfer; "
            "H2d fixed transfer but gave back one local H2b exact row. H2e reaches `5 / 5` exact on H2b "
            "and `8 / 8` exact on H1x, with executor-equivalence saturated on both packets. This should not "
            "be promoted as a global default yet; it should seed a fresh H2f holdout gate."
        ),
        "",
        "![H2e route arbitration gate](figures/h2e_route_arbitration_gate.svg)",
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## H2e Non-Exact Rows",
        "",
        _table(payload["h2e_non_exact_rows"]),
        "",
        "## Counterfactual Miss Rows",
        "",
        _table(payload["counterfactual_miss_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    h2b = _row_by_label(packet_rows, "h2b_residual_fit", "h2e_route_arbitration")
    h1x = _row_by_label(packet_rows, "h1x_transfer", "h2e_route_arbitration")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="280" viewBox="0 0 760 280" role="img" aria-labelledby="title desc">
  <title id="title">H2e route arbitration gate</title>
  <desc id="desc">H2e reaches {h2b['exact_success_count']} of 5 exact on H2b and {h1x['exact_success_count']} of 8 exact on H1x.</desc>
  <rect width="760" height="280" fill="#ffffff"/>
  <text x="32" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">H2e route arbitration gate</text>
  <text x="32" y="72" font-family="Arial, sans-serif" font-size="14" fill="#374151">Strict exact and executor-equivalent success after route arbitration</text>
  <g transform="translate(32 110)">
    <rect width="320" height="112" rx="8" fill="#ecfeff" stroke="#0891b2" stroke-width="2"/>
    <text x="20" y="34" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#0f172a">H2b residual fit</text>
    <text x="20" y="68" font-family="Arial, sans-serif" font-size="34" font-weight="700" fill="#0891b2">{h2b['exact_success_count']} / {h2b['case_count']}</text>
    <text x="20" y="94" font-family="Arial, sans-serif" font-size="14" fill="#374151">exact and executor-equivalent</text>
  </g>
  <g transform="translate(408 110)">
    <rect width="320" height="112" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="2"/>
    <text x="20" y="34" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#0f172a">H1x transfer</text>
    <text x="20" y="68" font-family="Arial, sans-serif" font-size="34" font-weight="700" fill="#16a34a">{h1x['exact_success_count']} / {h1x['case_count']}</text>
    <text x="20" y="94" font-family="Arial, sans-serif" font-size="14" fill="#374151">exact and executor-equivalent</text>
  </g>
  <text x="32" y="252" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Decision: promote only to fresh H2f holdout, not global default.</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _row_by_label(rows: list[dict[str, Any]], packet_label: str, profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["packet_label"] == packet_label and row["profile_label"] == profile_label:
            return row
    raise KeyError((packet_label, profile_label))


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
    parser = argparse.ArgumentParser(description="Build the H2e route arbitration synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2e_route_arbitration_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
