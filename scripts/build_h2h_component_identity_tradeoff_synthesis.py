from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2h_component_identity_tradeoff_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    suite: str
    profile_label: str
    packet_dir: Path


@dataclass(frozen=True)
class ComparisonSpec:
    suite: str
    comparison_label: str
    comparison_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2f",
        "h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2f_route_arbitration_h2e_execute_v1",
    ),
    PacketSpec(
        "h2f",
        "h2g_component_identity_query_contract",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1",
    ),
    PacketSpec(
        "h2f",
        "h2h_component_identity_negative_examples",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1",
    ),
    PacketSpec(
        "h2b",
        "h2c_scoped_residual_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h2b",
        "h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2e_route_arbitration_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h2b",
        "h2h_component_identity_negative_examples",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2h_component_identity_negative_examples_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h1x",
        "h2d_class_preserving_route",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2d_class_preserving_route_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1x",
        "h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2e_route_arbitration_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1x",
        "h2h_component_identity_negative_examples",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2h_component_identity_negative_examples_on_h1x_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2f",
        "h2h_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h2f_v1",
    ),
    ComparisonSpec(
        "h2f",
        "h2h_vs_h2g",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2h_component_identity_negative_examples_vs_h2g_on_h2f_v1",
    ),
    ComparisonSpec(
        "h2b",
        "h2h_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h2b_v1",
    ),
    ComparisonSpec(
        "h2b",
        "h2h_vs_h2c",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2h_component_identity_negative_examples_vs_h2c_on_h2b_v1",
    ),
    ComparisonSpec(
        "h1x",
        "h2h_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x",
        "h2h_vs_h2d",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2h_component_identity_negative_examples_vs_h2d_on_h1x_v1",
    ),
)


def build_h2h_component_identity_tradeoff_synthesis(
    *, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    h2h_non_exact_rows = _h2h_non_exact_rows(PACKET_SPECS)
    finding_rows = _finding_rows(packet_rows, comparison_rows, h2h_non_exact_rows)

    h2f_h2h = _packet_by_suite_profile(packet_rows, "h2f", "h2h_component_identity_negative_examples")
    h2b_h2h = _packet_by_suite_profile(packet_rows, "h2b", "h2h_component_identity_negative_examples")
    h1x_h2h = _packet_by_suite_profile(packet_rows, "h1x", "h2h_component_identity_negative_examples")
    h2f_delta = _comparison_by_suite_label(comparison_rows, "h2f", "h2h_vs_h2e")
    h2b_delta = _comparison_by_suite_label(comparison_rows, "h2b", "h2h_vs_h2e")
    h1x_delta = _comparison_by_suite_label(comparison_rows, "h1x", "h2h_vs_h2e")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "h2h_h2f_exact_success_count": h2f_h2h["exact_success_count"],
        "h2h_h2f_executor_success_count": h2f_h2h["executor_success_count"],
        "h2h_h2b_exact_success_count": h2b_h2h["exact_success_count"],
        "h2h_h2b_executor_success_count": h2b_h2h["executor_success_count"],
        "h2h_h1x_exact_success_count": h1x_h2h["exact_success_count"],
        "h2h_h1x_executor_success_count": h1x_h2h["executor_success_count"],
        "h2h_delta_exact_vs_h2e_on_h2f": h2f_delta["delta_exact_rate"],
        "h2h_delta_exact_vs_h2e_on_h2b": h2b_delta["delta_exact_rate"],
        "h2h_delta_exact_vs_h2e_on_h1x": h1x_delta["delta_exact_rate"],
        "h2h_non_exact_count": len(h2h_non_exact_rows),
        "promotion_decision": "reject_global_h2h_keep_as_h2f_scoped_repair",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "h2h_non_exact_rows": h2h_non_exact_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2h_tradeoff_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2h_tradeoff_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2h_tradeoff_non_exact_rows.csv", h2h_non_exact_rows)
    _write_csv(tables_dir / "h2h_tradeoff_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2h_tradeoff_gate.svg", packet_rows)
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
        "suite": spec.suite,
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
        "suite": spec.suite,
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


def _h2h_non_exact_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.profile_label != "h2h_component_identity_negative_examples":
            continue
        for row in _read_json(spec.packet_dir / "live_replay_results.json"):
            if row.get("replay_exact_match") is True:
                continue
            detail = _probe_detail(row)
            rows.append(
                {
                    "suite": spec.suite,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "failure_mode": row.get("replay_failure_mode", ""),
                    "expected_tool": detail["expected_tool"],
                    "expected_target_query": detail["expected_target_query"],
                    "actual_tool": detail["actual_tool"],
                    "actual_target_query": detail["actual_target_query"],
                    "query_error_class": _query_error_class(detail),
                }
            )
    return rows


def _probe_detail(row: dict[str, Any]) -> dict[str, str]:
    probe_path = Path(row["output_dir"]) / "probe_results.json"
    probe = _read_json(probe_path)[0]
    expected = (probe.get("expected_calls") or [{}])[0]
    actual = (probe.get("actual_calls") or [{}])[0]
    expected_args = expected.get("arguments", {})
    actual_args = actual.get("arguments", {})
    return {
        "expected_tool": str(expected.get("name", "")),
        "expected_target_query": str(expected_args.get("target_query", "")),
        "actual_tool": str(actual.get("name", "")),
        "actual_target_query": str(actual_args.get("target_query", "")),
    }


def _query_error_class(detail: dict[str, str]) -> str:
    if detail["actual_tool"] != detail["expected_tool"]:
        return "wrong_tool"
    expected = detail["expected_target_query"]
    actual = detail["actual_target_query"]
    if not actual:
        return "missing_query"
    if actual.startswith(expected) or expected.startswith(actual):
        return "alias_expansion_or_class_swap"
    return "component_class_or_value_substitution"


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    h2h_non_exact_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2f_h2h = _packet_by_suite_profile(packet_rows, "h2f", "h2h_component_identity_negative_examples")
    h2b_h2h = _packet_by_suite_profile(packet_rows, "h2b", "h2h_component_identity_negative_examples")
    h1x_h2h = _packet_by_suite_profile(packet_rows, "h1x", "h2h_component_identity_negative_examples")
    h2f_delta = _comparison_by_suite_label(comparison_rows, "h2f", "h2h_vs_h2e")
    h2b_delta = _comparison_by_suite_label(comparison_rows, "h2b", "h2h_vs_h2e")
    h1x_delta = _comparison_by_suite_label(comparison_rows, "h1x", "h2h_vs_h2e")
    residuals = "; ".join(
        f"{row['suite']}:{row['expected_target_query']}->{row['actual_target_query']}"
        for row in h2h_non_exact_rows
    )
    return [
        {
            "finding_id": "h2h_repairs_fresh_h2f",
            "finding": (
                f"H2h reaches {h2f_h2h['exact_success_count']}/{h2f_h2h['case_count']} strict and "
                f"{h2f_h2h['executor_success_count']}/{h2f_h2h['case_count']} executor-equivalent on H2f, "
                f"a {h2f_delta['delta_exact_rate']} exact-rate lift over H2e."
            ),
        },
        {
            "finding_id": "h2h_regresses_prior_transfer_gates",
            "finding": (
                f"H2h falls to {h2b_h2h['exact_success_count']}/{h2b_h2h['case_count']} on H2b "
                f"({h2b_delta['delta_exact_rate']} versus H2e) and "
                f"{h1x_h2h['exact_success_count']}/{h1x_h2h['case_count']} on H1x "
                f"({h1x_delta['delta_exact_rate']} versus H2e)."
            ),
        },
        {
            "finding_id": "h2h_failure_boundary",
            "finding": (
                "The residual substitutions are concentrated in component-class transfer and alias expansion: "
                f"{residuals}."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "The next hypothesis should be conditional arbitration, not a broader negative-example paragraph: "
                "keep H2e's route arbitration as the default and activate H2h-style negative examples only for "
                "explicit displayed-value component-identity prompts."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2h Component-Identity Tradeoff Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2h is a strong scoped repair for the fresh H2f holdout but not a global successor. It raises H2f "
            "from H2e/H2g's 6/10 strict exactness to 9/10, yet regresses the prior H2b and H1x gates that H2e "
            "had saturated. The research interpretation is that explicit negative examples can causally repair "
            "displayed-value component-identity failures, but the same prose can over-constrain related component "
            "classes and code-label rows."
        ),
        "",
        "![H2h tradeoff gate](figures/h2h_tradeoff_gate.svg)",
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## H2h Non-Exact Rows",
        "",
        _table(payload["h2h_non_exact_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    suites = ["h2f", "h2b", "h1x"]
    profiles = [
        "h2e_route_arbitration",
        "h2h_component_identity_negative_examples",
    ]
    rows = [
        row
        for row in packet_rows
        if row["suite"] in suites and row["profile_label"] in profiles
    ]
    width = 880
    height = 120 + len(rows) * 42
    left = 290
    top = 72
    bar_width = 410
    bar_height = 22
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2h transfer tradeoff gate</title>',
        '<desc id="desc">H2h improves H2f but regresses H2b and H1x against H2e.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="23" font-weight="700" fill="#111827">H2h is scoped, not global</text>',
        '<text x="32" y="58" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Strict exact success; executor-equivalence shown as count.</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * 42
        exact_width = int(bar_width * float(row["exact_rate"]))
        label = f"{row['suite']} / {row['profile_label']}"
        fill = "#155e75" if row["profile_label"] == "h2h_component_identity_negative_examples" else "#0891b2"
        parts.extend(
            [
                f'<text x="32" y="{y + 16}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" rx="4" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y}" width="{exact_width}" height="{bar_height}" rx="4" fill="{fill}"/>',
                f'<text x="{left + bar_width + 18}" y="{y + 16}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{row["exact_success_count"]}/{row["case_count"]} exact, {row["executor_success_count"]}/{row["case_count"]} exec</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _packet_by_suite_profile(rows: list[dict[str, Any]], suite: str, profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["suite"] == suite and row["profile_label"] == profile_label:
            return row
    raise KeyError((suite, profile_label))


def _comparison_by_suite_label(rows: list[dict[str, Any]], suite: str, comparison_label: str) -> dict[str, Any]:
    for row in rows:
        if row["suite"] == suite and row["comparison_label"] == comparison_label:
            return row
    raise KeyError((suite, comparison_label))


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
    parser = argparse.ArgumentParser(description="Build H2h component-identity transfer tradeoff synthesis.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2h_component_identity_tradeoff_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
