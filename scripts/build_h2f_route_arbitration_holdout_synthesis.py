from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2f_route_arbitration_holdout_synthesis"


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
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2f_route_arbitration_no_directive_execute_v1",
    ),
    PacketSpec(
        "h2a_component_label_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2f_route_arbitration_h2a_execute_v1",
    ),
    PacketSpec(
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2f_route_arbitration_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "h2c_scoped_residual_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2f_route_arbitration_h2c_execute_v1",
    ),
    PacketSpec(
        "h2d_class_preserving_route",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2f_route_arbitration_h2d_execute_v1",
    ),
    PacketSpec(
        "h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2f_route_arbitration_h2e_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2f_h2e_vs_h2c",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2f_route_arbitration_h2e_vs_h2c_v1",
    ),
    ComparisonSpec(
        "h2f_h2e_vs_h2d",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2f_route_arbitration_h2e_vs_h2d_v1",
    ),
    ComparisonSpec(
        "h2f_h2e_vs_h2a",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2f_route_arbitration_h2e_vs_h2a_v1",
    ),
    ComparisonSpec(
        "h2f_h2e_vs_component_residual_guard",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2f_route_arbitration_h2e_vs_component_residual_guard_v1",
    ),
    ComparisonSpec(
        "h2f_h2e_vs_no_directive",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2f_route_arbitration_h2e_vs_no_directive_v1",
    ),
)


def build_h2f_route_arbitration_holdout_synthesis(
    *, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    h2e_non_exact_rows = _non_exact_rows(
        "h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2f_route_arbitration_h2e_execute_v1",
    )
    all_non_exact_rows = _all_non_exact_rows(PACKET_SPECS)
    family_rows = _family_rows(PACKET_SPECS)
    failure_mode_rows = _failure_mode_rows(PACKET_SPECS)
    finding_rows = _finding_rows(packet_rows, comparison_rows, h2e_non_exact_rows)

    h2e = _packet_by_profile(packet_rows, "h2e_route_arbitration")
    h2c = _packet_by_profile(packet_rows, "h2c_scoped_residual_gate")
    no_directive = _packet_by_profile(packet_rows, "no_directive")
    h2e_vs_h2c = _comparison_by_label(comparison_rows, "h2f_h2e_vs_h2c")
    h2e_vs_no_directive = _comparison_by_label(comparison_rows, "h2f_h2e_vs_no_directive")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "h2e_exact_success_count": int(h2e["exact_success_count"]),
        "h2e_executor_success_count": int(h2e["executor_success_count"]),
        "h2c_exact_success_count": int(h2c["exact_success_count"]),
        "no_directive_exact_success_count": int(no_directive["exact_success_count"]),
        "h2e_non_exact_count": len(h2e_non_exact_rows),
        "h2e_delta_exact_vs_h2c": h2e_vs_h2c["delta_exact_rate"],
        "h2e_delta_executor_vs_h2c": h2e_vs_h2c["delta_executor_equivalence_rate"],
        "h2e_delta_exact_vs_no_directive": h2e_vs_no_directive["delta_exact_rate"],
        "h2e_delta_executor_vs_no_directive": h2e_vs_no_directive["delta_executor_equivalence_rate"],
        "h2e_failure_family_count": len({row["family"] for row in h2e_non_exact_rows}),
        "promotion_decision": "reject_global_h2e_build_h2g_component_identity_query_contract",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "h2e_non_exact_rows": h2e_non_exact_rows,
        "all_non_exact_rows": all_non_exact_rows,
        "family_rows": family_rows,
        "failure_mode_rows": failure_mode_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2f_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2f_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2f_h2e_non_exact_rows.csv", h2e_non_exact_rows)
    _write_csv(tables_dir / "h2f_all_non_exact_rows.csv", all_non_exact_rows)
    _write_csv(tables_dir / "h2f_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2f_failure_mode_summary.csv", failure_mode_rows)
    _write_csv(tables_dir / "h2f_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2f_holdout_profile_bars.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _packet_row(spec: PacketSpec) -> dict[str, Any]:
    summary = _read_json(spec.packet_dir / "summary.json")
    results = _read_json(spec.packet_dir / "live_replay_results.json")
    case_count = int(summary["case_count"])
    exact_success_count = sum(1 for row in results if row.get("replay_exact_match") is True)
    executable_success_count = sum(1 for row in results if row.get("replay_executable_match") is True)
    executor_success_count = sum(1 for row in results if row.get("replay_executor_equivalence_match") is True)
    return {
        "profile_label": spec.profile_label,
        "system_id": summary["system_id"],
        "packet_dir": str(spec.packet_dir.relative_to(ROOT)),
        "case_count": case_count,
        "exact_success_count": exact_success_count,
        "exact_rate": exact_success_count / case_count if case_count else 0.0,
        "executable_success_count": executable_success_count,
        "executable_rate": executable_success_count / case_count if case_count else 0.0,
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
        "baseline_executable_rate": summary["baseline_executable_rate"],
        "candidate_executable_rate": summary["candidate_executable_rate"],
        "delta_executable_rate": summary["delta_executable_rate"],
        "baseline_executor_equivalence_rate": summary["baseline_executor_equivalence_rate"],
        "candidate_executor_equivalence_rate": summary["candidate_executor_equivalence_rate"],
        "delta_executor_equivalence_rate": summary["delta_executor_equivalence_rate"],
    }


def _non_exact_rows(profile_label: str, packet_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_json(packet_dir / "live_replay_results.json"):
        if row.get("replay_exact_match") is True:
            continue
        detail = _probe_detail(row)
        rows.append(
            {
                "profile_label": profile_label,
                "packet_dir": str(packet_dir.relative_to(ROOT)),
                "case_id": row["case_id"],
                "family": row.get("family", ""),
                "failure_mode": row.get("replay_failure_mode", ""),
                "executable_match": row.get("replay_executable_match"),
                "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                "expected_tool": detail["expected_tool"],
                "expected_target_query": detail["expected_target_query"],
                "actual_tool": detail["actual_tool"],
                "actual_target_query": detail["actual_target_query"],
                "query_error_class": _query_error_class(detail),
            }
        )
    return rows


def _all_non_exact_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.extend(_non_exact_rows(spec.profile_label, spec.packet_dir))
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
    if not detail["actual_tool"]:
        return "no_tool_call"
    if detail["actual_tool"] != detail["expected_tool"]:
        return "wrong_tool"
    if not detail["actual_target_query"]:
        return "missing_query"
    if detail["actual_target_query"] != detail["expected_target_query"]:
        return "value_or_alias_query_substitution"
    return "other_non_exact"


def _family_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        summary: dict[tuple[str, str], dict[str, Any]] = {}
        for row in _read_json(spec.packet_dir / "live_replay_results.json"):
            key = (spec.profile_label, str(row.get("family", "")))
            bucket = summary.setdefault(
                key,
                {
                    "profile_label": spec.profile_label,
                    "family": row.get("family", ""),
                    "case_count": 0,
                    "exact_success_count": 0,
                    "executor_success_count": 0,
                },
            )
            bucket["case_count"] += 1
            bucket["exact_success_count"] += 1 if row.get("replay_exact_match") is True else 0
            bucket["executor_success_count"] += 1 if row.get("replay_executor_equivalence_match") is True else 0
        for bucket in summary.values():
            case_count = int(bucket["case_count"])
            rows.append(
                {
                    **bucket,
                    "exact_rate": bucket["exact_success_count"] / case_count if case_count else 0.0,
                    "executor_rate": bucket["executor_success_count"] / case_count if case_count else 0.0,
                }
            )
    return rows


def _failure_mode_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        counts: dict[str, int] = {}
        for row in _read_json(spec.packet_dir / "live_replay_results.json"):
            mode = str(row.get("replay_failure_mode", ""))
            counts[mode] = counts.get(mode, 0) + 1
        for mode, count in sorted(counts.items()):
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "failure_mode": mode,
                    "count": count,
                }
            )
    return rows


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    h2e_non_exact_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2e = _packet_by_profile(packet_rows, "h2e_route_arbitration")
    h2c = _packet_by_profile(packet_rows, "h2c_scoped_residual_gate")
    h2d = _packet_by_profile(packet_rows, "h2d_class_preserving_route")
    h2a = _packet_by_profile(packet_rows, "h2a_component_label_guard")
    v12 = _packet_by_profile(packet_rows, "component_residual_guard_v12")
    no_directive = _packet_by_profile(packet_rows, "no_directive")
    h2e_vs_h2c = _comparison_by_label(comparison_rows, "h2f_h2e_vs_h2c")
    h2e_vs_no_directive = _comparison_by_label(comparison_rows, "h2f_h2e_vs_no_directive")
    failed_families = ", ".join(sorted({row["family"] for row in h2e_non_exact_rows}))
    target_swaps = ", ".join(
        f"{row['expected_target_query']}->{row['actual_target_query']}" for row in h2e_non_exact_rows
    )
    return [
        {
            "finding_id": "h2f_breaks_h2e_saturation",
            "finding": (
                f"H2e reaches only {h2e['exact_success_count']}/10 exact and "
                f"{h2e['executor_success_count']}/10 executor-equivalent on the fresh H2f holdout, after "
                "previously saturating H2b and H1x."
            ),
        },
        {
            "finding_id": "route_arbitration_does_not_beat_h2c_on_h2f",
            "finding": (
                f"H2e ties H2c on H2f: delta exact={h2e_vs_h2c['delta_exact_rate']} and delta "
                f"executor-equivalence={h2e_vs_h2c['delta_executor_equivalence_rate']}."
            ),
        },
        {
            "finding_id": "controllers_remain_causal_against_floor",
            "finding": (
                f"No-directive reaches {no_directive['exact_success_count']}/10 exact while H2e reaches "
                f"{h2e['exact_success_count']}/10, a {h2e_vs_no_directive['delta_exact_rate']} exact-rate lift. "
                f"Intermediate rows are H2a={h2a['exact_success_count']}/10, v12={v12['exact_success_count']}/10, "
                f"H2d={h2d['exact_success_count']}/10, and H2c={h2c['exact_success_count']}/10."
            ),
        },
        {
            "finding_id": "remaining_failure_is_component_identity_binding",
            "finding": (
                f"All H2e non-exact rows are argument mismatches in {failed_families}. The model preserved the "
                f"right tool but substituted displayed values or aliases for requested component identities: {target_swaps}."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "Do not promote H2e globally. Build H2g around a component-identity query contract: when the user "
                "asks for a component class or visible label, the target_query should preserve that requested phrase "
                "instead of collapsing to the component value."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2f Route Arbitration Holdout Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2f is the fresh holdout that was supposed to test whether H2e route arbitration generalized beyond "
            "the saturated H2b/H1x gates. It does not. H2e keeps a large advantage over the no-directive floor, "
            "but it ties H2c and fails four cases by calling the right tool with the wrong query. The residual "
            "problem is component-identity binding under displayed-value decoys."
        ),
        "",
        "![H2f holdout profile bars](figures/h2f_holdout_profile_bars.svg)",
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
        "## Family Rows",
        "",
        _table(payload["family_rows"]),
        "",
        "## Failure Mode Rows",
        "",
        _table(payload["failure_mode_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    width = 920
    height = 360
    left = 280
    top = 68
    bar_width = 520
    bar_height = 24
    gap = 24
    rows = sorted(packet_rows, key=lambda row: float(row["exact_rate"]))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2f holdout profile bars</title>',
        '<desc id="desc">Exact and executor-equivalent success rates across H2f route-arbitration holdout rows.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="23" font-weight="700" fill="#111827">H2f fresh holdout breaks saturation</text>',
        '<text x="32" y="58" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Strict exact bars; executor-equivalence shown as right-side count.</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * (bar_height + gap)
        exact_width = int(bar_width * float(row["exact_rate"]))
        label = row["profile_label"]
        exact_count = row["exact_success_count"]
        executor_count = row["executor_success_count"]
        case_count = row["case_count"]
        fill = "#0891b2" if label == "h2e_route_arbitration" else "#64748b"
        parts.extend(
            [
                f'<text x="32" y="{y + 18}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" rx="4" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y}" width="{exact_width}" height="{bar_height}" rx="4" fill="{fill}"/>',
                f'<text x="{left + bar_width + 18}" y="{y + 18}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{exact_count}/{case_count} exact, {executor_count}/{case_count} exec</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _packet_by_profile(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="Build the H2f route-arbitration holdout synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2f_route_arbitration_holdout_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
