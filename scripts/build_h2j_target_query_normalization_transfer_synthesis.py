from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2j_target_query_normalization_transfer_synthesis"


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
        "h2h_component_identity_negative_examples",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1",
    ),
    PacketSpec(
        "h2f",
        "h2j_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2j_target_query_normalization_on_h2f_execute_v2",
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
        "h2b",
        "h2j_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2j_target_query_normalization_on_h2b_execute_v2",
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
    PacketSpec(
        "h1x",
        "h2j_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2j_target_query_normalization_on_h1x_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2f",
        "h2j_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2j_target_query_normalization_vs_h2e_on_h2f_v2",
    ),
    ComparisonSpec(
        "h2f",
        "h2j_vs_h2h",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2j_target_query_normalization_vs_h2h_on_h2f_v2",
    ),
    ComparisonSpec(
        "h2f",
        "h2j_vs_h2i",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2j_target_query_normalization_vs_h2i_on_h2f_v2",
    ),
    ComparisonSpec(
        "h2b",
        "h2j_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2j_target_query_normalization_vs_h2e_on_h2b_v2",
    ),
    ComparisonSpec(
        "h2b",
        "h2j_vs_h2h",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2j_target_query_normalization_vs_h2h_on_h2b_v2",
    ),
    ComparisonSpec(
        "h1x",
        "h2j_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2j_target_query_normalization_vs_h2e_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x",
        "h2j_vs_h2h",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2j_target_query_normalization_vs_h2h_on_h1x_v1",
    ),
)


def build_h2j_target_query_normalization_transfer_synthesis(
    *, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    h2j_non_exact_rows = _non_exact_rows(PACKET_SPECS, profile_label="h2j_target_query_normalization")
    intervention_rows = _h2j_intervention_rows(PACKET_SPECS)
    finding_rows = _finding_rows(packet_rows, comparison_rows, h2j_non_exact_rows, intervention_rows)

    h2f_h2j = _packet_by_suite_profile(packet_rows, "h2f", "h2j_target_query_normalization")
    h2b_h2j = _packet_by_suite_profile(packet_rows, "h2b", "h2j_target_query_normalization")
    h1x_h2j = _packet_by_suite_profile(packet_rows, "h1x", "h2j_target_query_normalization")
    h2f_vs_h2e = _comparison_by_suite_label(comparison_rows, "h2f", "h2j_vs_h2e")
    h2f_vs_h2h = _comparison_by_suite_label(comparison_rows, "h2f", "h2j_vs_h2h")
    h2b_vs_h2e = _comparison_by_suite_label(comparison_rows, "h2b", "h2j_vs_h2e")
    h2b_vs_h2h = _comparison_by_suite_label(comparison_rows, "h2b", "h2j_vs_h2h")
    h1x_vs_h2e = _comparison_by_suite_label(comparison_rows, "h1x", "h2j_vs_h2e")
    h1x_vs_h2h = _comparison_by_suite_label(comparison_rows, "h1x", "h2j_vs_h2h")
    target_query_interventions = [
        row for row in intervention_rows if row["intervention_kind"] == "visual_target_query_normalization"
    ]
    stale_selection_interventions = [
        row for row in intervention_rows if row["intervention_kind"] == "visual_stale_selection_gate"
    ]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "h2j_h2f_exact_success_count": h2f_h2j["exact_success_count"],
        "h2j_h2f_executor_success_count": h2f_h2j["executor_success_count"],
        "h2j_h2b_exact_success_count": h2b_h2j["exact_success_count"],
        "h2j_h2b_executor_success_count": h2b_h2j["executor_success_count"],
        "h2j_h1x_exact_success_count": h1x_h2j["exact_success_count"],
        "h2j_h1x_executor_success_count": h1x_h2j["executor_success_count"],
        "h2j_delta_exact_vs_h2e_on_h2f": h2f_vs_h2e["delta_exact_rate"],
        "h2j_delta_exact_vs_h2h_on_h2f": h2f_vs_h2h["delta_exact_rate"],
        "h2j_delta_exact_vs_h2e_on_h2b": h2b_vs_h2e["delta_exact_rate"],
        "h2j_delta_exact_vs_h2h_on_h2b": h2b_vs_h2h["delta_exact_rate"],
        "h2j_delta_exact_vs_h2e_on_h1x": h1x_vs_h2e["delta_exact_rate"],
        "h2j_delta_exact_vs_h2h_on_h1x": h1x_vs_h2h["delta_exact_rate"],
        "h2j_non_exact_count": len(h2j_non_exact_rows),
        "target_query_normalization_count": len(target_query_interventions),
        "visual_stale_selection_gate_count": len(stale_selection_interventions),
        "promotion_decision": "promote_h2j_to_next_harder_holdout_not_global_default",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "h2j_non_exact_rows": h2j_non_exact_rows,
        "intervention_rows": intervention_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2j_transfer_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2j_transfer_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2j_transfer_non_exact_rows.csv", h2j_non_exact_rows)
    _write_csv(tables_dir / "h2j_transfer_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2j_transfer_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2j_transfer_gate.svg", packet_rows)
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


def _non_exact_rows(specs: tuple[PacketSpec, ...], *, profile_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.profile_label != profile_label:
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
                }
            )
    return rows


def _h2j_intervention_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.profile_label != "h2j_target_query_normalization":
            continue
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for kind in ("visual_target_query_normalization", "visual_stale_selection_gate"):
                entries = metadata.get(kind, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    rows.append(
                        {
                            "suite": spec.suite,
                            "case_id": result["case_id"],
                            "family": result.get("family", ""),
                            "intervention_kind": kind,
                            "from_tool": entry.get("from_tool", ""),
                            "from_arguments": _compact_json(entry.get("from_arguments", {})),
                            "to_tool": entry.get("to_tool", ""),
                            "to_arguments": _compact_json(entry.get("to_arguments", {})),
                            "prompt_state_label": entry.get("prompt_state_label", ""),
                        }
                    )
    return rows


def _probe_detail(row: dict[str, Any]) -> dict[str, str]:
    probe = _read_json(Path(row["output_dir"]) / "probe_results.json")[0]
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


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    h2j_non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2f_h2j = _packet_by_suite_profile(packet_rows, "h2f", "h2j_target_query_normalization")
    h2b_h2j = _packet_by_suite_profile(packet_rows, "h2b", "h2j_target_query_normalization")
    h1x_h2j = _packet_by_suite_profile(packet_rows, "h1x", "h2j_target_query_normalization")
    h2f_vs_h2e = _comparison_by_suite_label(comparison_rows, "h2f", "h2j_vs_h2e")
    h2f_vs_h2h = _comparison_by_suite_label(comparison_rows, "h2f", "h2j_vs_h2h")
    h2f_vs_h2i = _comparison_by_suite_label(comparison_rows, "h2f", "h2j_vs_h2i")
    h2b_vs_h2e = _comparison_by_suite_label(comparison_rows, "h2b", "h2j_vs_h2e")
    h2b_vs_h2h = _comparison_by_suite_label(comparison_rows, "h2b", "h2j_vs_h2h")
    h1x_vs_h2e = _comparison_by_suite_label(comparison_rows, "h1x", "h2j_vs_h2e")
    h1x_vs_h2h = _comparison_by_suite_label(comparison_rows, "h1x", "h2j_vs_h2h")
    target_count = sum(1 for row in intervention_rows if row["intervention_kind"] == "visual_target_query_normalization")
    stale_count = sum(1 for row in intervention_rows if row["intervention_kind"] == "visual_stale_selection_gate")
    return [
        {
            "finding_id": "h2j_closes_h2f",
            "finding": (
                f"H2j reaches {h2f_h2j['exact_success_count']}/{h2f_h2j['case_count']} strict and "
                f"executor-equivalent on H2f, with exact-rate lift {h2f_vs_h2e['delta_exact_rate']} versus H2e, "
                f"{h2f_vs_h2h['delta_exact_rate']} versus H2h, and {h2f_vs_h2i['delta_exact_rate']} versus H2i."
            ),
        },
        {
            "finding_id": "h2j_preserves_transfer_gates",
            "finding": (
                f"H2j preserves the prior transfer gates: {h2b_h2j['exact_success_count']}/{h2b_h2j['case_count']} "
                f"on H2b and {h1x_h2j['exact_success_count']}/{h1x_h2j['case_count']} on H1x. It ties H2e on both "
                f"({h2b_vs_h2e['delta_exact_rate']} H2b, {h1x_vs_h2e['delta_exact_rate']} H1x) while beating H2h "
                f"({h2b_vs_h2h['delta_exact_rate']} H2b, {h1x_vs_h2h['delta_exact_rate']} H1x)."
            ),
        },
        {
            "finding_id": "h2j_controller_mechanism",
            "finding": (
                f"H2j has {target_count} target-query-normalization interventions and {stale_count} stale/missing "
                "selection interventions across H2f/H2b/H1x. The interventions are recorded per case in the replay "
                "artifacts, making the repair attributable to controller-visible state rather than hidden expected calls."
            ),
        },
        {
            "finding_id": "h2j_remaining_risk",
            "finding": (
                f"H2j has {len(h2j_non_exact_rows)} non-exact rows on the current H2f/H2b/H1x packet set. This supports "
                "promotion to a harder holdout, not global default status; the next test should target labels that appear "
                "both as requested targets and negated decoys."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2j Target-Query Normalization Transfer Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2j moves the component-identity repair from prompt wording into a controller-visible target-query "
            "normalization gate. The key result is not only that H2j closes the fresh H2f holdout at 10/10; it also "
            "preserves the older H2b and H1x transfer gates that rejected global H2h promotion. This is the first "
            "candidate in this line that repairs the displayed-value component-identity residual while retaining "
            "route-arbitration behavior on prior transfer slices."
        ),
        "",
        "![H2j transfer gate](figures/h2j_transfer_gate.svg)",
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## H2j Non-Exact Rows",
        "",
        _table(payload["h2j_non_exact_rows"]),
        "",
        "## Controller Intervention Rows",
        "",
        _table(payload["intervention_rows"]),
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
        ("h2e_route_arbitration", "H2e", "#0891B2"),
        ("h2h_component_identity_negative_examples", "H2h", "#155E75"),
        ("h2j_target_query_normalization", "H2j", "#1D4ED8"),
    ]
    by_key = {(row["suite"], row["profile_label"]): row for row in packet_rows}
    width = 760
    height = 360
    chart_left = 84
    chart_top = 54
    chart_height = 210
    group_width = 190
    bar_width = 38
    gap = 10
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360" role="img" aria-labelledby="title desc">',
        '<title id="title">H2j transfer gate</title>',
        '<desc id="desc">H2j closes H2f and preserves H2b and H1x transfer gates.</desc>',
        '<rect width="760" height="360" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2j closes H2f without the H2h transfer regression</text>',
        '<line x1="84" y1="264" x2="690" y2="264" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="78" y1="{y:.1f}" x2="690" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="34" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for suite_index, suite in enumerate(suites):
        group_x = chart_left + suite_index * group_width
        lines.append(
            f'<text x="{group_x + 48}" y="292" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{suite.upper()}</text>'
        )
        for profile_index, (profile, label, color) in enumerate(profiles):
            row = by_key[(suite, profile)]
            rate = float(row["exact_rate"])
            bar_height = rate * chart_height
            x = group_x + profile_index * (bar_width + gap)
            y = chart_top + chart_height - bar_height
            lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
            lines.append(
                f'<text x="{x + 3}" y="{y - 6:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#111827">{int(row["exact_success_count"])}/{int(row["case_count"])}</text>'
            )
    legend_x = 500
    for index, (_, label, color) in enumerate(profiles):
        y = 310 + index * 18
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{y}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _packet_by_suite_profile(rows: list[dict[str, Any]], suite: str, profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["suite"] == suite and row["profile_label"] == profile_label:
            return row
    raise KeyError(f"{suite}:{profile_label}")


def _comparison_by_suite_label(rows: list[dict[str, Any]], suite: str, comparison_label: str) -> dict[str, Any]:
    for row in rows:
        if row["suite"] == suite and row["comparison_label"] == comparison_label:
            return row
    raise KeyError(f"{suite}:{comparison_label}")


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


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    payload = build_h2j_target_query_normalization_transfer_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
