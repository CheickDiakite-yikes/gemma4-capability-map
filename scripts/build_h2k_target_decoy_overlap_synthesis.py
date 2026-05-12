from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2k_target_decoy_overlap_synthesis"


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
        "h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2k_target_decoy_overlap_h2e_execute_v1",
    ),
    PacketSpec(
        "h2h_component_identity_negative_examples",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2k_target_decoy_overlap_h2h_execute_v1",
    ),
    PacketSpec(
        "h2j_target_query_normalization_no_stale_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2k_target_decoy_overlap_h2j_no_stale_gate_execute_v1",
    ),
    PacketSpec(
        "h2j_target_query_normalization",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2k_target_decoy_overlap_h2j_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2j_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2k_target_decoy_overlap_h2j_vs_h2e_v1",
    ),
    ComparisonSpec(
        "h2j_vs_h2h",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2k_target_decoy_overlap_h2j_vs_h2h_v1",
    ),
    ComparisonSpec(
        "h2j_vs_no_stale_gate",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2k_target_decoy_overlap_h2j_vs_no_stale_gate_v1",
    ),
)


def build_h2k_target_decoy_overlap_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    non_exact_rows = _non_exact_rows(PACKET_SPECS)
    intervention_rows = _h2j_intervention_rows(PACKET_SPECS)
    finding_rows = _finding_rows(packet_rows, comparison_rows, non_exact_rows, intervention_rows)

    h2e = _packet_by_profile(packet_rows, "h2e_route_arbitration")
    h2h = _packet_by_profile(packet_rows, "h2h_component_identity_negative_examples")
    h2j_no_stale = _packet_by_profile(packet_rows, "h2j_target_query_normalization_no_stale_gate")
    h2j = _packet_by_profile(packet_rows, "h2j_target_query_normalization")
    h2j_vs_h2e = _comparison_by_label(comparison_rows, "h2j_vs_h2e")
    h2j_vs_h2h = _comparison_by_label(comparison_rows, "h2j_vs_h2h")
    h2j_vs_no_stale = _comparison_by_label(comparison_rows, "h2j_vs_no_stale_gate")
    target_interventions = [
        row
        for row in intervention_rows
        if row["profile_label"] == "h2j_target_query_normalization"
        and row["intervention_kind"] == "visual_target_query_normalization"
    ]
    stale_interventions = [
        row
        for row in intervention_rows
        if row["profile_label"] == "h2j_target_query_normalization"
        and row["intervention_kind"] == "visual_stale_selection_gate"
    ]
    no_stale_target_interventions = [
        row
        for row in intervention_rows
        if row["profile_label"] == "h2j_target_query_normalization_no_stale_gate"
        and row["intervention_kind"] == "visual_target_query_normalization"
    ]
    no_stale_stale_interventions = [
        row
        for row in intervention_rows
        if row["profile_label"] == "h2j_target_query_normalization_no_stale_gate"
        and row["intervention_kind"] == "visual_stale_selection_gate"
    ]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "h2e_exact_success_count": h2e["exact_success_count"],
        "h2e_executor_success_count": h2e["executor_success_count"],
        "h2h_exact_success_count": h2h["exact_success_count"],
        "h2h_executor_success_count": h2h["executor_success_count"],
        "h2j_no_stale_exact_success_count": h2j_no_stale["exact_success_count"],
        "h2j_no_stale_executor_success_count": h2j_no_stale["executor_success_count"],
        "h2j_exact_success_count": h2j["exact_success_count"],
        "h2j_executor_success_count": h2j["executor_success_count"],
        "h2j_delta_exact_vs_h2e": h2j_vs_h2e["delta_exact_rate"],
        "h2j_delta_executor_vs_h2e": h2j_vs_h2e["delta_executor_equivalence_rate"],
        "h2j_delta_exact_vs_h2h": h2j_vs_h2h["delta_exact_rate"],
        "h2j_delta_executor_vs_h2h": h2j_vs_h2h["delta_executor_equivalence_rate"],
        "h2j_delta_exact_vs_no_stale_gate": h2j_vs_no_stale["delta_exact_rate"],
        "h2j_delta_executor_vs_no_stale_gate": h2j_vs_no_stale["delta_executor_equivalence_rate"],
        "h2e_non_exact_count": sum(1 for row in non_exact_rows if row["profile_label"] == "h2e_route_arbitration"),
        "h2h_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"] == "h2h_component_identity_negative_examples"
        ),
        "h2j_no_stale_non_exact_count": sum(
            1
            for row in non_exact_rows
            if row["profile_label"] == "h2j_target_query_normalization_no_stale_gate"
        ),
        "h2j_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"] == "h2j_target_query_normalization"
        ),
        "target_query_normalization_count": len(target_interventions),
        "visual_stale_selection_gate_count": len(stale_interventions),
        "h2j_no_stale_target_query_normalization_count": len(no_stale_target_interventions),
        "h2j_no_stale_visual_stale_selection_gate_count": len(no_stale_stale_interventions),
        "promotion_decision": "h2k_supports_target_query_normalization_not_stale_selection_gate",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2k_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2k_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2k_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2k_h2j_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2k_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2k_target_decoy_overlap_gate.svg", packet_rows)
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


def _non_exact_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for row in _read_json(spec.packet_dir / "live_replay_results.json"):
            if row.get("replay_exact_match") is True:
                continue
            detail = _probe_detail(row)
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "failure_mode": row.get("replay_failure_mode", ""),
                    "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
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
        if spec.profile_label not in {
            "h2j_target_query_normalization",
            "h2j_target_query_normalization_no_stale_gate",
        }:
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
                            "profile_label": spec.profile_label,
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
    non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2e = _packet_by_profile(packet_rows, "h2e_route_arbitration")
    h2h = _packet_by_profile(packet_rows, "h2h_component_identity_negative_examples")
    h2j_no_stale = _packet_by_profile(packet_rows, "h2j_target_query_normalization_no_stale_gate")
    h2j = _packet_by_profile(packet_rows, "h2j_target_query_normalization")
    h2j_vs_h2e = _comparison_by_label(comparison_rows, "h2j_vs_h2e")
    h2j_vs_h2h = _comparison_by_label(comparison_rows, "h2j_vs_h2h")
    h2j_vs_no_stale = _comparison_by_label(comparison_rows, "h2j_vs_no_stale_gate")
    h2j_non_exact = [row for row in non_exact_rows if row["profile_label"] == "h2j_target_query_normalization"]
    full_target_count = sum(
        1
        for row in intervention_rows
        if row["profile_label"] == "h2j_target_query_normalization"
        and row["intervention_kind"] == "visual_target_query_normalization"
    )
    full_stale_count = sum(
        1
        for row in intervention_rows
        if row["profile_label"] == "h2j_target_query_normalization"
        and row["intervention_kind"] == "visual_stale_selection_gate"
    )
    no_stale_target_count = sum(
        1
        for row in intervention_rows
        if row["profile_label"] == "h2j_target_query_normalization_no_stale_gate"
        and row["intervention_kind"] == "visual_target_query_normalization"
    )
    no_stale_stale_count = sum(
        1
        for row in intervention_rows
        if row["profile_label"] == "h2j_target_query_normalization_no_stale_gate"
        and row["intervention_kind"] == "visual_stale_selection_gate"
    )
    return [
        {
            "finding_id": "h2k_is_discriminative",
            "finding": (
                f"H2k separates H2j from the prior candidates: H2e reaches {h2e['exact_success_count']}/8 exact, "
                f"H2h reaches {h2h['exact_success_count']}/8, H2j without stale-selection reaches "
                f"{h2j_no_stale['exact_success_count']}/8, and full H2j reaches {h2j['exact_success_count']}/8."
            ),
        },
        {
            "finding_id": "h2j_passes_target_decoy_overlap",
            "finding": (
                f"H2j improves over H2e by {h2j_vs_h2e['delta_exact_rate']} exact-rate and over H2h by "
                f"{h2j_vs_h2h['delta_exact_rate']} on H2k, ties the no-stale ablation with "
                f"{h2j_vs_no_stale['delta_exact_rate']} exact-rate delta, and has {len(h2j_non_exact)} "
                "H2j non-exact rows."
            ),
        },
        {
            "finding_id": "h2j_mechanism_is_target_normalization",
            "finding": (
                f"Full H2j records {full_target_count} target-query-normalization interventions and "
                f"{full_stale_count} stale-selection interventions on H2k; the stale-gate-off ablation records "
                f"{no_stale_target_count} target-query-normalization interventions and {no_stale_stale_count} "
                "stale-selection interventions, so this holdout isolates target normalization rather than stale rescue."
            ),
        },
        {
            "finding_id": "next_transfer_required",
            "finding": (
                "The next step is not another prompt-profile candidate. Treat H2e as the no-target-normalizer ablation "
                "on H2k, preserve the stale-selection gate globally for stale-origin packets, and build the next "
                "fresh holdout around target-query normalization overreach rather than stale rescue."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2k Target/Decoy Overlap Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2k is a post-H2j holdout that stresses prompts where the true visual target and a decoy share role, "
            "component class, displayed value, or code-label structure. H2j passes the packet at 8/8 while H2e and "
            "H2h remain below it, which supports the target-query normalization mechanism on a fresh overlap gate. "
            "The matched stale-gate-off ablation also passes at 8/8, while H2e remains the no-target-normalizer "
            "control at 3/8 strict exactness, so the current H2k mechanism is target normalization rather than stale rescue."
        ),
        "",
        "![H2k target/decoy overlap gate](figures/h2k_target_decoy_overlap_gate.svg)",
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## Non-Exact Rows",
        "",
        _table(payload["non_exact_rows"]),
        "",
        "## H2j Controller Intervention Rows",
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
    profiles = [
        ("h2e_route_arbitration", "H2e", "#0891B2"),
        ("h2h_component_identity_negative_examples", "H2h", "#155E75"),
        ("h2j_target_query_normalization_no_stale_gate", "H2j-no-stale", "#2563EB"),
        ("h2j_target_query_normalization", "H2j", "#1D4ED8"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 780
    height = 340
    chart_left = 96
    chart_top = 58
    chart_height = 200
    bar_width = 82
    gap = 38
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="780" height="340" viewBox="0 0 780 340" role="img" aria-labelledby="title desc">',
        '<title id="title">H2k target/decoy overlap gate</title>',
        '<desc id="desc">H2j and H2j without stale-selection gate pass H2k while H2e and H2h remain lower.</desc>',
        '<rect width="780" height="340" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2k separates target normalization from prompt repair</text>',
        '<line x1="96" y1="258" x2="660" y2="258" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="90" y1="{y:.1f}" x2="660" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="42" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, (profile, label, color) in enumerate(profiles):
        row = by_profile[profile]
        rate = float(row["exact_rate"])
        bar_height = rate * chart_height
        x = chart_left + index * (bar_width + gap)
        y = chart_top + chart_height - bar_height
        lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 14}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{int(row["exact_success_count"])}/{int(row["case_count"])}</text>'
        )
        lines.append(
            f'<text x="{x + 6}" y="286" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2k target/decoy overlap synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2k_target_decoy_overlap_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
