from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from build_h2l_target_normalization_overreach_synthesis import (
    ComparisonSpec,
    PacketSpec,
    _compact_json,
    _comparison_by_label,
    _comparison_row,
    _non_exact_rows,
    _packet_by_profile,
    _packet_row,
    _read_json,
    _table,
    _write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2v_semantic_negation_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2v_h2j_target_query_normalization",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2v_semantic_negation_h2j_execute_v1",
    ),
    PacketSpec(
        "h2v_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2v_semantic_negation_h2r_execute_v1",
    ),
    PacketSpec(
        "h2v_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2v_semantic_negation_h2u_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2v_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2v_semantic_negation_h2u_vs_h2r_v1",
    ),
    ComparisonSpec(
        "h2v_h2u_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2v_semantic_negation_h2u_vs_h2j_v1",
    ),
    ComparisonSpec(
        "h2v_h2r_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2v_semantic_negation_h2r_vs_h2j_v1",
    ),
)


INTERVENTION_KEYS = (
    "visual_target_query_normalization",
    "visual_stale_selection_gate",
    "visual_value_bearing_target_query_synthesis",
    "visual_contextual_surface_alias_routing",
    "visual_composed_route_gating",
    "visual_target_query_normalization_blocked",
    "visual_composed_route_gating_blocked",
)


def build_h2v_semantic_negation_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    family_rows = _family_rows(PACKET_SPECS)
    non_exact_rows = _non_exact_rows(PACKET_SPECS)
    intervention_rows = _controller_intervention_rows(PACKET_SPECS)
    fixed_case_rows = _fixed_case_rows(COMPARISON_SPECS)
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        family_rows=family_rows,
        non_exact_rows=non_exact_rows,
        fixed_case_rows=fixed_case_rows,
    )

    h2j = _packet_by_profile(packet_rows, "h2v_h2j_target_query_normalization")
    h2r = _packet_by_profile(packet_rows, "h2v_h2r_composed_route_gating")
    h2u = _packet_by_profile(packet_rows, "h2v_h2u_negation_guard")
    h2u_vs_h2r = _comparison_by_label(comparison_rows, "h2v_h2u_vs_h2r")
    h2u_vs_h2j = _comparison_by_label(comparison_rows, "h2v_h2u_vs_h2j")
    h2r_vs_h2j = _comparison_by_label(comparison_rows, "h2v_h2r_vs_h2j")
    h2u_family = [row for row in family_rows if row["profile_label"] == "h2v_h2u_negation_guard"]
    genuine_family = _family_by_name(h2u_family, "h2v_genuine_negated_target")
    stale_family = _family_by_name(h2u_family, "h2v_stale_example_negation_context")
    quoted_family = _family_by_name(h2u_family, "h2v_quoted_negation_context")
    instructional_family = _family_by_name(h2u_family, "h2v_instructional_negation_context")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2v_case_count": h2u["case_count"],
        "h2j_exact_success_count": h2j["exact_success_count"],
        "h2j_executor_success_count": h2j["executor_success_count"],
        "h2r_exact_success_count": h2r["exact_success_count"],
        "h2r_executor_success_count": h2r["executor_success_count"],
        "h2u_exact_success_count": h2u["exact_success_count"],
        "h2u_executor_success_count": h2u["executor_success_count"],
        "h2u_delta_exact_vs_h2r": h2u_vs_h2r["delta_exact_rate"],
        "h2u_delta_executor_vs_h2r": h2u_vs_h2r["delta_executor_equivalence_rate"],
        "h2u_delta_exact_vs_h2j": h2u_vs_h2j["delta_exact_rate"],
        "h2u_delta_executor_vs_h2j": h2u_vs_h2j["delta_executor_equivalence_rate"],
        "h2r_delta_exact_vs_h2j": h2r_vs_h2j["delta_exact_rate"],
        "h2r_delta_executor_vs_h2j": h2r_vs_h2j["delta_executor_equivalence_rate"],
        "h2u_non_exact_count": sum(1 for row in non_exact_rows if row["profile_label"] == "h2v_h2u_negation_guard"),
        "h2u_executor_non_equivalent_count": int(h2u["case_count"]) - int(h2u["executor_success_count"]),
        "h2u_quoted_exact_success_count": quoted_family["exact_success_count"],
        "h2u_instructional_exact_success_count": instructional_family["exact_success_count"],
        "h2u_stale_example_exact_success_count": stale_family["exact_success_count"],
        "h2u_genuine_negated_exact_success_count": genuine_family["exact_success_count"],
        "h2u_genuine_negated_executor_success_count": genuine_family["executor_success_count"],
        "h2u_fixed_case_count_vs_h2r": sum(
            1 for row in fixed_case_rows if row["comparison_label"] == "h2v_h2u_vs_h2r"
        ),
        "promotion_decision": "h2u_not_promoted_until_h2w_semantic_target_preservation",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "family_rows": family_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "fixed_case_rows": fixed_case_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2v_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2v_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2v_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2v_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2v_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2v_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h2v_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2v_semantic_negation_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _family_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        families: dict[str, dict[str, int]] = {}
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            family = str(result.get("family", ""))
            bucket = families.setdefault(
                family,
                {
                    "case_count": 0,
                    "exact_success_count": 0,
                    "executor_success_count": 0,
                },
            )
            bucket["case_count"] += 1
            bucket["exact_success_count"] += int(result.get("replay_exact_match") is True)
            bucket["executor_success_count"] += int(result.get("replay_executor_equivalence_match") is True)
        for family, counts in sorted(families.items()):
            case_count = counts["case_count"]
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "family": family,
                    "case_count": case_count,
                    "exact_success_count": counts["exact_success_count"],
                    "exact_rate": counts["exact_success_count"] / case_count if case_count else 0.0,
                    "executor_success_count": counts["executor_success_count"],
                    "executor_rate": counts["executor_success_count"] / case_count if case_count else 0.0,
                }
            )
    return rows


def _controller_intervention_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for kind in INTERVENTION_KEYS:
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
                            "preserved_target_query": entry.get("preserved_target_query", ""),
                            "blocked_label": entry.get("blocked_label", entry.get("prompt_state_label", "")),
                            "prompt_state_label": entry.get("prompt_state_label", ""),
                            "reason": entry.get("reason", ""),
                        }
                    )
    return rows


def _fixed_case_rows(specs: tuple[ComparisonSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
        for row in payload["case_deltas"]:
            if row.get("baseline_replay_exact_match") is not False or row.get("candidate_replay_exact_match") is not True:
                continue
            rows.append(
                {
                    "comparison_label": spec.comparison_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "baseline_failure_mode": row.get("baseline_replay_failure_mode", ""),
                    "candidate_failure_mode": row.get("candidate_replay_failure_mode", ""),
                    "baseline_executor_equivalence_match": row.get("baseline_replay_executor_equivalence_match"),
                    "candidate_executor_equivalence_match": row.get("candidate_replay_executor_equivalence_match"),
                }
            )
    return rows


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    fixed_case_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2j = _packet_by_profile(packet_rows, "h2v_h2j_target_query_normalization")
    h2r = _packet_by_profile(packet_rows, "h2v_h2r_composed_route_gating")
    h2u = _packet_by_profile(packet_rows, "h2v_h2u_negation_guard")
    h2u_vs_h2r = _comparison_by_label(comparison_rows, "h2v_h2u_vs_h2r")
    h2u_vs_h2j = _comparison_by_label(comparison_rows, "h2v_h2u_vs_h2j")
    h2r_vs_h2j = _comparison_by_label(comparison_rows, "h2v_h2r_vs_h2j")
    h2u_family = [row for row in family_rows if row["profile_label"] == "h2v_h2u_negation_guard"]
    quoted = _family_by_name(h2u_family, "h2v_quoted_negation_context")
    instructional = _family_by_name(h2u_family, "h2v_instructional_negation_context")
    stale = _family_by_name(h2u_family, "h2v_stale_example_negation_context")
    genuine = _family_by_name(h2u_family, "h2v_genuine_negated_target")
    h2u_non_exact = [row for row in non_exact_rows if row["profile_label"] == "h2v_h2u_negation_guard"]
    h2u_fixed = [row for row in fixed_case_rows if row["comparison_label"] in {"h2v_h2u_vs_h2r", "h2v_h2u_vs_h2j"}]
    return [
        {
            "finding_id": "h2v_breaks_h2u_transfer_saturation",
            "finding": (
                f"H2v breaks the prior H2u same-family transfer saturation: H2u reaches "
                f"{h2u['exact_success_count']}/{h2u['case_count']} strict and "
                f"{h2u['executor_success_count']}/{h2u['case_count']} executor-equivalent after H2u had preserved "
                "99/99 strict/executor-equivalent on the earlier transfer set."
            ),
        },
        {
            "finding_id": "h2u_negation_guard_help_is_real_but_small",
            "finding": (
                f"H2u improves over H2r by {_format_rate(h2u_vs_h2r['delta_exact_rate'])} strict and "
                f"{_format_rate(h2u_vs_h2r['delta_executor_equivalence_rate'])} executor-equivalence rate, and over "
                f"H2j by {_format_rate(h2u_vs_h2j['delta_exact_rate'])} strict and "
                f"{_format_rate(h2u_vs_h2j['delta_executor_equivalence_rate'])} executor-equivalence rate."
            ),
        },
        {
            "finding_id": "h2r_and_h2j_tie_on_h2v",
            "finding": (
                f"H2r ties H2j on H2v at {h2r['exact_success_count']}/{h2r['case_count']} strict and "
                f"{h2r['executor_success_count']}/{h2r['case_count']} executor-equivalent, with "
                f"{_format_rate(h2r_vs_h2j['delta_exact_rate'])} exact-rate delta. Composed route gating alone does "
                "not solve this semantic negation split."
            ),
        },
        {
            "finding_id": "h2v_family_split_identifies_next_repair",
            "finding": (
                f"H2u solves instructional negation at {instructional['exact_success_count']}/"
                f"{instructional['case_count']} and the clean control, but reaches only "
                f"{quoted['exact_success_count']}/{quoted['case_count']} on quoted negation, "
                f"{stale['exact_success_count']}/{stale['case_count']} on stale examples, and "
                f"{genuine['exact_success_count']}/{genuine['case_count']} strict plus "
                f"{genuine['executor_success_count']}/{genuine['case_count']} executor-equivalent on genuine negated "
                "targets."
            ),
        },
        {
            "finding_id": "h2u_fixed_case_is_one_quoted_context_row",
            "finding": (
                "The only strict H2u gain over both H2r and H2j is "
                f"{', '.join(sorted({row['case_id'] for row in h2u_fixed}))}; the remaining "
                f"{len(h2u_non_exact)} H2u non-exact rows are not repaired by the current negation guard."
            ),
        },
        {
            "finding_id": "next_h2w_should_preserve_semantic_targets",
            "finding": (
                "The next candidate should distinguish negated context that must be ignored from negated values that "
                "are themselves the target, while also treating stale examples as old context even when the word not "
                "appears near a tempting label."
            ),
        },
    ]


def _family_by_name(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    for row in rows:
        if row["family"] == family:
            return row
    raise KeyError(family)


def _format_rate(value: Any) -> str:
    return f"{float(value):.2f}"


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2v Semantic Negation Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2v is the first fresh semantic-negation holdout after H2u closed the same-family transfer backtest. "
            "It separates quoted negation, instructional negation, stale example captions, genuine negated targets, "
            "and a clean control. The result breaks the apparent H2u saturation: H2u is better than H2r/H2j, but "
            "only by one case."
        ),
        "",
        (
            f"H2j and H2r each reach `3 / {manifest['h2v_case_count']}` strict and "
            f"`4 / {manifest['h2v_case_count']}` executor-equivalent. H2u reaches "
            f"`{manifest['h2u_exact_success_count']} / {manifest['h2v_case_count']}` strict and "
            f"`{manifest['h2u_executor_success_count']} / {manifest['h2v_case_count']}` executor-equivalent, a "
            f"`{_format_rate(manifest['h2u_delta_exact_vs_h2r'])}` exact-rate improvement versus H2r."
        ),
        "",
        (
            "The failure split matters more than the top-line: H2u solves the two instructional-negation rows and "
            "the clean control, fixes one quoted-negation row, but fails both stale-example rows and all three "
            "genuine-negated-target rows under strict exactness."
        ),
        "",
        "![H2v semantic negation gate](figures/h2v_semantic_negation_gate.svg)",
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## Family Rows",
        "",
        _table(payload["family_rows"]),
        "",
        "## Non-Exact Rows",
        "",
        _table(payload["non_exact_rows"]),
        "",
        "## Controller Intervention Rows",
        "",
        _table(payload["intervention_rows"]),
        "",
        "## Fixed Case Rows",
        "",
        _table(payload["fixed_case_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    profiles = [
        ("h2v_h2j_target_query_normalization", "H2j", "#0F766E"),
        ("h2v_h2r_composed_route_gating", "H2r", "#2563EB"),
        ("h2v_h2u_negation_guard", "H2u", "#7C3AED"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 760
    height = 350
    chart_left = 86
    chart_top = 64
    chart_height = 190
    group_width = 150
    bar_width = 44
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2v semantic negation gate</title>',
        '<desc id="desc">H2u improves over H2r and H2j by one strict and executor-equivalent case on H2v.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2v breaks H2u semantic saturation</text>',
        '<line x1="86" y1="254" x2="620" y2="254" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="80" y1="{y:.1f}" x2="620" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="42" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, (profile, label, color) in enumerate(profiles):
        row = by_profile[profile]
        x = chart_left + index * group_width
        exact_height = float(row["exact_rate"]) * chart_height
        executor_height = float(row["executor_rate"]) * chart_height
        exact_y = chart_top + chart_height - exact_height
        executor_y = chart_top + chart_height - executor_height
        lines.append(f'<rect x="{x}" y="{exact_y:.1f}" width="{bar_width}" height="{exact_height:.1f}" fill="{color}"/>')
        lines.append(
            f'<rect x="{x + bar_width + 8}" y="{executor_y:.1f}" width="{bar_width}" height="{executor_height:.1f}" fill="{color}" opacity="0.45"/>'
        )
        lines.append(
            f'<text x="{x + 4}" y="{exact_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/10</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 12}" y="{executor_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["executor_success_count"])}/10</text>'
        )
        lines.append(
            f'<text x="{x + 20}" y="282" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.extend(
        [
            '<rect x="550" y="300" width="18" height="12" fill="#7C3AED"/>',
            '<text x="574" y="311" font-family="Arial, sans-serif" font-size="12" fill="#374151">strict</text>',
            '<rect x="550" y="320" width="18" height="12" fill="#7C3AED" opacity="0.45"/>',
            '<text x="574" y="331" font-family="Arial, sans-serif" font-size="12" fill="#374151">executor-equivalent</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2v semantic-negation synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2v_semantic_negation_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
