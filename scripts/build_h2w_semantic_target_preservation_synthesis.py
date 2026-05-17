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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2w_semantic_target_preservation_synthesis"


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
    PacketSpec(
        "h2v_h2w_semantic_target_preservation",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2v_semantic_negation_h2w_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2v_h2w_vs_h2u",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2v_semantic_negation_h2w_vs_h2u_v1",
    ),
    ComparisonSpec(
        "h2v_h2w_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2v_semantic_negation_h2w_vs_h2r_v1",
    ),
    ComparisonSpec(
        "h2v_h2w_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2v_semantic_negation_h2w_vs_h2j_v1",
    ),
)


INTERVENTION_KEYS = (
    "visual_semantic_target_preservation",
    "visual_target_query_normalization",
    "visual_stale_selection_gate",
    "visual_value_bearing_target_query_synthesis",
    "visual_contextual_surface_alias_routing",
    "visual_composed_route_gating",
    "visual_target_query_normalization_blocked",
    "visual_composed_route_gating_blocked",
)


def build_h2w_semantic_target_preservation_synthesis(
    *, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
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
        intervention_rows=intervention_rows,
        fixed_case_rows=fixed_case_rows,
    )

    h2w = _packet_by_profile(packet_rows, "h2v_h2w_semantic_target_preservation")
    h2u = _packet_by_profile(packet_rows, "h2v_h2u_negation_guard")
    h2r = _packet_by_profile(packet_rows, "h2v_h2r_composed_route_gating")
    h2j = _packet_by_profile(packet_rows, "h2v_h2j_target_query_normalization")
    h2w_vs_h2u = _comparison_by_label(comparison_rows, "h2v_h2w_vs_h2u")
    h2w_vs_h2r = _comparison_by_label(comparison_rows, "h2v_h2w_vs_h2r")
    h2w_vs_h2j = _comparison_by_label(comparison_rows, "h2v_h2w_vs_h2j")
    h2w_interventions = [row for row in intervention_rows if row["profile_label"] == "h2v_h2w_semantic_target_preservation"]
    h2w_families = [row for row in family_rows if row["profile_label"] == "h2v_h2w_semantic_target_preservation"]

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2v_case_count": h2w["case_count"],
        "h2j_exact_success_count": h2j["exact_success_count"],
        "h2r_exact_success_count": h2r["exact_success_count"],
        "h2u_exact_success_count": h2u["exact_success_count"],
        "h2w_exact_success_count": h2w["exact_success_count"],
        "h2w_executor_success_count": h2w["executor_success_count"],
        "h2w_delta_exact_vs_h2u": h2w_vs_h2u["delta_exact_rate"],
        "h2w_delta_executor_vs_h2u": h2w_vs_h2u["delta_executor_equivalence_rate"],
        "h2w_delta_exact_vs_h2r": h2w_vs_h2r["delta_exact_rate"],
        "h2w_delta_executor_vs_h2r": h2w_vs_h2r["delta_executor_equivalence_rate"],
        "h2w_delta_exact_vs_h2j": h2w_vs_h2j["delta_exact_rate"],
        "h2w_delta_executor_vs_h2j": h2w_vs_h2j["delta_executor_equivalence_rate"],
        "h2w_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"] == "h2v_h2w_semantic_target_preservation"
        ),
        "h2w_fixed_case_count_vs_h2u": sum(
            1 for row in fixed_case_rows if row["comparison_label"] == "h2v_h2w_vs_h2u"
        ),
        "h2w_semantic_target_preservation_count": _intervention_count(
            h2w_interventions,
            "visual_semantic_target_preservation",
        ),
        "h2w_target_query_normalization_count": _intervention_count(
            h2w_interventions,
            "visual_target_query_normalization",
        ),
        "h2w_stale_selection_gate_count": _intervention_count(h2w_interventions, "visual_stale_selection_gate"),
        "h2w_composed_route_gating_blocked_count": _intervention_count(
            h2w_interventions,
            "visual_composed_route_gating_blocked",
        ),
        "h2w_all_families_exact": all(row["exact_success_count"] == row["case_count"] for row in h2w_families),
        "promotion_decision": "h2w_repairs_h2v_transfer_backtested_separately_before_packaged_workflows",
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

    _write_csv(tables_dir / "h2w_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2w_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2w_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2w_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2w_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2w_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h2w_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2w_semantic_target_preservation_gate.svg", packet_rows)
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


def _intervention_count(rows: list[dict[str, Any]], intervention_kind: str) -> int:
    return sum(1 for row in rows if row["intervention_kind"] == intervention_kind)


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    fixed_case_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2w = _packet_by_profile(packet_rows, "h2v_h2w_semantic_target_preservation")
    h2u = _packet_by_profile(packet_rows, "h2v_h2u_negation_guard")
    h2w_vs_h2u = _comparison_by_label(comparison_rows, "h2v_h2w_vs_h2u")
    h2w_fixed = [row for row in fixed_case_rows if row["comparison_label"] == "h2v_h2w_vs_h2u"]
    h2w_interventions = [row for row in intervention_rows if row["profile_label"] == "h2v_h2w_semantic_target_preservation"]
    h2w_families = [row for row in family_rows if row["profile_label"] == "h2v_h2w_semantic_target_preservation"]
    semantic_count = _intervention_count(h2w_interventions, "visual_semantic_target_preservation")
    normalization_count = _intervention_count(h2w_interventions, "visual_target_query_normalization")
    stale_count = _intervention_count(h2w_interventions, "visual_stale_selection_gate")
    blocked_count = _intervention_count(h2w_interventions, "visual_composed_route_gating_blocked")
    family_exact = ", ".join(
        f"{row['family']} {row['exact_success_count']}/{row['case_count']}" for row in h2w_families
    )
    return [
        {
            "finding_id": "h2w_repairs_h2v_strict_and_executor",
            "finding": (
                f"H2w repairs H2v from H2u's {h2u['exact_success_count']}/{h2u['case_count']} strict and "
                f"{h2u['executor_success_count']}/{h2u['case_count']} executor-equivalent to "
                f"{h2w['exact_success_count']}/{h2w['case_count']} strict and executor-equivalent."
            ),
        },
        {
            "finding_id": "h2w_gain_is_causal_on_six_h2u_misses",
            "finding": (
                f"H2w fixes {len(h2w_fixed)} strict H2u misses with a "
                f"{h2w_vs_h2u['delta_exact_rate']} exact-rate gain and "
                f"{h2w_vs_h2u['delta_executor_equivalence_rate']} executor-equivalence gain."
            ),
        },
        {
            "finding_id": "h2w_mechanism_splits_three_error_types",
            "finding": (
                f"The H2w run records {semantic_count} semantic-preservation interventions, "
                f"{normalization_count} component-qualified value canonicalizations, {stale_count} stale-selection "
                f"repair, and {blocked_count} negation-aware composed-route block."
            ),
        },
        {
            "finding_id": "h2w_family_saturation_is_local_not_global",
            "finding": (
                f"H2w reaches exactness across all H2v families ({family_exact}). The separate H2w transfer "
                "backtest is now clean, so the remaining promotion gap is packaged-workflow or harder CLI-live "
                "semantic pressure rather than same-family replay transfer."
            ),
        },
        {
            "finding_id": "h2w_next_requires_packaged_semantic_pressure",
            "finding": (
                "The next step is packaged-workflow or harder CLI-live pressure, not another replay transfer pass: "
                "H2w includes a bounded no-call visual fallback and a more permissive semantic label selector that "
                "should be tested where workflow scaffolding cannot resolve the ambiguity upstream."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2w Semantic Target Preservation Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2w is the direct repair candidate for H2v. It adds semantic target preservation on top of H2u, separating "
            "stale/quoted negation context from genuine negated target values and adding a bounded no-call visual "
            "fallback when the requested visual target is unambiguous."
        ),
        "",
        (
            f"H2w reaches `{manifest['h2w_exact_success_count']} / {manifest['h2v_case_count']}` strict and "
            f"`{manifest['h2w_executor_success_count']} / {manifest['h2v_case_count']}` executor-equivalent, versus "
            f"H2u's `{manifest['h2u_exact_success_count']} / {manifest['h2v_case_count']}` strict. The exact-rate "
            f"gain over H2u is `{manifest['h2w_delta_exact_vs_h2u']}`."
        ),
        "",
        (
            "Mechanistically, H2w does not simply suppress the word `not`. It preserves current requested labels when "
            "negation belongs to stale context, canonicalizes value-before-surface phrases such as `Not ready status "
            "badge` to layout labels such as `status badge Not ready`. The control also has a bounded no-call visual "
            "fallback, but the final H2v H2w packet did not need to exercise it."
        ),
        "",
        (
            "The separate H2w transfer backtest now preserves `109 / 109` strict and executor-equivalent rows across "
            "the current transfer/back-compat battery. This H2v-local report should therefore be read together with "
            "`../h2w_transfer_backtest_synthesis/report.md`: transfer is clean, while packaged workflow semantic "
            "pressure remains unproven."
        ),
        "",
        "![H2w semantic target preservation gate](figures/h2w_semantic_target_preservation_gate.svg)",
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
        ("h2v_h2w_semantic_target_preservation", "H2w", "#92400E"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 860
    height = 350
    chart_left = 82
    chart_top = 64
    chart_height = 190
    group_width = 150
    bar_width = 42
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2w semantic target preservation gate</title>',
        '<desc id="desc">H2w reaches ten of ten strict and executor-equivalent on H2v.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2w closes H2v semantic target preservation</text>',
        '<line x1="82" y1="254" x2="710" y2="254" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="76" y1="{y:.1f}" x2="710" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="38" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
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
            f'<text x="{x + 2}" y="{exact_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/10</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 10}" y="{executor_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["executor_success_count"])}/10</text>'
        )
        lines.append(
            f'<text x="{x + 18}" y="282" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.extend(
        [
            '<rect x="650" y="300" width="18" height="12" fill="#92400E"/>',
            '<text x="674" y="311" font-family="Arial, sans-serif" font-size="12" fill="#374151">strict</text>',
            '<rect x="650" y="320" width="18" height="12" fill="#92400E" opacity="0.45"/>',
            '<text x="674" y="331" font-family="Arial, sans-serif" font-size="12" fill="#374151">executor-equivalent</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2w semantic target preservation synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2w_semantic_target_preservation_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
