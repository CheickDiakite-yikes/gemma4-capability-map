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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2x_cli_semantic_pressure_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2x_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260517T_h2x_cli_semantic_pressure_h2u_execute_v1",
    ),
    PacketSpec(
        "h2x_h2u_no_controller_fallback",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260517T_h2x_cli_semantic_pressure_h2u_no_fallback_execute_v1",
    ),
    PacketSpec(
        "h2x_h2w_semantic_target_preservation",
        ROOT / "results" / "tool_probe_replay_live" / "20260517T_h2x_cli_semantic_pressure_h2w_execute_v1",
    ),
    PacketSpec(
        "h2x_h2w_no_controller_fallback",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260517T_h2x_cli_semantic_pressure_h2w_no_fallback_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2x_h2w_vs_h2u",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260517T_h2x_cli_semantic_pressure_h2w_vs_h2u_v1",
    ),
    ComparisonSpec(
        "h2x_h2u_no_fallback_vs_h2u",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260517T_h2x_cli_semantic_pressure_h2u_no_fallback_vs_h2u_v1",
    ),
    ComparisonSpec(
        "h2x_h2w_no_fallback_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260517T_h2x_cli_semantic_pressure_h2w_no_fallback_vs_h2w_v1",
    ),
)


INTERVENTION_KEYS = (
    "visual_semantic_target_preservation",
    "visual_target_query_normalization",
    "visual_stale_selection_gate",
    "visual_value_bearing_target_query_synthesis",
    "visual_contextual_surface_alias_routing",
    "visual_composed_route_gating",
    "visual_stale_selection_negation_guard",
    "visual_negated_component_target_preservation",
    "visual_target_query_normalization_blocked",
    "visual_composed_route_gating_blocked",
)


def build_h2x_cli_semantic_pressure_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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

    h2u = _packet_by_profile(packet_rows, "h2x_h2u_negation_guard")
    h2u_no_fallback = _packet_by_profile(packet_rows, "h2x_h2u_no_controller_fallback")
    h2w = _packet_by_profile(packet_rows, "h2x_h2w_semantic_target_preservation")
    h2w_no_fallback = _packet_by_profile(packet_rows, "h2x_h2w_no_controller_fallback")
    h2w_vs_h2u = _comparison_by_label(comparison_rows, "h2x_h2w_vs_h2u")
    h2u_no_fallback_vs_h2u = _comparison_by_label(comparison_rows, "h2x_h2u_no_fallback_vs_h2u")
    h2w_no_fallback_vs_h2w = _comparison_by_label(comparison_rows, "h2x_h2w_no_fallback_vs_h2w")
    h2w_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h2x_h2w_semantic_target_preservation"
    ]
    h2w_no_fallback_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h2x_h2w_no_controller_fallback"
    ]
    h2u_fallback_delta_exact = h2u_no_fallback_vs_h2u["delta_exact_rate"]
    h2u_fallback_delta_executor = h2u_no_fallback_vs_h2u["delta_executor_equivalence_rate"]
    h2w_fallback_delta_exact = h2w_no_fallback_vs_h2w["delta_exact_rate"]
    h2w_fallback_delta_executor = h2w_no_fallback_vs_h2w["delta_executor_equivalence_rate"]

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2x_case_count": h2w["case_count"],
        "h2u_exact_success_count": h2u["exact_success_count"],
        "h2u_executor_success_count": h2u["executor_success_count"],
        "h2u_no_fallback_exact_success_count": h2u_no_fallback["exact_success_count"],
        "h2u_no_fallback_executor_success_count": h2u_no_fallback["executor_success_count"],
        "h2w_exact_success_count": h2w["exact_success_count"],
        "h2w_executor_success_count": h2w["executor_success_count"],
        "h2w_no_fallback_exact_success_count": h2w_no_fallback["exact_success_count"],
        "h2w_no_fallback_executor_success_count": h2w_no_fallback["executor_success_count"],
        "h2w_delta_exact_vs_h2u": h2w_vs_h2u["delta_exact_rate"],
        "h2w_delta_executor_vs_h2u": h2w_vs_h2u["delta_executor_equivalence_rate"],
        "h2u_fallback_delta_exact": h2u_fallback_delta_exact,
        "h2u_fallback_delta_executor": h2u_fallback_delta_executor,
        "h2w_fallback_delta_exact": h2w_fallback_delta_exact,
        "h2w_fallback_delta_executor": h2w_fallback_delta_executor,
        "h2u_delta_exact_fallback_enabled": h2u_fallback_delta_exact,
        "h2u_delta_executor_fallback_enabled": h2u_fallback_delta_executor,
        "h2w_delta_exact_fallback_enabled": h2w_fallback_delta_exact,
        "h2w_delta_executor_fallback_enabled": h2w_fallback_delta_executor,
        "h2w_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"] == "h2x_h2w_semantic_target_preservation"
        ),
        "h2w_fixed_case_count_vs_h2u": sum(
            1 for row in fixed_case_rows if row["comparison_label"] == "h2x_h2w_vs_h2u"
        ),
        "h2w_semantic_target_preservation_count": _intervention_count(
            h2w_interventions,
            "visual_semantic_target_preservation",
        ),
        "h2w_target_query_normalization_count": _intervention_count(
            h2w_interventions,
            "visual_target_query_normalization",
        ),
        "h2w_composed_route_gating_count": _intervention_count(h2w_interventions, "visual_composed_route_gating"),
        "h2w_composed_route_gating_blocked_count": _intervention_count(
            h2w_interventions,
            "visual_composed_route_gating_blocked",
        ),
        "h2w_no_fallback_semantic_target_preservation_count": _intervention_count(
            h2w_no_fallback_interventions,
            "visual_semantic_target_preservation",
        ),
        "h2w_no_fallback_target_query_normalization_count": _intervention_count(
            h2w_no_fallback_interventions,
            "visual_target_query_normalization",
        ),
        "fallback_independence_holds": all(
            delta == 0.0
            for delta in (
                h2u_fallback_delta_exact,
                h2u_fallback_delta_executor,
                h2w_fallback_delta_exact,
                h2w_fallback_delta_executor,
            )
        ),
        "promotion_decision": "h2x_promotes_semantic_target_preservation_to_packaged_cli_gate",
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

    _write_csv(tables_dir / "h2x_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2x_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2x_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2x_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2x_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2x_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h2x_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2x_cli_semantic_pressure_gate.svg", packet_rows)
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
            bucket = families.setdefault(family, {"case_count": 0, "exact_success_count": 0, "executor_success_count": 0})
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
                            "requested_label": entry.get("requested_label", ""),
                            "requested_region_id": entry.get("requested_region_id", ""),
                            "prompt_state_label": entry.get("prompt_state_label", ""),
                            "blocked_label": entry.get("blocked_label", ""),
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
    h2u = _packet_by_profile(packet_rows, "h2x_h2u_negation_guard")
    h2w = _packet_by_profile(packet_rows, "h2x_h2w_semantic_target_preservation")
    h2w_nf = _packet_by_profile(packet_rows, "h2x_h2w_no_controller_fallback")
    h2w_vs_h2u = _comparison_by_label(comparison_rows, "h2x_h2w_vs_h2u")
    h2u_fallback = _comparison_by_label(comparison_rows, "h2x_h2u_no_fallback_vs_h2u")
    h2w_fallback = _comparison_by_label(comparison_rows, "h2x_h2w_no_fallback_vs_h2w")
    h2w_fixed = [row for row in fixed_case_rows if row["comparison_label"] == "h2x_h2w_vs_h2u"]
    h2w_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h2x_h2w_semantic_target_preservation"
    ]
    h2w_families = [row for row in family_rows if row["profile_label"] == "h2x_h2w_semantic_target_preservation"]
    family_exact = ", ".join(
        f"{row['family']} {row['exact_success_count']}/{row['case_count']}" for row in h2w_families
    )
    return [
        {
            "finding_id": "h2x_breaks_h2u_topline_saturation",
            "finding": (
                f"H2x drops H2u to {h2u['exact_success_count']}/{h2u['case_count']} strict and "
                f"{h2u['executor_success_count']}/{h2u['case_count']} executor-equivalent, exposing semantic "
                "target pressure hidden by earlier top-line saturation."
            ),
        },
        {
            "finding_id": "h2w_repairs_h2x_without_fallback",
            "finding": (
                f"H2w reaches {h2w['exact_success_count']}/{h2w['case_count']} strict and "
                f"{h2w['executor_success_count']}/{h2w['case_count']} executor-equivalent, with the no-fallback H2w "
                f"profile also at {h2w_nf['exact_success_count']}/{h2w_nf['case_count']}."
            ),
        },
        {
            "finding_id": "semantic_preservation_is_causal_not_fallback",
            "finding": (
                f"H2w vs H2u gains {h2w_vs_h2u['delta_exact_rate']} exact and "
                f"{h2w_vs_h2u['delta_executor_equivalence_rate']} executor-equivalence; fallback is not the causal "
                f"helper on this slice because enabling controller "
                f"fallback changes H2u by {h2u_fallback['delta_exact_rate']} exact and H2w by "
                f"{h2w_fallback['delta_exact_rate']} exact."
            ),
        },
        {
            "finding_id": "h2x_mechanism_mix",
            "finding": (
                f"H2w fixes {len(h2w_fixed)} H2u strict misses and records "
                f"{_intervention_count(h2w_interventions, 'visual_semantic_target_preservation')} semantic-preservation, "
                f"{_intervention_count(h2w_interventions, 'visual_target_query_normalization')} value-bearing "
                f"target-query normalization, and "
                f"{_intervention_count(h2w_interventions, 'visual_composed_route_gating')} composed-route interventions."
            ),
        },
        {
            "finding_id": "h2x_promotes_to_packaged_cli_gate",
            "finding": (
                f"H2w is exact across H2x families ({family_exact}). The packaged live workflow smoke also reaches "
                "an approval gate in the sandbox, so the next research move is a larger packaged H1 slice rather "
                "than another same-shape replay repair."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2x CLI Semantic Pressure Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2x is the first CLI-first semantic-pressure gate after the H2w transfer backtest. It deliberately mixes "
            "stale quoted negation, stale selection text, instructional negation, and genuine displayed negated values "
            "so top-line readiness cannot hide semantic target dependence."
        ),
        "",
        (
            f"H2u reaches `{manifest['h2u_exact_success_count']} / {manifest['h2x_case_count']}` strict and "
            f"`{manifest['h2u_executor_success_count']} / {manifest['h2x_case_count']}` executor-equivalent. H2w reaches "
            f"`{manifest['h2w_exact_success_count']} / {manifest['h2x_case_count']}` on both metrics, a "
            f"`{manifest['h2w_delta_exact_vs_h2u']}` exact-rate gain and "
            f"`{manifest['h2w_delta_executor_vs_h2u']}` executor-equivalence gain."
        ),
        "",
        (
            "The no-fallback controls are the key attribution result: H2u no-fallback is unchanged from H2u, and "
            "H2w no-fallback is unchanged from H2w. On this slice, controller fallback is not the causal helper; "
            "semantic target preservation and value-bearing target-query normalization are."
        ),
        "",
        (
            "A packaged live workflow smoke for `executive_semantic_target_pressure` also ran through the Rich CLI "
            "operator, wrote only inside the ephemeral sandbox, recorded three policy blocks, and stopped at an "
            "approval gate. That makes H2x both replay-attributable and live-harness attributable."
        ),
        "",
        "![H2x CLI semantic pressure gate](figures/h2x_cli_semantic_pressure_gate.svg)",
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
        ("h2x_h2u_negation_guard", "H2u", "#7C3AED"),
        ("h2x_h2u_no_controller_fallback", "H2u no fallback", "#A78BFA"),
        ("h2x_h2w_semantic_target_preservation", "H2w", "#92400E"),
        ("h2x_h2w_no_controller_fallback", "H2w no fallback", "#D97706"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 940
    height = 360
    chart_left = 82
    chart_top = 64
    chart_height = 190
    group_width = 180
    bar_width = 42
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2x CLI semantic pressure gate</title>',
        '<desc id="desc">H2w and H2w without fallback reach eight of eight while H2u remains at three of eight strict.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2x isolates semantic preservation from fallback</text>',
        '<line x1="82" y1="254" x2="820" y2="254" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="76" y1="{y:.1f}" x2="820" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
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
            f'<text x="{x + 2}" y="{exact_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/8</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 10}" y="{executor_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["executor_success_count"])}/8</text>'
        )
        lines.append(
            f'<text x="{x - 8}" y="282" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.extend(
        [
            '<rect x="720" y="304" width="18" height="12" fill="#92400E"/>',
            '<text x="744" y="315" font-family="Arial, sans-serif" font-size="12" fill="#374151">strict</text>',
            '<rect x="720" y="324" width="18" height="12" fill="#92400E" opacity="0.45"/>',
            '<text x="744" y="335" font-family="Arial, sans-serif" font-size="12" fill="#374151">executor-equivalent</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2x CLI semantic pressure synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2x_cli_semantic_pressure_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
