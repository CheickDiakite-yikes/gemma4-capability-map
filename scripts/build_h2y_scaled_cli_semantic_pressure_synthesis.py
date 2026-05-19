from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from build_h2l_target_normalization_overreach_synthesis import (
    ComparisonSpec,
    PacketSpec,
    _comparison_by_label,
    _comparison_row,
    _non_exact_rows,
    _packet_by_profile,
    _packet_row,
    _read_json,
    _table,
    _write_csv,
)
from build_h2x_cli_semantic_pressure_synthesis import (
    _controller_intervention_rows,
    _family_rows,
    _fixed_case_rows,
    _intervention_count,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2y_scaled_cli_semantic_pressure_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2y_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h2y_scaled_cli_semantic_pressure_h2u_execute_v1",
    ),
    PacketSpec(
        "h2y_h2u_no_controller_fallback",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2u_no_fallback_execute_v1",
    ),
    PacketSpec(
        "h2y_h2w_semantic_target_preservation",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h2y_scaled_cli_semantic_pressure_h2w_execute_v1",
    ),
    PacketSpec(
        "h2y_h2w_no_controller_fallback",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2w_no_fallback_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2y_h2w_vs_h2u",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2w_vs_h2u_v1",
    ),
    ComparisonSpec(
        "h2y_h2u_no_fallback_vs_h2u",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2u_no_fallback_vs_h2u_v1",
    ),
    ComparisonSpec(
        "h2y_h2w_no_fallback_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2w_no_fallback_vs_h2w_v1",
    ),
)


def build_h2y_scaled_cli_semantic_pressure_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    unresolved_boundary_rows = _unresolved_boundary_rows(PACKET_SPECS)
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        family_rows=family_rows,
        intervention_rows=intervention_rows,
        fixed_case_rows=fixed_case_rows,
        unresolved_boundary_rows=unresolved_boundary_rows,
    )

    h2u = _packet_by_profile(packet_rows, "h2y_h2u_negation_guard")
    h2u_no_fallback = _packet_by_profile(packet_rows, "h2y_h2u_no_controller_fallback")
    h2w = _packet_by_profile(packet_rows, "h2y_h2w_semantic_target_preservation")
    h2w_no_fallback = _packet_by_profile(packet_rows, "h2y_h2w_no_controller_fallback")
    h2w_vs_h2u = _comparison_by_label(comparison_rows, "h2y_h2w_vs_h2u")
    h2u_no_fallback_vs_h2u = _comparison_by_label(comparison_rows, "h2y_h2u_no_fallback_vs_h2u")
    h2w_no_fallback_vs_h2w = _comparison_by_label(comparison_rows, "h2y_h2w_no_fallback_vs_h2w")
    h2w_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h2y_h2w_semantic_target_preservation"
    ]
    h2w_no_fallback_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h2y_h2w_no_controller_fallback"
    ]

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2y_case_count": h2w["case_count"],
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
        "h2u_fallback_delta_exact": h2u_no_fallback_vs_h2u["delta_exact_rate"],
        "h2u_fallback_delta_executor": h2u_no_fallback_vs_h2u["delta_executor_equivalence_rate"],
        "h2w_fallback_delta_exact": h2w_no_fallback_vs_h2w["delta_exact_rate"],
        "h2w_fallback_delta_executor": h2w_no_fallback_vs_h2w["delta_executor_equivalence_rate"],
        "h2w_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"] == "h2y_h2w_semantic_target_preservation"
        ),
        "h2w_fixed_case_count_vs_h2u": sum(
            1 for row in fixed_case_rows if row["comparison_label"] == "h2y_h2w_vs_h2u"
        ),
        "h2w_semantic_target_preservation_count": _intervention_count(
            h2w_interventions,
            "visual_semantic_target_preservation",
        ),
        "h2w_target_query_normalization_count": _intervention_count(
            h2w_interventions,
            "visual_target_query_normalization",
        ),
        "h2w_no_fallback_semantic_target_preservation_count": _intervention_count(
            h2w_no_fallback_interventions,
            "visual_semantic_target_preservation",
        ),
        "h2w_no_fallback_target_query_normalization_count": _intervention_count(
            h2w_no_fallback_interventions,
            "visual_target_query_normalization",
        ),
        "h2w_unresolved_boundary_count": sum(
            1
            for row in unresolved_boundary_rows
            if row["profile_label"] == "h2y_h2w_semantic_target_preservation"
        ),
        "h2w_unresolved_stale_selection_count": sum(
            1
            for row in unresolved_boundary_rows
            if row["profile_label"] == "h2y_h2w_semantic_target_preservation"
            and row["family"] == "h2y_stale_selection_negation_context"
        ),
        "fallback_independence_holds": all(
            delta == 0.0
            for delta in (
                h2u_no_fallback_vs_h2u["delta_exact_rate"],
                h2u_no_fallback_vs_h2u["delta_executor_equivalence_rate"],
                h2w_no_fallback_vs_h2w["delta_exact_rate"],
                h2w_no_fallback_vs_h2w["delta_executor_equivalence_rate"],
            )
        ),
        "promotion_decision": "h2y_confirms_semantic_preservation_gain_but_blocks_global_promotion",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "family_rows": family_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "fixed_case_rows": fixed_case_rows,
        "unresolved_boundary_rows": unresolved_boundary_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2y_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2y_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2y_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2y_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2y_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2y_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h2y_unresolved_boundary_rows.csv", unresolved_boundary_rows)
    _write_csv(tables_dir / "h2y_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2y_scaled_cli_semantic_pressure_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _unresolved_boundary_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_profiles = {"h2y_h2w_semantic_target_preservation", "h2y_h2w_no_controller_fallback"}
    for spec in specs:
        if spec.profile_label not in target_profiles:
            continue
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            if result.get("replay_exact_match") is True:
                continue
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            expected_calls = probe.get("expected_calls", [])
            actual_calls = probe.get("actual_calls", [])
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "case_id": result["case_id"],
                    "family": result.get("family", ""),
                    "failure_mode": result.get("replay_failure_mode", ""),
                    "expected_calls": json.dumps(expected_calls, sort_keys=True, ensure_ascii=False),
                    "actual_calls": json.dumps(actual_calls, sort_keys=True, ensure_ascii=False),
                }
            )
    return rows


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    fixed_case_rows: list[dict[str, Any]],
    unresolved_boundary_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2u = _packet_by_profile(packet_rows, "h2y_h2u_negation_guard")
    h2w = _packet_by_profile(packet_rows, "h2y_h2w_semantic_target_preservation")
    h2w_nf = _packet_by_profile(packet_rows, "h2y_h2w_no_controller_fallback")
    h2w_vs_h2u = _comparison_by_label(comparison_rows, "h2y_h2w_vs_h2u")
    h2u_fallback = _comparison_by_label(comparison_rows, "h2y_h2u_no_fallback_vs_h2u")
    h2w_fallback = _comparison_by_label(comparison_rows, "h2y_h2w_no_fallback_vs_h2w")
    h2w_fixed = [row for row in fixed_case_rows if row["comparison_label"] == "h2y_h2w_vs_h2u"]
    h2w_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h2y_h2w_semantic_target_preservation"
    ]
    h2w_families = [row for row in family_rows if row["profile_label"] == "h2y_h2w_semantic_target_preservation"]
    family_exact = ", ".join(
        f"{row['family']} {row['exact_success_count']}/{row['case_count']}" for row in h2w_families
    )
    stale_boundaries = [
        row
        for row in unresolved_boundary_rows
        if row["profile_label"] == "h2y_h2w_semantic_target_preservation"
        and row["family"] == "h2y_stale_selection_negation_context"
    ]
    return [
        {
            "finding_id": "h2y_scales_h2x_pressure_and_breaks_h2w_saturation",
            "finding": (
                f"H2y expands H2x to {h2w['case_count']} cases; H2u reaches "
                f"{h2u['exact_success_count']}/{h2u['case_count']} strict and "
                f"{h2u['executor_success_count']}/{h2u['case_count']} executor-equivalent, while H2w reaches "
                f"{h2w['exact_success_count']}/{h2w['case_count']} on both metrics."
            ),
        },
        {
            "finding_id": "semantic_preservation_remains_causal_but_partial",
            "finding": (
                f"H2w fixes {len(h2w_fixed)} H2u strict misses, gaining "
                f"{h2w_vs_h2u['delta_exact_rate']} exact rate and "
                f"{h2w_vs_h2u['delta_executor_equivalence_rate']} executor-equivalence rate, but leaves "
                f"{len(stale_boundaries)} stale-selection negation rows plus one value-before-component row unresolved."
            ),
        },
        {
            "finding_id": "fallback_remains_non_causal_on_h2y",
            "finding": (
                f"No-fallback controls tie their full rows: H2u fallback delta is {h2u_fallback['delta_exact_rate']} "
                f"exact/{h2u_fallback['delta_executor_equivalence_rate']} executor-equivalent, and H2w fallback delta is "
                f"{h2w_fallback['delta_exact_rate']} exact/{h2w_fallback['delta_executor_equivalence_rate']} executor-equivalent."
            ),
        },
        {
            "finding_id": "h2y_mechanism_mix_and_boundary",
            "finding": (
                f"H2w records {_intervention_count(h2w_interventions, 'visual_semantic_target_preservation')} "
                f"semantic-preservation interventions and "
                f"{_intervention_count(h2w_interventions, 'visual_target_query_normalization')} target-query "
                f"normalizations. Its exact family profile is {family_exact}."
            ),
        },
        {
            "finding_id": "next_helper_target_is_stale_selection_negation_and_short_component_value",
            "finding": (
                "The unresolved H2w rows show stale selection IDs passing through as refine_selection calls and "
                "the `not active alert banner` value collapsing to the short component query `alert`; the next helper "
                "ablation should target those two mechanisms, not fallback."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2y Scaled CLI Semantic Pressure Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2y scales the H2x packaged/CLI semantic-pressure gate to sixteen cases across quoted stale negation, "
            "stale selection negation, instructional negation, and genuine displayed negated values. It preserves "
            "the replay-live attribution path and runs matched no-fallback controls."
        ),
        "",
        (
            f"H2u reaches `{manifest['h2u_exact_success_count']} / {manifest['h2y_case_count']}` strict and "
            f"`{manifest['h2u_executor_success_count']} / {manifest['h2y_case_count']}` executor-equivalent. H2w reaches "
            f"`{manifest['h2w_exact_success_count']} / {manifest['h2y_case_count']}` on both metrics, a "
            f"`{manifest['h2w_delta_exact_vs_h2u']}` exact-rate gain and "
            f"`{manifest['h2w_delta_executor_vs_h2u']}` executor-equivalence gain."
        ),
        "",
        (
            "The no-fallback rows tie their full-controller rows, so fallback is still not the causal helper. The "
            "important new result is that H2w is no longer saturated: all stale-selection negation rows remain "
            "unresolved, and one value-before-component row collapses to a short component query."
        ),
        "",
        "![H2y scaled CLI semantic pressure gate](figures/h2y_scaled_cli_semantic_pressure_gate.svg)",
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
        "## Unresolved H2w Boundary Rows",
        "",
        _table(payload["unresolved_boundary_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    profiles = [
        ("h2y_h2u_negation_guard", "H2u", "#7C3AED"),
        ("h2y_h2u_no_controller_fallback", "H2u no fallback", "#A78BFA"),
        ("h2y_h2w_semantic_target_preservation", "H2w", "#92400E"),
        ("h2y_h2w_no_controller_fallback", "H2w no fallback", "#D97706"),
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
        '<title id="title">H2y scaled CLI semantic pressure gate</title>',
        '<desc id="desc">H2w improves over H2u but remains below saturation on stale-selection negation rows.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2y breaks H2w saturation while preserving fallback independence</text>',
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
            f'<text x="{x + 2}" y="{exact_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/16</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 10}" y="{executor_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["executor_success_count"])}/16</text>'
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
    parser = argparse.ArgumentParser(description="Build the H2y scaled CLI semantic pressure synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2y_scaled_cli_semantic_pressure_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
