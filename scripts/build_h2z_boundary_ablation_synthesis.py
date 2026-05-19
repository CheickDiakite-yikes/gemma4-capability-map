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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2z_boundary_ablation_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2y_h2w_semantic_target_preservation",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h2y_scaled_cli_semantic_pressure_h2w_execute_v1",
    ),
    PacketSpec(
        "h2z_stale_selection_negation_guard",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_stale_negation_execute_v1",
    ),
    PacketSpec(
        "h2z_negated_component_target_preservation",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_negated_component_execute_v1",
    ),
    PacketSpec(
        "h2z_boundary_combined",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_combined_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2z_stale_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_stale_vs_h2w_v1",
    ),
    ComparisonSpec(
        "h2z_component_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_component_vs_h2w_v1",
    ),
    ComparisonSpec(
        "h2z_combined_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_combined_vs_h2w_v1",
    ),
    ComparisonSpec(
        "h2z_combined_vs_stale",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_combined_vs_stale_v1",
    ),
    ComparisonSpec(
        "h2z_combined_vs_component",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_combined_vs_component_v1",
    ),
)


def build_h2z_boundary_ablation_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
        intervention_rows=intervention_rows,
        fixed_case_rows=fixed_case_rows,
        non_exact_rows=non_exact_rows,
    )

    h2w = _packet_by_profile(packet_rows, "h2y_h2w_semantic_target_preservation")
    stale = _packet_by_profile(packet_rows, "h2z_stale_selection_negation_guard")
    component = _packet_by_profile(packet_rows, "h2z_negated_component_target_preservation")
    combined = _packet_by_profile(packet_rows, "h2z_boundary_combined")
    stale_vs_h2w = _comparison_by_label(comparison_rows, "h2z_stale_vs_h2w")
    component_vs_h2w = _comparison_by_label(comparison_rows, "h2z_component_vs_h2w")
    combined_vs_h2w = _comparison_by_label(comparison_rows, "h2z_combined_vs_h2w")
    combined_vs_stale = _comparison_by_label(comparison_rows, "h2z_combined_vs_stale")
    combined_vs_component = _comparison_by_label(comparison_rows, "h2z_combined_vs_component")
    stale_interventions = [row for row in intervention_rows if row["profile_label"] == "h2z_stale_selection_negation_guard"]
    component_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h2z_negated_component_target_preservation"
    ]
    combined_interventions = [row for row in intervention_rows if row["profile_label"] == "h2z_boundary_combined"]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2y_case_count": combined["case_count"],
        "h2w_exact_success_count": h2w["exact_success_count"],
        "h2w_executor_success_count": h2w["executor_success_count"],
        "h2z_stale_exact_success_count": stale["exact_success_count"],
        "h2z_stale_executor_success_count": stale["executor_success_count"],
        "h2z_component_exact_success_count": component["exact_success_count"],
        "h2z_component_executor_success_count": component["executor_success_count"],
        "h2z_combined_exact_success_count": combined["exact_success_count"],
        "h2z_combined_executor_success_count": combined["executor_success_count"],
        "h2z_stale_delta_exact_vs_h2w": stale_vs_h2w["delta_exact_rate"],
        "h2z_component_delta_exact_vs_h2w": component_vs_h2w["delta_exact_rate"],
        "h2z_combined_delta_exact_vs_h2w": combined_vs_h2w["delta_exact_rate"],
        "h2z_stale_delta_executor_vs_h2w": stale_vs_h2w["delta_executor_equivalence_rate"],
        "h2z_component_delta_executor_vs_h2w": component_vs_h2w["delta_executor_equivalence_rate"],
        "h2z_combined_delta_executor_vs_h2w": combined_vs_h2w["delta_executor_equivalence_rate"],
        "h2z_combined_delta_exact_vs_stale": combined_vs_stale["delta_exact_rate"],
        "h2z_combined_delta_exact_vs_component": combined_vs_component["delta_exact_rate"],
        "h2z_stale_fixed_case_count_vs_h2w": _fixed_count(fixed_case_rows, "h2z_stale_vs_h2w"),
        "h2z_component_fixed_case_count_vs_h2w": _fixed_count(fixed_case_rows, "h2z_component_vs_h2w"),
        "h2z_combined_fixed_case_count_vs_h2w": _fixed_count(fixed_case_rows, "h2z_combined_vs_h2w"),
        "h2z_combined_fixed_case_count_vs_stale": _fixed_count(fixed_case_rows, "h2z_combined_vs_stale"),
        "h2z_combined_fixed_case_count_vs_component": _fixed_count(fixed_case_rows, "h2z_combined_vs_component"),
        "h2z_stale_negation_intervention_count": _intervention_count(
            stale_interventions,
            "visual_stale_selection_negation_guard",
        ),
        "h2z_component_preservation_intervention_count": _intervention_count(
            component_interventions,
            "visual_negated_component_target_preservation",
        ),
        "h2z_combined_stale_negation_intervention_count": _intervention_count(
            combined_interventions,
            "visual_stale_selection_negation_guard",
        ),
        "h2z_combined_component_preservation_intervention_count": _intervention_count(
            combined_interventions,
            "visual_negated_component_target_preservation",
        ),
        "h2z_combined_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"] == "h2z_boundary_combined"
        ),
        "additive_boundary_closure_holds": combined["exact_success_count"] == combined["case_count"]
        and _fixed_count(fixed_case_rows, "h2z_stale_vs_h2w") == 3
        and _fixed_count(fixed_case_rows, "h2z_component_vs_h2w") == 1
        and _fixed_count(fixed_case_rows, "h2z_combined_vs_h2w") == 4,
        "promotion_decision": "h2z_closes_h2y_boundary_but_requires_harder_holdout_before_global_promotion",
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

    _write_csv(tables_dir / "h2z_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2z_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2z_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2z_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2z_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2z_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h2z_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2z_boundary_ablation_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _fixed_count(rows: list[dict[str, Any]], comparison_label: str) -> int:
    return sum(1 for row in rows if row["comparison_label"] == comparison_label)


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    fixed_case_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2w = _packet_by_profile(packet_rows, "h2y_h2w_semantic_target_preservation")
    stale = _packet_by_profile(packet_rows, "h2z_stale_selection_negation_guard")
    component = _packet_by_profile(packet_rows, "h2z_negated_component_target_preservation")
    combined = _packet_by_profile(packet_rows, "h2z_boundary_combined")
    combined_vs_h2w = _comparison_by_label(comparison_rows, "h2z_combined_vs_h2w")
    stale_fixed = [
        row["case_id"] for row in fixed_case_rows if row["comparison_label"] == "h2z_stale_vs_h2w"
    ]
    component_fixed = [
        row["case_id"] for row in fixed_case_rows if row["comparison_label"] == "h2z_component_vs_h2w"
    ]
    combined_non_exact = [
        row["case_id"] for row in non_exact_rows if row["profile_label"] == "h2z_boundary_combined"
    ]
    combined_interventions = [row for row in intervention_rows if row["profile_label"] == "h2z_boundary_combined"]
    return [
        {
            "finding_id": "h2z_closes_h2y_scaled_cli_boundary",
            "finding": (
                f"H2z combined reaches {combined['exact_success_count']}/{combined['case_count']} strict and "
                f"{combined['executor_success_count']}/{combined['case_count']} executor-equivalent on H2y, up from "
                f"H2w's {h2w['exact_success_count']}/{h2w['case_count']}."
            ),
        },
        {
            "finding_id": "h2z_factorial_split_is_additive",
            "finding": (
                f"Stale-selection negation alone reaches {stale['exact_success_count']}/{stale['case_count']} and fixes "
                f"{len(stale_fixed)} H2w cases; negated-component preservation alone reaches "
                f"{component['exact_success_count']}/{component['case_count']} and fixes {len(component_fixed)} H2w case."
            ),
        },
        {
            "finding_id": "h2z_specific_mechanism_counts_match_cases",
            "finding": (
                "The combined row records "
                f"{_intervention_count(combined_interventions, 'visual_stale_selection_negation_guard')} stale-selection "
                "negation interventions and "
                f"{_intervention_count(combined_interventions, 'visual_negated_component_target_preservation')} "
                "negated-component target-preservation intervention."
            ),
        },
        {
            "finding_id": "h2z_strict_and_executor_metrics_move_together",
            "finding": (
                f"The H2z combined delta over H2w is {combined_vs_h2w['delta_exact_rate']} strict and "
                f"{combined_vs_h2w['delta_executor_equivalence_rate']} executor-equivalent; this is not an "
                "executor-only rescue."
            ),
        },
        {
            "finding_id": "h2z_next_step_is_harder_holdout",
            "finding": (
                "The combined row has "
                f"{len(combined_non_exact)} non-exact H2y rows, so this slice should not be overpromoted; the next "
                "scientific step is a harder H1/H3 holdout with new state, language, and workflow-family variation."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2z Boundary Ablation Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2z converts the H2y residual into a factorial controller-ablation result. H2w left four H2y rows "
            "unresolved; stale-selection negation fixes the three wrong-tool rows, negated-component preservation "
            "fixes the single short-query value row, and the combined profile closes the full 16-case packet."
        ),
        "",
        (
            f"H2w: `{manifest['h2w_exact_success_count']} / {manifest['h2y_case_count']}`. "
            f"H2z stale-only: `{manifest['h2z_stale_exact_success_count']} / {manifest['h2y_case_count']}`. "
            f"H2z component-only: `{manifest['h2z_component_exact_success_count']} / {manifest['h2y_case_count']}`. "
            f"H2z combined: `{manifest['h2z_combined_exact_success_count']} / {manifest['h2y_case_count']}`."
        ),
        "",
        "![H2z boundary ablation gate](figures/h2z_boundary_ablation_gate.svg)",
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
        ("h2y_h2w_semantic_target_preservation", "H2w", "#92400E"),
        ("h2z_stale_selection_negation_guard", "H2z stale", "#713F12"),
        ("h2z_negated_component_target_preservation", "H2z component", "#854D0E"),
        ("h2z_boundary_combined", "H2z combined", "#166534"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 960
    height = 360
    chart_left = 82
    chart_top = 64
    chart_height = 190
    group_width = 188
    bar_width = 42
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2z boundary ablation gate</title>',
        '<desc id="desc">Stale-selection negation and negated-component target preservation close the H2y residual.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2z turns the H2y residual into separable controller effects</text>',
        '<line x1="82" y1="254" x2="860" y2="254" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="76" y1="{y:.1f}" x2="860" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
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
            f'<text x="{x + 2}" y="{max(18, exact_y - 8):.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/16</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 10}" y="{max(18, executor_y - 8):.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["executor_success_count"])}/16</text>'
        )
        lines.append(
            f'<text x="{x - 8}" y="282" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.extend(
        [
            '<rect x="720" y="304" width="18" height="12" fill="#166534"/>',
            '<text x="744" y="315" font-family="Arial, sans-serif" font-size="12" fill="#374151">strict</text>',
            '<rect x="720" y="324" width="18" height="12" fill="#166534" opacity="0.45"/>',
            '<text x="744" y="335" font-family="Arial, sans-serif" font-size="12" fill="#374151">executor-equivalent</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2z boundary ablation synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2z_boundary_ablation_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
