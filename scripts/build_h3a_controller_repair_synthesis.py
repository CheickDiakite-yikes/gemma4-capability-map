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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h3a_controller_repair_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h3_h2z_boundary_combined",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h3_cli_controller_holdout_h2z_combined_execute_v1",
    ),
    PacketSpec(
        "h3a_stale_selection_paraphrase_guard",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h3_cli_controller_holdout_h3a_stale_paraphrase_execute_v1",
    ),
    PacketSpec(
        "h3a_negative_value_target_preservation",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h3_cli_controller_holdout_h3a_negative_value_execute_v1",
    ),
    PacketSpec(
        "h3a_boundary_combined",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h3_cli_controller_holdout_h3a_combined_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h3a_combined_vs_h2z_combined",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h3_cli_controller_holdout_h3a_combined_vs_h2z_combined_v1",
    ),
    ComparisonSpec(
        "h3a_stale_vs_h2z_combined",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h3_cli_controller_holdout_h3a_stale_vs_h2z_combined_v1",
    ),
    ComparisonSpec(
        "h3a_negative_vs_h2z_combined",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h3_cli_controller_holdout_h3a_negative_vs_h2z_combined_v1",
    ),
    ComparisonSpec(
        "h3a_combined_vs_stale",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h3_cli_controller_holdout_h3a_combined_vs_stale_v1",
    ),
    ComparisonSpec(
        "h3a_combined_vs_negative",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h3_cli_controller_holdout_h3a_combined_vs_negative_v1",
    ),
)


def build_h3a_controller_repair_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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

    h2z = _packet_by_profile(packet_rows, "h3_h2z_boundary_combined")
    stale = _packet_by_profile(packet_rows, "h3a_stale_selection_paraphrase_guard")
    negative = _packet_by_profile(packet_rows, "h3a_negative_value_target_preservation")
    combined = _packet_by_profile(packet_rows, "h3a_boundary_combined")
    combined_vs_h2z = _comparison_by_label(comparison_rows, "h3a_combined_vs_h2z_combined")
    stale_vs_h2z = _comparison_by_label(comparison_rows, "h3a_stale_vs_h2z_combined")
    negative_vs_h2z = _comparison_by_label(comparison_rows, "h3a_negative_vs_h2z_combined")
    combined_vs_stale = _comparison_by_label(comparison_rows, "h3a_combined_vs_stale")
    combined_vs_negative = _comparison_by_label(comparison_rows, "h3a_combined_vs_negative")
    stale_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h3a_stale_selection_paraphrase_guard"
    ]
    negative_interventions = [
        row for row in intervention_rows if row["profile_label"] == "h3a_negative_value_target_preservation"
    ]
    combined_interventions = [row for row in intervention_rows if row["profile_label"] == "h3a_boundary_combined"]

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h3_case_count": combined["case_count"],
        "h2z_combined_exact_success_count": h2z["exact_success_count"],
        "h2z_combined_executor_success_count": h2z["executor_success_count"],
        "h3a_stale_exact_success_count": stale["exact_success_count"],
        "h3a_stale_executor_success_count": stale["executor_success_count"],
        "h3a_negative_exact_success_count": negative["exact_success_count"],
        "h3a_negative_executor_success_count": negative["executor_success_count"],
        "h3a_combined_exact_success_count": combined["exact_success_count"],
        "h3a_combined_executor_success_count": combined["executor_success_count"],
        "h3a_stale_delta_exact_vs_h2z": stale_vs_h2z["delta_exact_rate"],
        "h3a_stale_delta_executor_vs_h2z": stale_vs_h2z["delta_executor_equivalence_rate"],
        "h3a_negative_delta_exact_vs_h2z": negative_vs_h2z["delta_exact_rate"],
        "h3a_negative_delta_executor_vs_h2z": negative_vs_h2z["delta_executor_equivalence_rate"],
        "h3a_combined_delta_exact_vs_h2z": combined_vs_h2z["delta_exact_rate"],
        "h3a_combined_delta_executor_vs_h2z": combined_vs_h2z["delta_executor_equivalence_rate"],
        "h3a_combined_delta_exact_vs_stale": combined_vs_stale["delta_exact_rate"],
        "h3a_combined_delta_executor_vs_stale": combined_vs_stale["delta_executor_equivalence_rate"],
        "h3a_combined_delta_exact_vs_negative": combined_vs_negative["delta_exact_rate"],
        "h3a_combined_delta_executor_vs_negative": combined_vs_negative["delta_executor_equivalence_rate"],
        "h3a_stale_fixed_case_count_vs_h2z": _fixed_count(fixed_case_rows, "h3a_stale_vs_h2z_combined"),
        "h3a_negative_fixed_case_count_vs_h2z": _fixed_count(fixed_case_rows, "h3a_negative_vs_h2z_combined"),
        "h3a_combined_fixed_case_count_vs_h2z": _fixed_count(
            fixed_case_rows,
            "h3a_combined_vs_h2z_combined",
        ),
        "h3a_combined_fixed_case_count_vs_stale": _fixed_count(fixed_case_rows, "h3a_combined_vs_stale"),
        "h3a_combined_fixed_case_count_vs_negative": _fixed_count(fixed_case_rows, "h3a_combined_vs_negative"),
        "h3a_stale_paraphrase_intervention_count": _intervention_count(
            stale_interventions,
            "visual_stale_selection_paraphrase_guard",
        ),
        "h3a_negative_value_intervention_count": _intervention_count(
            negative_interventions,
            "visual_negative_value_component_target_preservation",
        ),
        "h3a_combined_stale_paraphrase_intervention_count": _intervention_count(
            combined_interventions,
            "visual_stale_selection_paraphrase_guard",
        ),
        "h3a_combined_negative_value_intervention_count": _intervention_count(
            combined_interventions,
            "visual_negative_value_component_target_preservation",
        ),
        "h3a_combined_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"] == "h3a_boundary_combined"
        ),
        "h3a_closes_h3_exact_and_executor": combined["exact_success_count"] == combined["case_count"]
        and combined["executor_success_count"] == combined["case_count"],
        "h3a_factorial_residuals_separate": (
            _fixed_count(fixed_case_rows, "h3a_stale_vs_h2z_combined") == 4
            and _fixed_count(fixed_case_rows, "h3a_negative_vs_h2z_combined") == 1
            and _fixed_count(fixed_case_rows, "h3a_combined_vs_h2z_combined") == 5
        ),
        "promotion_decision": "use_h3a_as_next_candidate_but_require_h2y_h2z_h3_transfer_reruns_before_global_promotion",
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

    _write_csv(tables_dir / "h3a_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h3a_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h3a_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h3a_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h3a_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h3a_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h3a_findings.csv", finding_rows)
    _write_svg(figures_dir / "h3a_controller_repair_gate.svg", packet_rows)
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
    h2z = _packet_by_profile(packet_rows, "h3_h2z_boundary_combined")
    stale = _packet_by_profile(packet_rows, "h3a_stale_selection_paraphrase_guard")
    negative = _packet_by_profile(packet_rows, "h3a_negative_value_target_preservation")
    combined = _packet_by_profile(packet_rows, "h3a_boundary_combined")
    combined_vs_h2z = _comparison_by_label(comparison_rows, "h3a_combined_vs_h2z_combined")
    stale_fixed = [row["case_id"] for row in fixed_case_rows if row["comparison_label"] == "h3a_stale_vs_h2z_combined"]
    negative_fixed = [
        row["case_id"] for row in fixed_case_rows if row["comparison_label"] == "h3a_negative_vs_h2z_combined"
    ]
    combined_fixed = [
        row["case_id"] for row in fixed_case_rows if row["comparison_label"] == "h3a_combined_vs_h2z_combined"
    ]
    stale_non_exact = [
        row["case_id"] for row in non_exact_rows if row["profile_label"] == "h3a_stale_selection_paraphrase_guard"
    ]
    negative_non_exact = [
        row["case_id"] for row in non_exact_rows if row["profile_label"] == "h3a_negative_value_target_preservation"
    ]
    combined_interventions = [row for row in intervention_rows if row["profile_label"] == "h3a_boundary_combined"]
    stale_intervention_count = _intervention_count(
        combined_interventions,
        "visual_stale_selection_paraphrase_guard",
    )
    negative_intervention_count = _intervention_count(
        combined_interventions,
        "visual_negative_value_component_target_preservation",
    )
    return [
        {
            "finding_id": "h3a_combined_closes_h3",
            "finding": (
                f"H3a combined moves the H3 holdout from H2z {h2z['exact_success_count']}/{h2z['case_count']} "
                f"to {combined['exact_success_count']}/{combined['case_count']} strict and executor-equivalent, "
                f"a {combined_vs_h2z['delta_exact_rate']} exact-rate delta."
            ),
        },
        {
            "finding_id": "h3a_stale_paraphrase_guard_is_separable",
            "finding": (
                f"The stale-selection paraphrase guard reaches {stale['exact_success_count']}/{stale['case_count']} "
                f"and fixes {len(stale_fixed)} H2z misses ({', '.join(stale_fixed)}), while leaving "
                f"{', '.join(stale_non_exact)}."
            ),
        },
        {
            "finding_id": "h3a_negative_value_guard_is_separable",
            "finding": (
                f"The negative-value target-preservation guard reaches {negative['exact_success_count']}/"
                f"{negative['case_count']} and fixes {len(negative_fixed)} H2z miss "
                f"({', '.join(negative_fixed)}), while leaving {', '.join(negative_non_exact)}."
            ),
        },
        {
            "finding_id": "h3a_interventions_match_repair_surface",
            "finding": (
                f"The combined row records {stale_intervention_count} stale-paraphrase interventions and "
                f"{negative_intervention_count} negative-value interventions, matching the five fixed H3 cases "
                f"({', '.join(combined_fixed)})."
            ),
        },
        {
            "finding_id": "h3a_benchmark_quality_step",
            "finding": (
                "H3a upgrades the H3 result from a promotion-blocking negative gate to a row-attributable repair "
                "experiment: baseline, two single-helper rows, combined helper row, executor-equivalence scores, "
                "fixed-case lists, and controller-helper traces all agree."
            ),
        },
        {
            "finding_id": "h3a_next_transfer_gate",
            "finding": (
                "H3a should be treated as the next candidate controller posture, not a global promotion, until H2y, "
                "H2z, H3, and transfer/back-compat replay rows are rerun under the same CLI attribution standard."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H3a Controller Repair Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H3a repairs the fresh H3 controller holdout with two separable helper extensions: stale-selection "
            "paraphrase detection and negative-value component target preservation."
        ),
        "",
        (
            f"H2z combined baseline: `{manifest['h2z_combined_exact_success_count']} / {manifest['h3_case_count']}`. "
            f"H3a stale-only: `{manifest['h3a_stale_exact_success_count']} / {manifest['h3_case_count']}`. "
            f"H3a negative-only: `{manifest['h3a_negative_exact_success_count']} / {manifest['h3_case_count']}`. "
            f"H3a combined: `{manifest['h3a_combined_exact_success_count']} / {manifest['h3_case_count']}`."
        ),
        "",
        (
            "Decision: H3a is the next candidate controller posture, but it still needs transfer regression before "
            "global promotion. The result is strong because fixed cases and intervention traces split cleanly across "
            "the two residual mechanism classes."
        ),
        "",
        "![H3a controller repair gate](figures/h3a_controller_repair_gate.svg)",
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
        ("h3_h2z_boundary_combined", "H2z combined", "#B91C1C"),
        ("h3a_stale_selection_paraphrase_guard", "H3a stale", "#047857"),
        ("h3a_negative_value_target_preservation", "H3a negative", "#B45309"),
        ("h3a_boundary_combined", "H3a combined", "#1D4ED8"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 980
    height = 370
    chart_left = 82
    chart_top = 64
    chart_height = 196
    group_width = 192
    bar_width = 42
    denominator = int(next(iter(by_profile.values()))["case_count"]) if by_profile else 20
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H3a controller repair gate</title>',
        '<desc id="desc">H3a closes the fresh H3 holdout with separable stale-paraphrase and negative-value controls.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H3a closes the H3 controller holdout</text>',
        '<line x1="82" y1="260" x2="880" y2="260" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="76" y1="{y:.1f}" x2="880" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
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
            f'<text x="{x + 2}" y="{max(18, exact_y - 8):.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/{denominator}</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 10}" y="{max(18, executor_y - 8):.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["executor_success_count"])}/{denominator}</text>'
        )
        lines.append(
            f'<text x="{x - 8}" y="288" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.extend(
        [
            '<rect x="730" y="312" width="18" height="12" fill="#1D4ED8"/>',
            '<text x="754" y="323" font-family="Arial, sans-serif" font-size="12" fill="#374151">strict</text>',
            '<rect x="730" y="332" width="18" height="12" fill="#1D4ED8" opacity="0.45"/>',
            '<text x="754" y="343" font-family="Arial, sans-serif" font-size="12" fill="#374151">executor-equivalent</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H3a controller repair synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h3a_controller_repair_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
