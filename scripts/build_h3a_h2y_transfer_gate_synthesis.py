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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h3a_h2y_transfer_gate_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2y_h2w_semantic_target_preservation",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h2y_scaled_cli_semantic_pressure_h2w_execute_v1",
    ),
    PacketSpec(
        "h2y_h2z_boundary_combined",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h2z_combined_execute_v1",
    ),
    PacketSpec(
        "h2y_h3a_boundary_combined",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h3a_combined_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2y_h3a_vs_h2z",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h3a_combined_vs_h2z_combined_v1",
    ),
    ComparisonSpec(
        "h2y_h3a_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h2y_scaled_cli_semantic_pressure_h3a_combined_vs_h2w_v1",
    ),
)


def build_h3a_h2y_transfer_gate_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    )

    h2w = _packet_by_profile(packet_rows, "h2y_h2w_semantic_target_preservation")
    h2z = _packet_by_profile(packet_rows, "h2y_h2z_boundary_combined")
    h3a = _packet_by_profile(packet_rows, "h2y_h3a_boundary_combined")
    h3a_vs_h2z = _comparison_by_label(comparison_rows, "h2y_h3a_vs_h2z")
    h3a_vs_h2w = _comparison_by_label(comparison_rows, "h2y_h3a_vs_h2w")
    h3a_interventions = [row for row in intervention_rows if row["profile_label"] == "h2y_h3a_boundary_combined"]

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2y_case_count": h3a["case_count"],
        "h2w_exact_success_count": h2w["exact_success_count"],
        "h2w_executor_success_count": h2w["executor_success_count"],
        "h2z_exact_success_count": h2z["exact_success_count"],
        "h2z_executor_success_count": h2z["executor_success_count"],
        "h3a_exact_success_count": h3a["exact_success_count"],
        "h3a_executor_success_count": h3a["executor_success_count"],
        "h3a_delta_exact_vs_h2z": h3a_vs_h2z["delta_exact_rate"],
        "h3a_delta_executor_vs_h2z": h3a_vs_h2z["delta_executor_equivalence_rate"],
        "h3a_delta_exact_vs_h2w": h3a_vs_h2w["delta_exact_rate"],
        "h3a_delta_executor_vs_h2w": h3a_vs_h2w["delta_executor_equivalence_rate"],
        "h3a_fixed_case_count_vs_h2z": _fixed_count(fixed_case_rows, "h2y_h3a_vs_h2z"),
        "h3a_fixed_case_count_vs_h2w": _fixed_count(fixed_case_rows, "h2y_h3a_vs_h2w"),
        "h3a_stale_negation_intervention_count": _intervention_count(
            h3a_interventions,
            "visual_stale_selection_negation_guard",
        ),
        "h3a_negated_component_intervention_count": _intervention_count(
            h3a_interventions,
            "visual_negated_component_target_preservation",
        ),
        "h3a_stale_paraphrase_intervention_count": _intervention_count(
            h3a_interventions,
            "visual_stale_selection_paraphrase_guard",
        ),
        "h3a_negative_value_intervention_count": _intervention_count(
            h3a_interventions,
            "visual_negative_value_component_target_preservation",
        ),
        "h3a_non_exact_count": sum(1 for row in non_exact_rows if row["profile_label"] == "h2y_h3a_boundary_combined"),
        "h3a_preserves_h2z_h2y_closure": (
            h3a["exact_success_count"] == h2z["exact_success_count"]
            and h3a_vs_h2z["delta_exact_rate"] == 0.0
            and h3a_vs_h2z["delta_executor_equivalence_rate"] == 0.0
        ),
        "h3a_new_helpers_do_not_overtrigger_on_h2y": (
            _intervention_count(h3a_interventions, "visual_stale_selection_paraphrase_guard") == 0
            and _intervention_count(h3a_interventions, "visual_negative_value_component_target_preservation") == 0
        ),
        "promotion_decision": "h3a_passes_first_h2y_transfer_gate_but_needs_broader_backcompat_before_global_promotion",
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

    _write_csv(tables_dir / "h3a_h2y_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h3a_h2y_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h3a_h2y_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h3a_h2y_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h3a_h2y_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h3a_h2y_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h3a_h2y_findings.csv", finding_rows)
    _write_svg(figures_dir / "h3a_h2y_transfer_gate.svg", packet_rows)
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
) -> list[dict[str, str]]:
    h2w = _packet_by_profile(packet_rows, "h2y_h2w_semantic_target_preservation")
    h2z = _packet_by_profile(packet_rows, "h2y_h2z_boundary_combined")
    h3a = _packet_by_profile(packet_rows, "h2y_h3a_boundary_combined")
    h3a_vs_h2z = _comparison_by_label(comparison_rows, "h2y_h3a_vs_h2z")
    h3a_vs_h2w = _comparison_by_label(comparison_rows, "h2y_h3a_vs_h2w")
    h3a_interventions = [row for row in intervention_rows if row["profile_label"] == "h2y_h3a_boundary_combined"]
    fixed_vs_h2w = [row["case_id"] for row in fixed_case_rows if row["comparison_label"] == "h2y_h3a_vs_h2w"]
    return [
        {
            "finding_id": "h3a_preserves_h2z_h2y_closure",
            "finding": (
                f"H3a reaches {h3a['exact_success_count']}/{h3a['case_count']} strict and executor-equivalent on "
                f"H2y, tying H2z at {h2z['exact_success_count']}/{h2z['case_count']} with "
                f"{h3a_vs_h2z['delta_exact_rate']} exact-rate delta."
            ),
        },
        {
            "finding_id": "h3a_retains_h2w_delta_on_h2y",
            "finding": (
                f"H3a retains the H2z boundary gain over H2w: H2w is {h2w['exact_success_count']}/"
                f"{h2w['case_count']}, H3a is {h3a['exact_success_count']}/{h3a['case_count']}, and "
                f"the strict/executor delta is {h3a_vs_h2w['delta_exact_rate']}."
            ),
        },
        {
            "finding_id": "h3a_h2y_fixed_cases_match_h2z_boundary",
            "finding": (
                f"H3a fixes {len(fixed_vs_h2w)} H2w misses versus H2w on H2y: {', '.join(fixed_vs_h2w)}."
            ),
        },
        {
            "finding_id": "h3a_h2y_uses_original_h2z_helpers",
            "finding": (
                "H3a records "
                f"{_intervention_count(h3a_interventions, 'visual_stale_selection_negation_guard')} "
                "stale-selection negation interventions and "
                f"{_intervention_count(h3a_interventions, 'visual_negated_component_target_preservation')} "
                "negated-component preservation interventions on H2y."
            ),
        },
        {
            "finding_id": "h3a_new_helpers_do_not_overtrigger_on_h2y",
            "finding": (
                "The H3a-specific helpers record "
                f"{_intervention_count(h3a_interventions, 'visual_stale_selection_paraphrase_guard')} "
                "stale-paraphrase interventions and "
                f"{_intervention_count(h3a_interventions, 'visual_negative_value_component_target_preservation')} "
                "negative-value interventions on H2y, so this first transfer gate does not show new-helper overreach."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H3a H2y Transfer Gate Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "This is the first transfer gate after the H3a repair. It reruns H3a combined on the original H2y "
            "scaled CLI semantic-pressure packet that H2z closed."
        ),
        "",
        (
            f"H2w: `{manifest['h2w_exact_success_count']} / {manifest['h2y_case_count']}`. "
            f"H2z combined: `{manifest['h2z_exact_success_count']} / {manifest['h2y_case_count']}`. "
            f"H3a combined: `{manifest['h3a_exact_success_count']} / {manifest['h2y_case_count']}`."
        ),
        "",
        (
            "Decision: H3a passes this first transfer gate by tying H2z on H2y and keeping the H2z gain over H2w. "
            "This is not yet global promotion; it is one regression slice in the broader transfer plan."
        ),
        "",
        "![H3a H2y transfer gate](figures/h3a_h2y_transfer_gate.svg)",
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
        ("h2y_h2z_boundary_combined", "H2z", "#166534"),
        ("h2y_h3a_boundary_combined", "H3a", "#1D4ED8"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 820
    height = 350
    chart_left = 82
    chart_top = 64
    chart_height = 186
    group_width = 190
    bar_width = 42
    denominator = int(next(iter(by_profile.values()))["case_count"]) if by_profile else 16
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H3a H2y transfer gate</title>',
        '<desc id="desc">H3a ties H2z on the H2y transfer gate and preserves the H2w delta.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H3a preserves the H2y closure</text>',
        '<line x1="82" y1="250" x2="690" y2="250" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="76" y1="{y:.1f}" x2="690" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
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
            f'<text x="{x + 2}" y="278" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.extend(
        [
            '<rect x="600" y="302" width="18" height="12" fill="#1D4ED8"/>',
            '<text x="624" y="313" font-family="Arial, sans-serif" font-size="12" fill="#374151">strict</text>',
            '<rect x="600" y="322" width="18" height="12" fill="#1D4ED8" opacity="0.45"/>',
            '<text x="624" y="333" font-family="Arial, sans-serif" font-size="12" fill="#374151">executor-equivalent</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H3a H2y transfer-gate synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h3a_h2y_transfer_gate_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
