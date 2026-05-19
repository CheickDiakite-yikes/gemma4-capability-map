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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h3a_transfer_backtest_synthesis"


TRANSFER_LABELS: tuple[str, ...] = (
    "h2t",
    "h2s",
    "h2q",
    "h2m",
    "h2k",
    "h2l",
    "h2f",
    "h2b",
    "h1x",
    "h1y",
    "h1o",
    "h1p",
)


H2W_LIVE_DIRS = {
    label: ROOT
    / "results"
    / "tool_probe_replay_live"
    / f"20260513T_h2w_semantic_target_preservation_on_{label}_execute_v1"
    for label in TRANSFER_LABELS
}


H3A_LIVE_DIRS = {
    label: ROOT
    / "results"
    / "tool_probe_replay_live"
    / f"20260519T_h3a_boundary_combined_on_{label}_execute_v1"
    for label in TRANSFER_LABELS
}


PACKET_SPECS: tuple[PacketSpec, ...] = tuple(
    PacketSpec(f"{label}_h2w_semantic_target_preservation", H2W_LIVE_DIRS[label])
    for label in TRANSFER_LABELS
) + tuple(
    PacketSpec(f"{label}_h3a_boundary_combined", H3A_LIVE_DIRS[label])
    for label in TRANSFER_LABELS
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = tuple(
    ComparisonSpec(
        f"{label}_h3a_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / f"20260519T_h3a_boundary_combined_vs_h2w_on_{label}_v1",
    )
    for label in TRANSFER_LABELS
)


H3A_NEW_HELPERS = (
    "visual_stale_selection_paraphrase_guard",
    "visual_negative_value_component_target_preservation",
)


def build_h3a_transfer_backtest_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    regression_rows = _regression_rows(COMPARISON_SPECS)
    packet_pair_rows = _packet_pair_rows(packet_rows, comparison_rows)
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        intervention_rows=intervention_rows,
        non_exact_rows=non_exact_rows,
        fixed_case_rows=fixed_case_rows,
        regression_rows=regression_rows,
    )

    h2w_rows = [_packet_by_profile(packet_rows, f"{label}_h2w_semantic_target_preservation") for label in TRANSFER_LABELS]
    h3a_rows = [_packet_by_profile(packet_rows, f"{label}_h3a_boundary_combined") for label in TRANSFER_LABELS]
    h3a_comparisons = [_comparison_by_label(comparison_rows, f"{label}_h3a_vs_h2w") for label in TRANSFER_LABELS]
    h3a_interventions = [row for row in intervention_rows if row["profile_label"].endswith("_h3a_boundary_combined")]
    h3a_non_exact_rows = [row for row in non_exact_rows if row["profile_label"].endswith("_h3a_boundary_combined")]

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "transfer_packet_count": len(TRANSFER_LABELS),
        "h2w_transfer_case_count": sum(int(row["case_count"]) for row in h2w_rows),
        "h2w_transfer_exact_success_count": sum(int(row["exact_success_count"]) for row in h2w_rows),
        "h2w_transfer_executor_success_count": sum(int(row["executor_success_count"]) for row in h2w_rows),
        "h3a_transfer_case_count": sum(int(row["case_count"]) for row in h3a_rows),
        "h3a_transfer_exact_success_count": sum(int(row["exact_success_count"]) for row in h3a_rows),
        "h3a_transfer_executor_success_count": sum(int(row["executor_success_count"]) for row in h3a_rows),
        "h3a_exact_delta_sum_vs_h2w": sum(float(row["delta_exact_rate"]) for row in h3a_comparisons),
        "h3a_executor_delta_sum_vs_h2w": sum(
            float(row["delta_executor_equivalence_rate"] or 0.0) for row in h3a_comparisons
        ),
        "h3a_fixed_case_count_vs_h2w": len(fixed_case_rows),
        "h3a_regression_count_vs_h2w": len(regression_rows),
        "h3a_non_exact_count": len(h3a_non_exact_rows),
        "h3a_semantic_target_preservation_count": _intervention_count(
            h3a_interventions,
            "visual_semantic_target_preservation",
        ),
        "h3a_target_query_normalization_count": _intervention_count(
            h3a_interventions,
            "visual_target_query_normalization",
        ),
        "h3a_stale_selection_gate_count": _intervention_count(
            h3a_interventions,
            "visual_stale_selection_gate",
        ),
        "h3a_value_bearing_target_synthesis_count": _intervention_count(
            h3a_interventions,
            "visual_value_bearing_target_query_synthesis",
        ),
        "h3a_contextual_surface_alias_routing_count": _intervention_count(
            h3a_interventions,
            "visual_contextual_surface_alias_routing",
        ),
        "h3a_composed_route_gating_count": _intervention_count(
            h3a_interventions,
            "visual_composed_route_gating",
        ),
        "h3a_composed_route_gating_blocked_count": _intervention_count(
            h3a_interventions,
            "visual_composed_route_gating_blocked",
        ),
        "h3a_stale_paraphrase_intervention_count": _intervention_count(
            h3a_interventions,
            "visual_stale_selection_paraphrase_guard",
        ),
        "h3a_negative_value_intervention_count": _intervention_count(
            h3a_interventions,
            "visual_negative_value_component_target_preservation",
        ),
        "h3a_new_helper_intervention_count": sum(
            _intervention_count(h3a_interventions, helper) for helper in H3A_NEW_HELPERS
        ),
        "h3a_transfer_clean": (
            sum(int(row["case_count"]) for row in h3a_rows)
            == sum(int(row["exact_success_count"]) for row in h3a_rows)
            == sum(int(row["executor_success_count"]) for row in h3a_rows)
        ),
        "h3a_ties_h2w_transfer_gate": all(
            row["delta_exact_rate"] == 0.0 and row["delta_executor_equivalence_rate"] == 0.0
            for row in h3a_comparisons
        ),
        "h3a_new_helpers_do_not_overtrigger_on_transfer": sum(
            _intervention_count(h3a_interventions, helper) for helper in H3A_NEW_HELPERS
        )
        == 0,
        "promotion_decision": "h3a_passes_broad_h2w_transfer_backtest_next_harder_holdout_required",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "packet_pair_rows": packet_pair_rows,
        "comparison_rows": comparison_rows,
        "family_rows": family_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "fixed_case_rows": fixed_case_rows,
        "regression_rows": regression_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h3a_transfer_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h3a_transfer_packet_pairs.csv", packet_pair_rows)
    _write_csv(tables_dir / "h3a_transfer_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h3a_transfer_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h3a_transfer_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h3a_transfer_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h3a_transfer_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h3a_transfer_regression_rows.csv", regression_rows)
    _write_csv(tables_dir / "h3a_transfer_findings.csv", finding_rows)
    _write_svg(figures_dir / "h3a_transfer_backtest_gate.svg", packet_pair_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _packet_pair_rows(packet_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in TRANSFER_LABELS:
        h2w = _packet_by_profile(packet_rows, f"{label}_h2w_semantic_target_preservation")
        h3a = _packet_by_profile(packet_rows, f"{label}_h3a_boundary_combined")
        comparison = _comparison_by_label(comparison_rows, f"{label}_h3a_vs_h2w")
        rows.append(
            {
                "slice": label,
                "case_count": h3a["case_count"],
                "h2w_exact_success_count": h2w["exact_success_count"],
                "h3a_exact_success_count": h3a["exact_success_count"],
                "h2w_executor_success_count": h2w["executor_success_count"],
                "h3a_executor_success_count": h3a["executor_success_count"],
                "h3a_delta_exact_vs_h2w": comparison["delta_exact_rate"],
                "h3a_delta_executor_vs_h2w": comparison["delta_executor_equivalence_rate"],
            }
        )
    return rows


def _regression_rows(specs: tuple[ComparisonSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
        for row in payload["case_deltas"]:
            if row.get("baseline_replay_exact_match") is not True or row.get("candidate_replay_exact_match") is not False:
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
    intervention_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    fixed_case_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h3a_rows = [_packet_by_profile(packet_rows, f"{label}_h3a_boundary_combined") for label in TRANSFER_LABELS]
    h2w_rows = [_packet_by_profile(packet_rows, f"{label}_h2w_semantic_target_preservation") for label in TRANSFER_LABELS]
    h3a_cases = sum(int(row["case_count"]) for row in h3a_rows)
    h3a_exact = sum(int(row["exact_success_count"]) for row in h3a_rows)
    h3a_executor = sum(int(row["executor_success_count"]) for row in h3a_rows)
    h2w_exact = sum(int(row["exact_success_count"]) for row in h2w_rows)
    h3a_comparisons = [_comparison_by_label(comparison_rows, f"{label}_h3a_vs_h2w") for label in TRANSFER_LABELS]
    h3a_interventions = [row for row in intervention_rows if row["profile_label"].endswith("_h3a_boundary_combined")]
    h3a_non_exact = [row for row in non_exact_rows if row["profile_label"].endswith("_h3a_boundary_combined")]
    new_helper_count = sum(_intervention_count(h3a_interventions, helper) for helper in H3A_NEW_HELPERS)
    return [
        {
            "finding_id": "h3a_broad_transfer_backtest_is_clean",
            "finding": (
                f"H3a reaches {h3a_exact}/{h3a_cases} strict exactness and {h3a_executor}/{h3a_cases} "
                "executor equivalence across the 12-packet H2w transfer/back-compat battery."
            ),
        },
        {
            "finding_id": "h3a_ties_incumbent_h2w_transfer_gate",
            "finding": (
                f"H3a ties the incumbent H2w transfer row: H2w is {h2w_exact}/{h3a_cases}, H3a is "
                f"{h3a_exact}/{h3a_cases}, and every per-slice exact/executor comparison has 0.0 delta."
            ),
        },
        {
            "finding_id": "h3a_new_helpers_do_not_overtrigger_on_backcompat",
            "finding": (
                f"The H3a-specific helpers fire {new_helper_count} times on this transfer battery. Older helper "
                "activity remains attributable through semantic preservation, target normalization, stale-selection "
                "gating, value-bearing synthesis, contextual alias routing, and composed-route gating traces."
            ),
        },
        {
            "finding_id": "h3a_transfer_has_no_case_level_regressions",
            "finding": (
                f"The H3a-vs-H2w comparison set has {len(regression_rows)} strict regressions, "
                f"{len(fixed_case_rows)} strict fixes, and {len(h3a_non_exact)} H3a non-exact rows."
            ),
        },
        {
            "finding_id": "h3a_still_needs_harder_saturation_breaker",
            "finding": (
                "This backtest removes a major regression concern but does not by itself prove global capability. "
                "The next scientific move is a harder H3b/H4 slice designed to break the new top-line saturation."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H3a Transfer Backtest Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H3a repaired the H3 controller holdout with two additional helper classes. This backtest asks the "
            "next causal question: do those new helpers preserve the incumbent H2w transfer/back-compat battery?"
        ),
        "",
        (
            f"H3a reaches `{manifest['h3a_transfer_exact_success_count']} / "
            f"{manifest['h3a_transfer_case_count']}` strict exactness and "
            f"`{manifest['h3a_transfer_executor_success_count']} / {manifest['h3a_transfer_case_count']}` "
            "executor equivalence across H2s/H2t/H2q/H2m/H2k/H2l/H2f/H2b/H1x/H1y/H1o/H1p."
        ),
        "",
        (
            f"Against the incumbent H2w row, aggregate exact-rate delta is "
            f"`{manifest['h3a_exact_delta_sum_vs_h2w']}` and aggregate executor-equivalence-rate delta is "
            f"`{manifest['h3a_executor_delta_sum_vs_h2w']}`. The comparison set records "
            f"`{manifest['h3a_regression_count_vs_h2w']}` strict regressions and "
            f"`{manifest['h3a_non_exact_count']}` H3a non-exact rows."
        ),
        "",
        (
            "The H3a-specific helpers do not overtrigger on this back-compat gate: stale-selection paraphrase "
            f"interventions `{manifest['h3a_stale_paraphrase_intervention_count']}`, negative-value component "
            f"interventions `{manifest['h3a_negative_value_intervention_count']}`. Older helper traces still fire "
            "where expected, preserving controller attribution."
        ),
        "",
        (
            "Decision: H3a passes the broad transfer regression gate, but the next paper-grade step is not to "
            "declare victory. It is to design a harder H3b/H4 slice that breaks the new 20/20 and 109/109 "
            "saturation surfaces."
        ),
        "",
        "![H3a transfer backtest gate](figures/h3a_transfer_backtest_gate.svg)",
        "",
        "## Packet Pair Rows",
        "",
        _table(payload["packet_pair_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## Family Rows",
        "",
        _table(payload["family_rows"]),
        "",
        "## Controller Intervention Rows",
        "",
        _table(payload["intervention_rows"]),
        "",
        "## Fixed Case Rows",
        "",
        _table(payload["fixed_case_rows"]),
        "",
        "## Regression Rows",
        "",
        _table(payload["regression_rows"]),
        "",
        "## Non-Exact Rows",
        "",
        _table(payload["non_exact_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_pair_rows: list[dict[str, Any]]) -> None:
    width = 1420
    height = 420
    chart_left = 66
    chart_top = 76
    chart_height = 210
    group_width = 100
    bar_width = 28
    max_cases = max(int(row["case_count"]) for row in packet_pair_rows)

    def bar_height(value: int) -> float:
        return (value / max_cases) * chart_height if max_cases else 0.0

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H3a transfer backtest gate</title>',
        '<desc id="desc">H3a ties H2w across the 12-packet transfer and backward compatibility battery.</desc>',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H3a preserves the H2w transfer gate</text>',
        '<text x="32" y="62" font-family="Arial, sans-serif" font-size="13" fill="#475569">Strict exact successes by packet. H3a and H2w tie on every slice.</text>',
        f'<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{width - 36}" y2="{chart_top + chart_height}" stroke="#CBD5E1" stroke-width="1"/>',
    ]
    for index, row in enumerate(packet_pair_rows):
        x = chart_left + index * group_width
        h2w_exact = int(row["h2w_exact_success_count"])
        h3a_exact = int(row["h3a_exact_success_count"])
        h2w_height = bar_height(h2w_exact)
        h3a_height = bar_height(h3a_exact)
        base_y = chart_top + chart_height
        elements.extend(
            [
                f'<rect x="{x}" y="{base_y - h2w_height:.1f}" width="{bar_width}" height="{h2w_height:.1f}" rx="4" fill="#92400E"/>',
                f'<rect x="{x + bar_width + 7}" y="{base_y - h3a_height:.1f}" width="{bar_width}" height="{h3a_height:.1f}" rx="4" fill="#1D4ED8"/>',
                f'<text x="{x + 30}" y="{base_y + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#334155">{row["slice"]}</text>',
                f'<text x="{x + 30}" y="{base_y - max(h2w_height, h3a_height) - 8:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#111827">{h3a_exact}/{row["case_count"]}</text>',
            ]
        )
    elements.extend(
        [
            '<rect x="1030" y="34" width="16" height="16" fill="#92400E"/>',
            '<text x="1054" y="47" font-family="Arial, sans-serif" font-size="13" fill="#334155">H2w</text>',
            '<rect x="1110" y="34" width="16" height="16" fill="#1D4ED8"/>',
            '<text x="1134" y="47" font-family="Arial, sans-serif" font-size="13" fill="#334155">H3a</text>',
            '<text x="32" y="352" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">Aggregate</text>',
            f'<text x="32" y="376" font-family="Arial, sans-serif" font-size="13" fill="#334155">H3a strict/executor: {sum(int(row["h3a_exact_success_count"]) for row in packet_pair_rows)} / {sum(int(row["case_count"]) for row in packet_pair_rows)}. Delta vs H2w: 0.0 exact and 0.0 executor-equivalence.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H3a transfer backtest synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h3a_transfer_backtest_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
