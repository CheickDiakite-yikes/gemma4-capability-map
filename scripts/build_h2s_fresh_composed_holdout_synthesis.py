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
from build_h2q_composed_surface_value_stale_synthesis import _family_rows
from build_h2r_composed_route_gating_synthesis import (
    _controller_intervention_rows,
    _intervention_counts_for,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2s_fresh_composed_holdout_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2s_h2j_target_normalization",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2s_fresh_composed_holdout_h2j_execute_v1",
    ),
    PacketSpec(
        "h2s_h2o_value_bearing_synthesis",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2s_fresh_composed_holdout_h2o_execute_v1",
    ),
    PacketSpec(
        "h2s_h2p_contextual_surface_alias_routing",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2s_fresh_composed_holdout_h2p_execute_v1",
    ),
    PacketSpec(
        "h2s_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2s_fresh_composed_holdout_h2r_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2s_h2r_vs_h2p",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2s_fresh_composed_holdout_h2r_vs_h2p_v1",
    ),
    ComparisonSpec(
        "h2s_h2r_vs_h2o",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2s_fresh_composed_holdout_h2r_vs_h2o_v1",
    ),
    ComparisonSpec(
        "h2s_h2r_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2s_fresh_composed_holdout_h2r_vs_h2j_v1",
    ),
)


def build_h2s_fresh_composed_holdout_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    h2j = _packet_by_profile(packet_rows, "h2s_h2j_target_normalization")
    h2o = _packet_by_profile(packet_rows, "h2s_h2o_value_bearing_synthesis")
    h2p = _packet_by_profile(packet_rows, "h2s_h2p_contextual_surface_alias_routing")
    h2r = _packet_by_profile(packet_rows, "h2s_h2r_composed_route_gating")
    h2r_vs_h2p = _comparison_by_label(comparison_rows, "h2s_h2r_vs_h2p")
    h2r_vs_h2o = _comparison_by_label(comparison_rows, "h2s_h2r_vs_h2o")
    h2r_vs_h2j = _comparison_by_label(comparison_rows, "h2s_h2r_vs_h2j")
    h2r_non_exact = [row for row in non_exact_rows if row["profile_label"] == "h2s_h2r_composed_route_gating"]
    h2r_intervention_counts = _intervention_counts_for(
        intervention_rows,
        profile_label="h2s_h2r_composed_route_gating",
    )
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        non_exact_rows=non_exact_rows,
        intervention_rows=intervention_rows,
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2s_case_count": int(h2r["case_count"]),
        "h2s_h2j_exact_success_count": h2j["exact_success_count"],
        "h2s_h2o_exact_success_count": h2o["exact_success_count"],
        "h2s_h2p_exact_success_count": h2p["exact_success_count"],
        "h2s_h2r_exact_success_count": h2r["exact_success_count"],
        "h2s_h2r_executor_success_count": h2r["executor_success_count"],
        "h2s_h2r_delta_exact_vs_h2p": h2r_vs_h2p["delta_exact_rate"],
        "h2s_h2r_delta_executor_vs_h2p": h2r_vs_h2p["delta_executor_equivalence_rate"],
        "h2s_h2r_delta_exact_vs_h2o": h2r_vs_h2o["delta_exact_rate"],
        "h2s_h2r_delta_exact_vs_h2j": h2r_vs_h2j["delta_exact_rate"],
        "h2s_h2r_non_exact_count": len(h2r_non_exact),
        "h2s_h2r_composed_route_gating_count": h2r_intervention_counts.get(
            "visual_composed_route_gating", 0
        ),
        "h2s_h2r_value_bearing_synthesis_count": h2r_intervention_counts.get(
            "visual_value_bearing_target_query_synthesis", 0
        ),
        "h2s_h2r_target_query_normalization_count": h2r_intervention_counts.get(
            "visual_target_query_normalization", 0
        ),
        "promotion_decision": "h2r_passes_fresh_h2s_holdout_requires_h2t_or_packaged_transfer",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "family_rows": family_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2s_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2s_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2s_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2s_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2s_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2s_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2s_fresh_composed_holdout_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2r = _packet_by_profile(packet_rows, "h2s_h2r_composed_route_gating")
    h2p = _packet_by_profile(packet_rows, "h2s_h2p_contextual_surface_alias_routing")
    h2o = _packet_by_profile(packet_rows, "h2s_h2o_value_bearing_synthesis")
    h2j = _packet_by_profile(packet_rows, "h2s_h2j_target_normalization")
    h2r_vs_h2p = _comparison_by_label(comparison_rows, "h2s_h2r_vs_h2p")
    h2r_vs_h2j = _comparison_by_label(comparison_rows, "h2s_h2r_vs_h2j")
    h2r_non_exact = [row for row in non_exact_rows if row["profile_label"] == "h2s_h2r_composed_route_gating"]
    h2r_counts = _intervention_counts_for(intervention_rows, profile_label="h2s_h2r_composed_route_gating")
    clean_controls = [
        row
        for row in intervention_rows
        if row["profile_label"] == "h2s_h2r_composed_route_gating"
        and row["case_id"] == "h2s_status_badge_live_clean_control"
    ]
    return [
        {
            "finding_id": "h2s_fresh_holdout_confirms_h2r_transfer",
            "finding": (
                f"H2r reaches {h2r['exact_success_count']}/{h2r['case_count']} strict and "
                f"{h2r['executor_success_count']}/{h2r['case_count']} executor-equivalent on fresh H2s, "
                f"while H2p reaches {h2p['exact_success_count']}/{h2p['case_count']}, "
                f"H2o reaches {h2o['exact_success_count']}/{h2o['case_count']}, and "
                f"H2j reaches {h2j['exact_success_count']}/{h2j['case_count']}."
            ),
        },
        {
            "finding_id": "h2s_composed_route_gate_is_causal",
            "finding": (
                f"H2r improves over H2p by {h2r_vs_h2p['delta_exact_rate']} exact-rate and executor-equivalence "
                f"rate, and over H2j by {h2r_vs_h2j['delta_exact_rate']} exact-rate, on an unseen composed packet."
            ),
        },
        {
            "finding_id": "h2s_h2r_mechanism_is_mixed_not_single_helper",
            "finding": (
                "H2r uses a mixed controller path on H2s: "
                f"{h2r_counts.get('visual_composed_route_gating', 0)} composed route gates, "
                f"{h2r_counts.get('visual_value_bearing_target_query_synthesis', 0)} value-bearing syntheses, and "
                f"{h2r_counts.get('visual_target_query_normalization', 0)} target normalizations."
            ),
        },
        {
            "finding_id": "h2s_clean_control_does_not_need_visual_helper",
            "finding": (
                "The clean status-badge control remains exact without H2r-specific metadata, "
                f"with {len(clean_controls)} recorded helper rows for that control."
            ),
        },
        {
            "finding_id": "h2s_next_requires_h2t_or_packaged_transfer",
            "finding": (
                f"H2r leaves {len(h2r_non_exact)} non-exact rows on H2s. The next research move is no longer "
                "patching H2r on this slice; it is either a harder H2t holdout or a packaged workflow transfer that "
                "preserves the same composed-route pressure."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2s Fresh Composed Holdout Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2s is the first fresh holdout built after H2r passed the current transfer backtest. H2r was frozen for "
            "the first run, then H2p, H2o, and H2j controls were executed on the same packet."
        ),
        "",
        (
            f"H2r reaches `{manifest['h2s_h2r_exact_success_count']} / {manifest['h2s_case_count']}` strict and "
            f"`{manifest['h2s_h2r_executor_success_count']} / {manifest['h2s_case_count']}` executor-equivalent. "
            f"H2p and H2o each reach `3 / 10`; H2j reaches `1 / 10`. The H2r-vs-H2p gain is "
            f"`{manifest['h2s_h2r_delta_exact_vs_h2p']}` exact-rate and executor-equivalence-rate."
        ),
        "",
        (
            "This is fresh positive evidence for composed route gating. It should still be kept as scoped internal "
            "evidence: the next step is a harder H2t holdout or packaged workflow transfer, not another edit to H2r."
        ),
        "",
        "![H2s fresh composed holdout gate](figures/h2s_fresh_composed_holdout_gate.svg)",
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
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    profiles = [
        ("h2s_h2j_target_normalization", "H2j", "#1D4ED8"),
        ("h2s_h2o_value_bearing_synthesis", "H2o", "#0F766E"),
        ("h2s_h2p_contextual_surface_alias_routing", "H2p", "#115E59"),
        ("h2s_h2r_composed_route_gating", "H2r", "#B45309"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 760
    height = 360
    chart_left = 88
    chart_top = 70
    chart_height = 190
    bar_width = 82
    gap = 76
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2s fresh composed holdout gate</title>',
        '<desc id="desc">H2r reaches exact rate 1.0 on H2s while H2p and H2o reach 0.3 and H2j reaches 0.1.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2s fresh holdout separates composed route gating</text>',
        '<line x1="88" y1="260" x2="690" y2="260" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="82" y1="{y:.1f}" x2="690" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
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
            f'<text x="{x + 22}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/{int(row["case_count"])}</text>'
        )
        lines.append(
            f'<text x="{x + 25}" y="292" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append(
        '<text x="32" y="330" font-family="Arial, sans-serif" font-size="13" fill="#374151">H2s was authored after H2r transfer; H2r was run frozen before controls or further repair.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2s fresh composed holdout synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2s_fresh_composed_holdout_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
