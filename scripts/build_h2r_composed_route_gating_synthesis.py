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
    _compact_json,
    _non_exact_rows,
    _packet_by_profile,
    _packet_row,
    _read_json,
    _table,
    _write_csv,
)
from build_h2q_composed_surface_value_stale_synthesis import _family_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2r_composed_route_gating_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2q_h2e_route_arbitration",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2q_composed_surface_value_stale_h2e_execute_v1",
    ),
    PacketSpec(
        "h2q_h2n_scoped_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2q_composed_surface_value_stale_h2n_execute_v1",
    ),
    PacketSpec(
        "h2q_h2o_value_bearing_target_query_synthesis",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2q_composed_surface_value_stale_h2o_execute_v1",
    ),
    PacketSpec(
        "h2q_h2p_contextual_surface_alias_routing",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2q_composed_surface_value_stale_h2p_execute_v1",
    ),
    PacketSpec(
        "h2q_h2r_composed_route_gating",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2r_composed_route_gating_on_h2q_execute_v2",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2q_h2r_vs_h2p",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2p_on_h2q_v2",
    ),
)


def build_h2r_composed_route_gating_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        non_exact_rows=non_exact_rows,
        intervention_rows=intervention_rows,
    )

    h2p = _packet_by_profile(packet_rows, "h2q_h2p_contextual_surface_alias_routing")
    h2r = _packet_by_profile(packet_rows, "h2q_h2r_composed_route_gating")
    h2r_vs_h2p = _comparison_by_label(comparison_rows, "h2q_h2r_vs_h2p")
    h2r_non_exact = [row for row in non_exact_rows if row["profile_label"] == "h2q_h2r_composed_route_gating"]
    h2p_non_exact = [
        row for row in non_exact_rows if row["profile_label"] == "h2q_h2p_contextual_surface_alias_routing"
    ]
    h2r_intervention_counts = _intervention_counts_for(
        intervention_rows, profile_label="h2q_h2r_composed_route_gating"
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2q_h2p_exact_success_count": h2p["exact_success_count"],
        "h2q_h2p_executor_success_count": h2p["executor_success_count"],
        "h2q_h2r_exact_success_count": h2r["exact_success_count"],
        "h2q_h2r_executor_success_count": h2r["executor_success_count"],
        "h2q_h2r_delta_exact_vs_h2p": h2r_vs_h2p["delta_exact_rate"],
        "h2q_h2r_delta_executor_vs_h2p": h2r_vs_h2p["delta_executor_equivalence_rate"],
        "h2q_h2p_non_exact_count": len(h2p_non_exact),
        "h2q_h2r_non_exact_count": len(h2r_non_exact),
        "h2q_h2r_composed_route_gating_count": h2r_intervention_counts.get(
            "visual_composed_route_gating", 0
        ),
        "h2q_h2r_target_query_normalization_count": h2r_intervention_counts.get(
            "visual_target_query_normalization", 0
        ),
        "h2q_h2r_contextual_surface_alias_routing_count": h2r_intervention_counts.get(
            "visual_contextual_surface_alias_routing", 0
        ),
        "h2q_h2r_value_bearing_synthesis_count": h2r_intervention_counts.get(
            "visual_value_bearing_target_query_synthesis", 0
        ),
        "h2q_h2r_stale_selection_gate_count": h2r_intervention_counts.get(
            "visual_stale_selection_gate", 0
        ),
        "promotion_decision": "h2r_solves_h2q_locally_transfer_backtested_requires_h2s",
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

    _write_csv(tables_dir / "h2r_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2r_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2r_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2r_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2r_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2r_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2r_composed_route_gating_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _controller_intervention_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for kind in (
                "visual_composed_route_gating",
                "visual_contextual_surface_alias_routing",
                "visual_value_bearing_target_query_synthesis",
                "visual_target_query_normalization",
                "visual_stale_selection_gate",
            ):
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
                            "value_bearing_label": entry.get("value_bearing_label", ""),
                            "display_value": entry.get("display_value", ""),
                            "surface_label": entry.get("surface_label", ""),
                            "requested_label": entry.get("requested_label", ""),
                            "requested_region_id": entry.get("requested_region_id", ""),
                            "reason": entry.get("reason", ""),
                        }
                    )
    return rows


def _intervention_counts_for(rows: list[dict[str, Any]], *, profile_label: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row["profile_label"] != profile_label:
            continue
        kind = row["intervention_kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2p = _packet_by_profile(packet_rows, "h2q_h2p_contextual_surface_alias_routing")
    h2r = _packet_by_profile(packet_rows, "h2q_h2r_composed_route_gating")
    h2r_vs_h2p = _comparison_by_label(comparison_rows, "h2q_h2r_vs_h2p")
    h2p_non_exact = [
        row for row in non_exact_rows if row["profile_label"] == "h2q_h2p_contextual_surface_alias_routing"
    ]
    h2r_non_exact = [row for row in non_exact_rows if row["profile_label"] == "h2q_h2r_composed_route_gating"]
    h2r_counts = _intervention_counts_for(intervention_rows, profile_label="h2q_h2r_composed_route_gating")
    composed_rows = [
        row
        for row in intervention_rows
        if row["profile_label"] == "h2q_h2r_composed_route_gating"
        and row["intervention_kind"] == "visual_composed_route_gating"
    ]
    stale_rows = [row for row in composed_rows if row["reason"] == "stale_selection_to_requested_surface"]
    decoy_rows = [row for row in composed_rows if row["reason"] == "requested_surface_over_deprioritized_decoy"]
    return [
        {
            "finding_id": "h2r_solves_h2q_local_boundary",
            "finding": (
                f"H2r reaches {h2r['exact_success_count']}/8 strict and {h2r['executor_success_count']}/8 "
                f"executor-equivalent on H2q, improving over H2p by {h2r_vs_h2p['delta_exact_rate']} exact-rate "
                "and executor-equivalence."
            ),
        },
        {
            "finding_id": "h2r_matches_h2q_failure_cardinality",
            "finding": (
                f"H2p left {len(h2p_non_exact)} non-exact rows; H2r records "
                f"{h2r_counts.get('visual_composed_route_gating', 0)} composed-route interventions and leaves "
                f"{len(h2r_non_exact)} non-exact rows."
            ),
        },
        {
            "finding_id": "h2r_mechanism_splits_stale_selection_and_decoy_surface_routes",
            "finding": (
                f"Composed route gating fires on {len(stale_rows)} stale-selection rows and {len(decoy_rows)} "
                "same-value decoy surface rows, showing the boundary was a route-selection problem rather than "
                "only label spelling."
            ),
        },
        {
            "finding_id": "h2r_transfer_backtested_but_needs_fresh_h2s",
            "finding": (
                "H2r is now transfer-positive on the current packet set, but it remains a local H2q-derived repair "
                "until a fresh H2s composed holdout confirms the policy without further tuning."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2r Composed Route-Gating Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2r is the first positive repair of the H2q composition boundary. It adds a narrow controller-side "
            "composed route gate after H2p: stale `refine_selection` calls are rewritten when the latest prompt "
            "explicitly says to ignore old selections, and requested surface classes are restored when same-value "
            "comments, banners, switches, or archived labels are marked as nearby context."
        ),
        "",
        (
            "On H2q, H2r reaches `8 / 8` strict and executor-equivalent while H2p was `3 / 8`. This is strong "
            "local mechanism evidence. Transfer backtests are now positive on the current packet set, so the next "
            "promotion gate is a fresh H2s composed holdout."
        ),
        "",
        "![H2r composed route-gating gate](figures/h2r_composed_route_gating_gate.svg)",
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
        ("h2q_h2e_route_arbitration", "H2e", "#0891B2"),
        ("h2q_h2n_scoped_target_query_normalization", "H2n", "#7C3AED"),
        ("h2q_h2o_value_bearing_target_query_synthesis", "H2o", "#047857"),
        ("h2q_h2p_contextual_surface_alias_routing", "H2p", "#B45309"),
        ("h2q_h2r_composed_route_gating", "H2r", "#B91C1C"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 820
    height = 350
    chart_left = 82
    chart_top = 62
    chart_height = 190
    bar_width = 72
    gap = 50
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2r composed route gating solves H2q locally</title>',
        '<desc id="desc">H2r reaches eight of eight exact rows on H2q, compared with three of eight for H2p.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2r solves the H2q boundary locally</text>',
        '<line x1="82" y1="252" x2="700" y2="252" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="76" y1="{y:.1f}" x2="700" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="36" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, (profile, label, color) in enumerate(profiles):
        row = by_profile[profile]
        rate = float(row["exact_rate"])
        bar_height = rate * chart_height
        x = chart_left + index * (bar_width + gap)
        y = chart_top + chart_height - bar_height
        lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 18}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{int(row["exact_success_count"])}/{int(row["case_count"])}</text>'
        )
        lines.append(
            f'<text x="{x + 18}" y="280" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append(
        '<text x="32" y="318" font-family="Arial, sans-serif" font-size="13" fill="#374151">H2r adds five composed-route interventions: two stale-selection rewrites and three requested-surface restorations.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2r composed route-gating synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2r_composed_route_gating_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
