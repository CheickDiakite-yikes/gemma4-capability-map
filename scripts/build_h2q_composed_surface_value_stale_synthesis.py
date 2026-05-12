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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2q_composed_surface_value_stale_synthesis"


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
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2q_h2p_vs_h2o",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2q_composed_surface_value_stale_h2p_vs_h2o_v1",
    ),
    ComparisonSpec(
        "h2q_h2p_vs_h2n",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2q_composed_surface_value_stale_h2p_vs_h2n_v1",
    ),
    ComparisonSpec(
        "h2q_h2p_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2q_composed_surface_value_stale_h2p_vs_h2e_v1",
    ),
)


def build_h2q_composed_surface_value_stale_synthesis(
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
    h2p_non_exact = [
        row for row in non_exact_rows if row["profile_label"] == "h2q_h2p_contextual_surface_alias_routing"
    ]
    h2p_intervention_counts = _intervention_counts_for(
        intervention_rows, profile_label="h2q_h2p_contextual_surface_alias_routing"
    )
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        non_exact_rows=non_exact_rows,
        intervention_rows=intervention_rows,
    )

    h2e = _packet_by_profile(packet_rows, "h2q_h2e_route_arbitration")
    h2n = _packet_by_profile(packet_rows, "h2q_h2n_scoped_target_query_normalization")
    h2o = _packet_by_profile(packet_rows, "h2q_h2o_value_bearing_target_query_synthesis")
    h2p = _packet_by_profile(packet_rows, "h2q_h2p_contextual_surface_alias_routing")
    h2p_vs_h2o = _comparison_by_label(comparison_rows, "h2q_h2p_vs_h2o")
    h2p_vs_h2n = _comparison_by_label(comparison_rows, "h2q_h2p_vs_h2n")
    h2p_vs_h2e = _comparison_by_label(comparison_rows, "h2q_h2p_vs_h2e")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2q_h2e_exact_success_count": h2e["exact_success_count"],
        "h2q_h2e_executor_success_count": h2e["executor_success_count"],
        "h2q_h2n_exact_success_count": h2n["exact_success_count"],
        "h2q_h2n_executor_success_count": h2n["executor_success_count"],
        "h2q_h2o_exact_success_count": h2o["exact_success_count"],
        "h2q_h2o_executor_success_count": h2o["executor_success_count"],
        "h2q_h2p_exact_success_count": h2p["exact_success_count"],
        "h2q_h2p_executor_success_count": h2p["executor_success_count"],
        "h2q_h2p_delta_exact_vs_h2o": h2p_vs_h2o["delta_exact_rate"],
        "h2q_h2p_delta_executor_vs_h2o": h2p_vs_h2o["delta_executor_equivalence_rate"],
        "h2q_h2p_delta_exact_vs_h2n": h2p_vs_h2n["delta_exact_rate"],
        "h2q_h2p_delta_executor_vs_h2n": h2p_vs_h2n["delta_executor_equivalence_rate"],
        "h2q_h2p_delta_exact_vs_h2e": h2p_vs_h2e["delta_exact_rate"],
        "h2q_h2p_delta_executor_vs_h2e": h2p_vs_h2e["delta_executor_equivalence_rate"],
        "h2q_h2p_non_exact_count": len(h2p_non_exact),
        "h2q_h2p_wrong_tool_count": sum(
            1 for row in h2p_non_exact if row["failure_mode"] == "wrong_tool"
        ),
        "h2q_h2p_argument_mismatch_count": sum(
            1 for row in h2p_non_exact if row["failure_mode"] == "argument_mismatch"
        ),
        "h2q_h2p_contextual_surface_alias_routing_count": h2p_intervention_counts.get(
            "visual_contextual_surface_alias_routing", 0
        ),
        "h2q_h2p_value_bearing_synthesis_count": h2p_intervention_counts.get(
            "visual_value_bearing_target_query_synthesis", 0
        ),
        "h2q_h2p_target_query_normalization_count": h2p_intervention_counts.get(
            "visual_target_query_normalization", 0
        ),
        "h2q_h2p_stale_selection_gate_count": h2p_intervention_counts.get(
            "visual_stale_selection_gate", 0
        ),
        "promotion_decision": "h2q_breaks_h2p_composed_surface_value_stale_boundary",
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

    _write_csv(tables_dir / "h2q_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2q_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2q_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2q_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2q_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2q_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2q_composed_surface_value_stale_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _family_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        results = _read_json(spec.packet_dir / "live_replay_results.json")
        by_family: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            by_family.setdefault(str(result.get("family", "")), []).append(result)
        for family, family_results in sorted(by_family.items()):
            case_count = len(family_results)
            exact = sum(1 for row in family_results if row.get("replay_exact_match") is True)
            executor = sum(
                1 for row in family_results if row.get("replay_executor_equivalence_match") is True
            )
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "family": family,
                    "case_count": case_count,
                    "exact_success_count": exact,
                    "exact_rate": exact / case_count if case_count else 0.0,
                    "executor_success_count": executor,
                    "executor_rate": executor / case_count if case_count else 0.0,
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
            for kind in (
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
                            "value_suffix": entry.get("value_suffix", ""),
                            "matched_phrase": entry.get("matched_phrase", ""),
                            "display_value": entry.get("display_value", ""),
                            "surface_label": entry.get("surface_label", ""),
                            "surface_text": entry.get("surface_text", ""),
                            "surface_region_id": entry.get("surface_region_id", ""),
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
    h2o = _packet_by_profile(packet_rows, "h2q_h2o_value_bearing_target_query_synthesis")
    h2n = _packet_by_profile(packet_rows, "h2q_h2n_scoped_target_query_normalization")
    h2e = _packet_by_profile(packet_rows, "h2q_h2e_route_arbitration")
    h2p_vs_h2o = _comparison_by_label(comparison_rows, "h2q_h2p_vs_h2o")
    h2p_vs_h2n = _comparison_by_label(comparison_rows, "h2q_h2p_vs_h2n")
    h2p_vs_h2e = _comparison_by_label(comparison_rows, "h2q_h2p_vs_h2e")
    h2p_non_exact = [
        row for row in non_exact_rows if row["profile_label"] == "h2q_h2p_contextual_surface_alias_routing"
    ]
    h2p_wrong_tool = [row for row in h2p_non_exact if row["failure_mode"] == "wrong_tool"]
    h2p_argument_mismatch = [row for row in h2p_non_exact if row["failure_mode"] == "argument_mismatch"]
    h2p_counts = _intervention_counts_for(
        intervention_rows, profile_label="h2q_h2p_contextual_surface_alias_routing"
    )
    return [
        {
            "finding_id": "h2q_breaks_h2p_saturation",
            "finding": (
                f"H2q breaks the post-H2p H2m saturation: H2p reaches only "
                f"{h2p['exact_success_count']}/8 strict and {h2p['executor_success_count']}/8 "
                "executor-equivalent on the composed surface/value/stale packet."
            ),
        },
        {
            "finding_id": "h2q_h2p_remains_directionally_best",
            "finding": (
                f"H2p is still the best current row: H2o is {h2o['exact_success_count']}/8, "
                f"H2n is {h2n['exact_success_count']}/8, and H2e is {h2e['exact_success_count']}/8 strict. "
                f"H2p adds {h2p_vs_h2o['delta_exact_rate']} strict over H2o, "
                f"{h2p_vs_h2n['delta_exact_rate']} over H2n, and {h2p_vs_h2e['delta_exact_rate']} over H2e."
            ),
        },
        {
            "finding_id": "h2q_failures_are_tool_route_and_decoy_selection_failures",
            "finding": (
                f"H2p leaves {len(h2p_non_exact)} non-exact rows: {len(h2p_argument_mismatch)} argument "
                f"mismatches and {len(h2p_wrong_tool)} wrong-tool rows, so remaining error is not merely "
                "strict spelling drift."
            ),
        },
        {
            "finding_id": "h2q_composition_exposes_incomplete_helper_interaction",
            "finding": (
                "H2p records "
                f"{h2p_counts.get('visual_contextual_surface_alias_routing', 0)} contextual surface-alias, "
                f"{h2p_counts.get('visual_value_bearing_target_query_synthesis', 0)} value-bearing, "
                f"{h2p_counts.get('visual_target_query_normalization', 0)} target-normalization, and "
                f"{h2p_counts.get('visual_stale_selection_gate', 0)} stale-selection interventions, but still "
                "fails five rows under composed pressure."
            ),
        },
        {
            "finding_id": "next_slice_should_target_composed_route_gating",
            "finding": (
                "The next slice should target composed route gating: refuse stale refine_selection calls when the "
                "prompt says ignore old selection IDs, and prioritize requested surface classes over nearby same-value "
                "comments, banners, controls, and history context."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2q Composed Surface/Value/Stale Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2q is the first post-H2p saturation breaker. It composes surface aliases, value-bearing labels, "
            "stale-selection hints, and decoy overlap in one replay packet. H2p remains the strongest current "
            "controller stack, but reaches only 3/8 strict and executor-equivalent, so the research target has "
            "moved from isolated surface/value repair to composed route gating."
        ),
        "",
        "![H2q composed surface/value/stale gate](figures/h2q_composed_surface_value_stale_gate.svg)",
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
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 760
    height = 340
    chart_left = 88
    chart_top = 62
    chart_height = 190
    bar_width = 80
    gap = 62
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2q composed surface value stale gate</title>',
        '<desc id="desc">H2q breaks H2p saturation while H2p remains directionally better than H2o, H2n, and H2e.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2q breaks the post-H2p saturation</text>',
        '<line x1="88" y1="252" x2="650" y2="252" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="82" y1="{y:.1f}" x2="650" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
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
            f'<text x="{x + 22}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{int(row["exact_success_count"])}/{int(row["case_count"])}</text>'
        )
        lines.append(
            f'<text x="{x + 22}" y="280" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append(
        '<text x="32" y="318" font-family="Arial, sans-serif" font-size="13" fill="#374151">H2p is best but unsolved: 3/8 strict, with two wrong-tool stale-selection rows and three argument mismatches.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2q composed surface/value/stale synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2q_composed_surface_value_stale_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
