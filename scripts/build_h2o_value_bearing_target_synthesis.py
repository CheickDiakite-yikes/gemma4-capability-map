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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2o_value_bearing_target_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2m_h2e_route_arbitration",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2m_less_direct_target_normalization_overreach_h2e_execute_v1",
    ),
    PacketSpec(
        "h2m_h2j_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_execute_v1",
    ),
    PacketSpec(
        "h2m_h2n_scoped_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2n_scoped_target_normalization_on_h2m_execute_v1",
    ),
    PacketSpec(
        "h2m_h2o_value_bearing_target_query_synthesis",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2o_value_bearing_target_synthesis_on_h2m_execute_v1",
    ),
    PacketSpec(
        "h2k_h2j_target_query_normalization",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2k_target_decoy_overlap_h2j_execute_v1",
    ),
    PacketSpec(
        "h2k_h2o_value_bearing_target_query_synthesis",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2o_value_bearing_target_synthesis_on_h2k_execute_v1",
    ),
    PacketSpec(
        "h2l_h2j_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2l_target_normalization_overreach_h2j_execute_v1",
    ),
    PacketSpec(
        "h2l_h2o_value_bearing_target_query_synthesis",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2o_value_bearing_target_synthesis_on_h2l_execute_v1",
    ),
    PacketSpec(
        "h2f_h2j_target_query_normalization",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2j_target_query_normalization_on_h2f_execute_v2",
    ),
    PacketSpec(
        "h2f_h2o_value_bearing_target_query_synthesis",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2o_value_bearing_target_synthesis_on_h2f_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2m_h2o_vs_h2n",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2o_value_bearing_target_synthesis_vs_h2n_on_h2m_v1",
    ),
    ComparisonSpec(
        "h2m_h2o_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2m_v1",
    ),
    ComparisonSpec(
        "h2m_h2o_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2o_value_bearing_target_synthesis_vs_h2e_on_h2m_v1",
    ),
    ComparisonSpec(
        "h2k_h2o_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2k_v1",
    ),
    ComparisonSpec(
        "h2l_h2o_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2l_v1",
    ),
    ComparisonSpec(
        "h2f_h2o_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2f_v1",
    ),
)


def build_h2o_value_bearing_target_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    non_exact_rows = _non_exact_rows(PACKET_SPECS)
    intervention_rows = _controller_intervention_rows(PACKET_SPECS)
    synthesis_rows = [
        row
        for row in intervention_rows
        if row["intervention_kind"] == "visual_value_bearing_target_query_synthesis"
    ]
    finding_rows = _finding_rows(packet_rows, comparison_rows, non_exact_rows, intervention_rows, synthesis_rows)

    h2m_h2e = _packet_by_profile(packet_rows, "h2m_h2e_route_arbitration")
    h2m_h2j = _packet_by_profile(packet_rows, "h2m_h2j_target_query_normalization")
    h2m_h2n = _packet_by_profile(packet_rows, "h2m_h2n_scoped_target_query_normalization")
    h2m_h2o = _packet_by_profile(packet_rows, "h2m_h2o_value_bearing_target_query_synthesis")
    h2k_h2o = _packet_by_profile(packet_rows, "h2k_h2o_value_bearing_target_query_synthesis")
    h2l_h2o = _packet_by_profile(packet_rows, "h2l_h2o_value_bearing_target_query_synthesis")
    h2f_h2o = _packet_by_profile(packet_rows, "h2f_h2o_value_bearing_target_query_synthesis")
    h2m_vs_h2n = _comparison_by_label(comparison_rows, "h2m_h2o_vs_h2n")
    h2m_vs_h2j = _comparison_by_label(comparison_rows, "h2m_h2o_vs_h2j")
    h2m_vs_h2e = _comparison_by_label(comparison_rows, "h2m_h2o_vs_h2e")
    h2k_vs_h2j = _comparison_by_label(comparison_rows, "h2k_h2o_vs_h2j")
    h2l_vs_h2j = _comparison_by_label(comparison_rows, "h2l_h2o_vs_h2j")
    h2f_vs_h2j = _comparison_by_label(comparison_rows, "h2f_h2o_vs_h2j")
    h2m_synthesis_rows = _interventions_for(
        synthesis_rows,
        profile_label="h2m_h2o_value_bearing_target_query_synthesis",
        intervention_kind="visual_value_bearing_target_query_synthesis",
    )
    h2m_target_rewrites = _interventions_for(
        intervention_rows,
        profile_label="h2m_h2o_value_bearing_target_query_synthesis",
        intervention_kind="visual_target_query_normalization",
    )
    h2m_stale_interventions = _interventions_for(
        intervention_rows,
        profile_label="h2m_h2o_value_bearing_target_query_synthesis",
        intervention_kind="visual_stale_selection_gate",
    )
    h2m_non_exact = [
        row for row in non_exact_rows if row["profile_label"] == "h2m_h2o_value_bearing_target_query_synthesis"
    ]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "h2m_h2e_exact_success_count": h2m_h2e["exact_success_count"],
        "h2m_h2e_executor_success_count": h2m_h2e["executor_success_count"],
        "h2m_h2j_exact_success_count": h2m_h2j["exact_success_count"],
        "h2m_h2j_executor_success_count": h2m_h2j["executor_success_count"],
        "h2m_h2n_exact_success_count": h2m_h2n["exact_success_count"],
        "h2m_h2n_executor_success_count": h2m_h2n["executor_success_count"],
        "h2m_h2o_exact_success_count": h2m_h2o["exact_success_count"],
        "h2m_h2o_executor_success_count": h2m_h2o["executor_success_count"],
        "h2m_h2o_delta_exact_vs_h2n": h2m_vs_h2n["delta_exact_rate"],
        "h2m_h2o_delta_executor_vs_h2n": h2m_vs_h2n["delta_executor_equivalence_rate"],
        "h2m_h2o_delta_exact_vs_h2j": h2m_vs_h2j["delta_exact_rate"],
        "h2m_h2o_delta_executor_vs_h2j": h2m_vs_h2j["delta_executor_equivalence_rate"],
        "h2m_h2o_delta_exact_vs_h2e": h2m_vs_h2e["delta_exact_rate"],
        "h2m_h2o_delta_executor_vs_h2e": h2m_vs_h2e["delta_executor_equivalence_rate"],
        "h2k_h2o_exact_success_count": h2k_h2o["exact_success_count"],
        "h2k_h2o_executor_success_count": h2k_h2o["executor_success_count"],
        "h2l_h2o_exact_success_count": h2l_h2o["exact_success_count"],
        "h2l_h2o_executor_success_count": h2l_h2o["executor_success_count"],
        "h2f_h2o_exact_success_count": h2f_h2o["exact_success_count"],
        "h2f_h2o_executor_success_count": h2f_h2o["executor_success_count"],
        "h2k_h2o_delta_exact_vs_h2j": h2k_vs_h2j["delta_exact_rate"],
        "h2l_h2o_delta_exact_vs_h2j": h2l_vs_h2j["delta_exact_rate"],
        "h2f_h2o_delta_exact_vs_h2j": h2f_vs_h2j["delta_exact_rate"],
        "h2m_value_bearing_synthesis_count": len(h2m_synthesis_rows),
        "h2m_target_query_normalization_count": len(h2m_target_rewrites),
        "h2m_stale_selection_count": len(h2m_stale_interventions),
        "h2m_non_exact_count": len(h2m_non_exact),
        "h2m_remaining_non_exact_case_id": "; ".join(row["case_id"] for row in h2m_non_exact),
        "promotion_decision": "h2o_value_bearing_synthesis_repairs_h2m_strict_with_contextual_alias_residue",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "synthesis_rows": synthesis_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2o_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2o_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2o_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2o_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2o_value_bearing_synthesis_rows.csv", synthesis_rows)
    _write_csv(tables_dir / "h2o_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2o_value_bearing_target_synthesis_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _controller_intervention_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if "h2o_value_bearing_target_query_synthesis" not in spec.profile_label:
            continue
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for kind in (
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
                            "reason": entry.get("reason", ""),
                        }
                    )
    return rows


def _interventions_for(
    rows: list[dict[str, Any]], *, profile_label: str, intervention_kind: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["profile_label"] == profile_label and row["intervention_kind"] == intervention_kind
    ]


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    synthesis_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2m_h2j = _packet_by_profile(packet_rows, "h2m_h2j_target_query_normalization")
    h2m_h2n = _packet_by_profile(packet_rows, "h2m_h2n_scoped_target_query_normalization")
    h2m_h2o = _packet_by_profile(packet_rows, "h2m_h2o_value_bearing_target_query_synthesis")
    h2m_vs_h2n = _comparison_by_label(comparison_rows, "h2m_h2o_vs_h2n")
    h2m_vs_h2j = _comparison_by_label(comparison_rows, "h2m_h2o_vs_h2j")
    h2m_non_exact = [
        row for row in non_exact_rows if row["profile_label"] == "h2m_h2o_value_bearing_target_query_synthesis"
    ]
    h2m_synthesis = _interventions_for(
        synthesis_rows,
        profile_label="h2m_h2o_value_bearing_target_query_synthesis",
        intervention_kind="visual_value_bearing_target_query_synthesis",
    )
    h2m_rewrites = _interventions_for(
        intervention_rows,
        profile_label="h2m_h2o_value_bearing_target_query_synthesis",
        intervention_kind="visual_target_query_normalization",
    )
    return [
        {
            "finding_id": "h2o_repairs_h2m_strict_value_bearing_rows",
            "finding": (
                f"H2o improves H2m strict exactness from H2n's {h2m_h2n['exact_success_count']}/8 to "
                f"{h2m_h2o['exact_success_count']}/8 and executor-equivalence from "
                f"{h2m_h2n['executor_success_count']}/8 to {h2m_h2o['executor_success_count']}/8."
            ),
        },
        {
            "finding_id": "h2o_outperforms_h2j_on_less_direct_h2m",
            "finding": (
                f"Against H2j, H2o adds {h2m_vs_h2j['delta_exact_rate']} exact-rate and "
                f"{h2m_vs_h2j['delta_executor_equivalence_rate']} executor-equivalence-rate on H2m, moving "
                f"from {h2m_h2j['exact_success_count']}/8 to {h2m_h2o['exact_success_count']}/8 strict."
            ),
        },
        {
            "finding_id": "h2o_mechanism_is_selective_canonical_synthesis",
            "finding": (
                f"H2o records {len(h2m_synthesis)} value-bearing target-query syntheses and "
                f"{len(h2m_rewrites)} ordinary contextual-label rewrites on H2m."
            ),
        },
        {
            "finding_id": "h2o_transfers_without_regression",
            "finding": (
                "H2o preserves saturated transfer gates: 8/8 on H2k, 8/8 on H2l, and 10/10 on H2f with zero "
                "exact-rate delta versus H2j on each transfer packet."
            ),
        },
        {
            "finding_id": "h2p_should_target_contextual_surface_aliases",
            "finding": (
                f"H2o leaves {len(h2m_non_exact)} H2m non-exact row: `h2m_result_tile_contextual_alias`, where "
                "the model kept the value `Blocked` instead of the surface-type alias `result tile`."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2o Value-Bearing Target Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2o turns the H2n no-op/blocking policy into selective canonical target-query construction. When a "
            "longer value-bearing visual label is recoverable from the image-state catalog and the prompt evidence "
            "asks for that value-bearing component, the controller rewrites the model's shorter or reordered "
            "target query to the canonical layout label. This repairs the main H2m strict failures: H2o reaches "
            "7/8 strict and executor-equivalent, compared with 3/8 strict for H2j and H2n. The transfer gates stay "
            "saturated on H2k, H2l, and H2f. The remaining H2m miss is a different problem: contextual surface-type "
            "alias routing, where the prompt says a tile-style result surface should target `result tile` but the "
            "model keeps the value `Blocked`."
        ),
        "",
        "![H2o value-bearing target synthesis gate](figures/h2o_value_bearing_target_synthesis_gate.svg)",
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## Non-Exact Rows",
        "",
        _table(payload["non_exact_rows"]),
        "",
        "## Controller Intervention Rows",
        "",
        _table(payload["intervention_rows"]),
        "",
        "## Value-Bearing Synthesis Rows",
        "",
        _table(payload["synthesis_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    profiles = [
        ("h2m_h2e_route_arbitration", "H2m H2e", "#0891B2"),
        ("h2m_h2j_target_query_normalization", "H2m H2j", "#1D4ED8"),
        ("h2m_h2n_scoped_target_query_normalization", "H2m H2n", "#0E7490"),
        ("h2m_h2o_value_bearing_target_query_synthesis", "H2m H2o", "#0F766E"),
        ("h2k_h2o_value_bearing_target_query_synthesis", "H2k H2o", "#047857"),
        ("h2l_h2o_value_bearing_target_query_synthesis", "H2l H2o", "#166534"),
        ("h2f_h2o_value_bearing_target_query_synthesis", "H2f H2o", "#365314"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 980
    height = 370
    chart_left = 78
    chart_top = 66
    chart_height = 198
    bar_width = 70
    gap = 44
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2o value-bearing target synthesis gate</title>',
        '<desc id="desc">H2o repairs H2m value-bearing target exactness while preserving H2k, H2l, and H2f transfer gates.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2o synthesizes canonical value-bearing targets</text>',
        '<line x1="78" y1="264" x2="884" y2="264" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="72" y1="{y:.1f}" x2="884" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="32" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, (profile, label, color) in enumerate(profiles):
        row = by_profile[profile]
        rate = float(row["exact_rate"])
        bar_height = rate * chart_height
        x = chart_left + index * (bar_width + gap)
        y = chart_top + chart_height - bar_height
        lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 16}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{int(row["exact_success_count"])}/{int(row["case_count"])}</text>'
        )
        lines.append(
            f'<text x="{x - 6}" y="292" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append(
        '<text x="32" y="336" font-family="Arial, sans-serif" font-size="13" fill="#374151">H2o H2m strict: 7/8; remaining miss is contextual surface-type alias routing.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2o value-bearing target synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2o_value_bearing_target_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
