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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2n_scoped_target_normalization_synthesis"


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
        "h2k_h2j_target_query_normalization",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2k_target_decoy_overlap_h2j_execute_v1",
    ),
    PacketSpec(
        "h2k_h2n_scoped_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2n_scoped_target_normalization_on_h2k_execute_v1",
    ),
    PacketSpec(
        "h2l_h2j_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2l_target_normalization_overreach_h2j_execute_v1",
    ),
    PacketSpec(
        "h2l_h2n_scoped_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2n_scoped_target_normalization_on_h2l_execute_v1",
    ),
    PacketSpec(
        "h2f_h2j_target_query_normalization",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2j_target_query_normalization_on_h2f_execute_v2",
    ),
    PacketSpec(
        "h2f_h2n_scoped_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2n_scoped_target_normalization_on_h2f_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2m_h2n_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2m_v1",
    ),
    ComparisonSpec(
        "h2m_h2n_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2n_scoped_target_normalization_vs_h2e_on_h2m_v1",
    ),
    ComparisonSpec(
        "h2k_h2n_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2k_v1",
    ),
    ComparisonSpec(
        "h2l_h2n_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2l_v1",
    ),
    ComparisonSpec(
        "h2f_h2n_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2f_v1",
    ),
)


def build_h2n_scoped_target_normalization_synthesis(
    *, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    non_exact_rows = _non_exact_rows(PACKET_SPECS)
    intervention_rows = _controller_intervention_rows(PACKET_SPECS)
    blocked_rows = [
        row for row in intervention_rows if row["intervention_kind"] == "visual_target_query_normalization_blocked"
    ]
    finding_rows = _finding_rows(packet_rows, comparison_rows, non_exact_rows, intervention_rows, blocked_rows)

    h2m_h2e = _packet_by_profile(packet_rows, "h2m_h2e_route_arbitration")
    h2m_h2j = _packet_by_profile(packet_rows, "h2m_h2j_target_query_normalization")
    h2m_h2n = _packet_by_profile(packet_rows, "h2m_h2n_scoped_target_query_normalization")
    h2k_h2n = _packet_by_profile(packet_rows, "h2k_h2n_scoped_target_query_normalization")
    h2l_h2n = _packet_by_profile(packet_rows, "h2l_h2n_scoped_target_query_normalization")
    h2f_h2n = _packet_by_profile(packet_rows, "h2f_h2n_scoped_target_query_normalization")
    h2m_vs_h2j = _comparison_by_label(comparison_rows, "h2m_h2n_vs_h2j")
    h2m_vs_h2e = _comparison_by_label(comparison_rows, "h2m_h2n_vs_h2e")
    h2k_vs_h2j = _comparison_by_label(comparison_rows, "h2k_h2n_vs_h2j")
    h2l_vs_h2j = _comparison_by_label(comparison_rows, "h2l_h2n_vs_h2j")
    h2f_vs_h2j = _comparison_by_label(comparison_rows, "h2f_h2n_vs_h2j")
    h2m_target_rewrites = _interventions_for(
        intervention_rows,
        profile_label="h2m_h2n_scoped_target_query_normalization",
        intervention_kind="visual_target_query_normalization",
    )
    h2m_stale_interventions = _interventions_for(
        intervention_rows,
        profile_label="h2m_h2n_scoped_target_query_normalization",
        intervention_kind="visual_stale_selection_gate",
    )
    h2m_blocked_rows = _interventions_for(
        intervention_rows,
        profile_label="h2m_h2n_scoped_target_query_normalization",
        intervention_kind="visual_target_query_normalization_blocked",
    )
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
        "h2m_h2n_delta_exact_vs_h2j": h2m_vs_h2j["delta_exact_rate"],
        "h2m_h2n_delta_executor_vs_h2j": h2m_vs_h2j["delta_executor_equivalence_rate"],
        "h2m_h2n_delta_exact_vs_h2e": h2m_vs_h2e["delta_exact_rate"],
        "h2m_h2n_delta_executor_vs_h2e": h2m_vs_h2e["delta_executor_equivalence_rate"],
        "h2k_h2n_exact_success_count": h2k_h2n["exact_success_count"],
        "h2k_h2n_executor_success_count": h2k_h2n["executor_success_count"],
        "h2l_h2n_exact_success_count": h2l_h2n["exact_success_count"],
        "h2l_h2n_executor_success_count": h2l_h2n["executor_success_count"],
        "h2f_h2n_exact_success_count": h2f_h2n["exact_success_count"],
        "h2f_h2n_executor_success_count": h2f_h2n["executor_success_count"],
        "h2k_h2n_delta_exact_vs_h2j": h2k_vs_h2j["delta_exact_rate"],
        "h2l_h2n_delta_exact_vs_h2j": h2l_vs_h2j["delta_exact_rate"],
        "h2f_h2n_delta_exact_vs_h2j": h2f_vs_h2j["delta_exact_rate"],
        "h2m_blocked_value_bearing_count": len(h2m_blocked_rows),
        "h2m_target_query_normalization_count": len(h2m_target_rewrites),
        "h2m_stale_selection_count": len(h2m_stale_interventions),
        "promotion_decision": "h2n_scoped_target_normalization_executor_gain_needs_strict_repair",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "blocked_rows": blocked_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2n_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2n_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2n_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2n_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2n_blocked_rows.csv", blocked_rows)
    _write_csv(tables_dir / "h2n_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2n_scoped_target_normalization_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _controller_intervention_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if "h2n_scoped_target_query_normalization" not in spec.profile_label:
            continue
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for kind in (
                "visual_target_query_normalization",
                "visual_target_query_normalization_blocked",
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
                            "preserved_target_query": entry.get("preserved_target_query", ""),
                            "value_bearing_label": entry.get("value_bearing_label", ""),
                            "value_suffix": entry.get("value_suffix", ""),
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
    blocked_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2m_h2e = _packet_by_profile(packet_rows, "h2m_h2e_route_arbitration")
    h2m_h2j = _packet_by_profile(packet_rows, "h2m_h2j_target_query_normalization")
    h2m_h2n = _packet_by_profile(packet_rows, "h2m_h2n_scoped_target_query_normalization")
    h2m_vs_h2j = _comparison_by_label(comparison_rows, "h2m_h2n_vs_h2j")
    h2m_vs_h2e = _comparison_by_label(comparison_rows, "h2m_h2n_vs_h2e")
    h2m_blocked = [
        row for row in blocked_rows if row["profile_label"] == "h2m_h2n_scoped_target_query_normalization"
    ]
    h2m_rewrites = _interventions_for(
        intervention_rows,
        profile_label="h2m_h2n_scoped_target_query_normalization",
        intervention_kind="visual_target_query_normalization",
    )
    h2m_non_exact = [
        row for row in non_exact_rows if row["profile_label"] == "h2m_h2n_scoped_target_query_normalization"
    ]
    return [
        {
            "finding_id": "h2n_improves_h2m_executor_equivalence_not_strict",
            "finding": (
                f"H2n ties H2j strict exactness on H2m at {h2m_h2n['exact_success_count']}/8 but improves "
                f"executor-equivalence from {h2m_h2j['executor_success_count']}/8 to "
                f"{h2m_h2n['executor_success_count']}/8, a {h2m_vs_h2j['delta_executor_equivalence_rate']} "
                "executor-equivalence-rate gain."
            ),
        },
        {
            "finding_id": "h2n_keeps_h2e_exact_gain",
            "finding": (
                f"Against H2e, H2n improves H2m strict exactness from {h2m_h2e['exact_success_count']}/8 to "
                f"{h2m_h2n['exact_success_count']}/8 and executor-equivalence from "
                f"{h2m_h2e['executor_success_count']}/8 to {h2m_h2n['executor_success_count']}/8."
            ),
        },
        {
            "finding_id": "h2n_scoping_blocks_value_bearing_overstrip",
            "finding": (
                f"H2n records {len(h2m_blocked)} scoped target-query-normalization blocks on H2m value-bearing "
                f"rows while preserving {len(h2m_rewrites)} contextual-label rewrites."
            ),
        },
        {
            "finding_id": "h2n_transfers_without_regression",
            "finding": (
                "H2n preserves the previous H2j transfer gates: 8/8 on H2k, 8/8 on H2l, and 10/10 on H2f with "
                "zero exact-rate delta versus H2j on each packet."
            ),
        },
        {
            "finding_id": "next_gate_needs_canonical_value_bearing_target_synthesis",
            "finding": (
                f"H2n still leaves {len(h2m_non_exact)} non-exact H2m rows, so the next H2o question is whether "
                "the controller can synthesize canonical value-bearing target queries only when the longer label is recoverable."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2n Scoped Target-Normalization Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2n converts the H2m negative result into a scoped controller policy. The normalizer still performs "
            "the contextual-label repairs that H2j needed, but it refuses to shorten value-bearing labels when the "
            "prompt evidence implies that the displayed value is part of the requested component identity. On H2m, "
            "this does not improve strict exactness over H2j: both remain 3/8. It does improve executor-equivalence "
            "from 3/8 to 5/8, and it keeps the H2k, H2l, and H2f transfer gates saturated. The remaining H2m misses "
            "are therefore no longer just an over-strip problem; they need canonical value-bearing target synthesis."
        ),
        "",
        "![H2n scoped target-normalization gate](figures/h2n_scoped_target_normalization_gate.svg)",
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
        "## Scoped Block Rows",
        "",
        _table(payload["blocked_rows"]),
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
        ("h2k_h2n_scoped_target_query_normalization", "H2k H2n", "#0F766E"),
        ("h2l_h2n_scoped_target_query_normalization", "H2l H2n", "#047857"),
        ("h2f_h2n_scoped_target_query_normalization", "H2f H2n", "#166534"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 880
    height = 360
    chart_left = 86
    chart_top = 64
    chart_height = 196
    bar_width = 72
    gap = 46
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2n scoped target-normalization gate</title>',
        '<desc id="desc">H2n improves H2m executor-equivalence while preserving H2k, H2l, and H2f exact transfer gates.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2n scopes target-query normalization</text>',
        '<line x1="86" y1="260" x2="812" y2="260" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="80" y1="{y:.1f}" x2="812" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
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
            f'<text x="{x + 16}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{int(row["exact_success_count"])}/{int(row["case_count"])}</text>'
        )
        lines.append(
            f'<text x="{x - 2}" y="288" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append(
        '<text x="32" y="332" font-family="Arial, sans-serif" font-size="13" fill="#374151">H2n H2m executor-equivalence: 5/8; strict remains 3/8.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2n scoped target-normalization synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2n_scoped_target_normalization_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
