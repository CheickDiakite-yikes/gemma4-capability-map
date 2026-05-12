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
from build_h2q_composed_surface_value_stale_synthesis import _family_rows
from build_h2r_composed_route_gating_synthesis import (
    _controller_intervention_rows,
    _intervention_counts_for,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2r_transfer_backtest_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2q_origin_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2q_execute_v2",
    ),
    PacketSpec(
        "h2m_transfer_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2m_execute_v1",
    ),
    PacketSpec(
        "h2k_transfer_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2k_execute_v2",
    ),
    PacketSpec(
        "h2l_transfer_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2l_execute_v2",
    ),
    PacketSpec(
        "h2f_transfer_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2f_execute_v1",
    ),
    PacketSpec(
        "h2b_regression_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h1x_regression_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1y_transfer_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1y_execute_v1",
    ),
    PacketSpec(
        "h1o_transfer_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1o_execute_v1",
    ),
    PacketSpec(
        "h1p_transfer_h2r",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1p_execute_v1",
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
    ComparisonSpec(
        "h2m_h2r_vs_h2p",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2p_on_h2m_v1",
    ),
    ComparisonSpec(
        "h2m_h2r_vs_h2o",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2o_on_h2m_v1",
    ),
    ComparisonSpec(
        "h2k_h2r_vs_h2p",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2p_on_h2k_v2",
    ),
    ComparisonSpec(
        "h2k_h2r_vs_h2o",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2o_on_h2k_v2",
    ),
    ComparisonSpec(
        "h2l_h2r_vs_h2p",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2p_on_h2l_v2",
    ),
    ComparisonSpec(
        "h2l_h2r_vs_h2o",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2o_on_h2l_v2",
    ),
    ComparisonSpec(
        "h2f_h2r_vs_h2p",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2p_on_h2f_v1",
    ),
    ComparisonSpec(
        "h2f_h2r_vs_h2o",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2o_on_h2f_v1",
    ),
    ComparisonSpec(
        "h2f_h2r_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2j_on_h2f_v1",
    ),
    ComparisonSpec(
        "h2b_h2r_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2j_on_h2b_v1",
    ),
    ComparisonSpec(
        "h2b_h2r_vs_h2h",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2h_on_h2b_v1",
    ),
    ComparisonSpec(
        "h1x_h2r_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2j_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x_h2r_vs_h2h",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2h_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1y_h2r_vs_h2a",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2a_on_h1y_v1",
    ),
    ComparisonSpec(
        "h1y_h2r_vs_component_residual",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_component_residual_on_h1y_v1",
    ),
    ComparisonSpec(
        "h1o_h2r_vs_h1s",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h1s_on_h1o_v1",
    ),
    ComparisonSpec(
        "h1o_h2r_vs_h2a",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2a_on_h1o_v1",
    ),
    ComparisonSpec(
        "h1p_h2r_vs_h1s",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h1s_on_h1p_v1",
    ),
    ComparisonSpec(
        "h1p_h2r_vs_h2a",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2r_composed_route_gating_vs_h2a_on_h1p_v1",
    ),
)


TRANSFER_PROFILE_LABELS = tuple(spec.profile_label for spec in PACKET_SPECS if spec.profile_label != "h2q_origin_h2r")


def build_h2r_transfer_backtest_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    transfer_rows = [row for row in packet_rows if row["profile_label"] in TRANSFER_PROFILE_LABELS]
    transfer_case_count = sum(int(row["case_count"]) for row in transfer_rows)
    transfer_exact_count = sum(int(row["exact_success_count"]) for row in transfer_rows)
    transfer_executor_count = sum(int(row["executor_success_count"]) for row in transfer_rows)
    all_case_count = sum(int(row["case_count"]) for row in packet_rows)
    all_exact_count = sum(int(row["exact_success_count"]) for row in packet_rows)
    h2q = _packet_by_profile(packet_rows, "h2q_origin_h2r")
    h2b_vs_h2h = _comparison_by_label(comparison_rows, "h2b_h2r_vs_h2h")
    h1x_vs_h2h = _comparison_by_label(comparison_rows, "h1x_h2r_vs_h2h")
    h1y_vs_h2a = _comparison_by_label(comparison_rows, "h1y_h2r_vs_h2a")
    h1o_vs_h1s = _comparison_by_label(comparison_rows, "h1o_h2r_vs_h1s")
    h1p_vs_h1s = _comparison_by_label(comparison_rows, "h1p_h2r_vs_h1s")
    intervention_counts = _aggregate_intervention_counts(intervention_rows)
    finding_rows = _finding_rows(
        transfer_case_count=transfer_case_count,
        transfer_exact_count=transfer_exact_count,
        transfer_executor_count=transfer_executor_count,
        all_case_count=all_case_count,
        all_exact_count=all_exact_count,
        h2b_vs_h2h=h2b_vs_h2h,
        h1x_vs_h2h=h1x_vs_h2h,
        h1y_vs_h2a=h1y_vs_h2a,
        h1o_vs_h1s=h1o_vs_h1s,
        h1p_vs_h1s=h1p_vs_h1s,
        intervention_counts=intervention_counts,
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "origin_case_count": int(h2q["case_count"]),
        "origin_exact_success_count": int(h2q["exact_success_count"]),
        "transfer_packet_count": len(transfer_rows),
        "transfer_case_count": transfer_case_count,
        "transfer_exact_success_count": transfer_exact_count,
        "transfer_executor_success_count": transfer_executor_count,
        "all_packet_count": len(packet_rows),
        "all_case_count": all_case_count,
        "all_exact_success_count": all_exact_count,
        "non_exact_count": len(non_exact_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2b_delta_exact_vs_h2h": h2b_vs_h2h["delta_exact_rate"],
        "h1x_delta_exact_vs_h2h": h1x_vs_h2h["delta_exact_rate"],
        "h1y_delta_exact_vs_h2a": h1y_vs_h2a["delta_exact_rate"],
        "h1o_delta_exact_vs_h1s": h1o_vs_h1s["delta_exact_rate"],
        "h1p_delta_exact_vs_h1s": h1p_vs_h1s["delta_exact_rate"],
        "intervention_counts": intervention_counts,
        "promotion_decision": "h2r_transfer_positive_current_packets_requires_fresh_h2s_holdout",
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

    _write_csv(tables_dir / "h2r_transfer_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2r_transfer_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2r_transfer_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2r_transfer_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2r_transfer_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2r_transfer_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2r_transfer_backtest_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _aggregate_intervention_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in (spec.profile_label for spec in PACKET_SPECS):
        profile_counts = _intervention_counts_for(rows, profile_label=label)
        for kind, count in profile_counts.items():
            counts[kind] = counts.get(kind, 0) + count
    return counts


def _finding_rows(
    *,
    transfer_case_count: int,
    transfer_exact_count: int,
    transfer_executor_count: int,
    all_case_count: int,
    all_exact_count: int,
    h2b_vs_h2h: dict[str, Any],
    h1x_vs_h2h: dict[str, Any],
    h1y_vs_h2a: dict[str, Any],
    h1o_vs_h1s: dict[str, Any],
    h1p_vs_h1s: dict[str, Any],
    intervention_counts: dict[str, int],
) -> list[dict[str, str]]:
    return [
        {
            "finding_id": "h2r_transfer_preserves_current_gates",
            "finding": (
                f"H2r reaches {transfer_exact_count}/{transfer_case_count} strict and "
                f"{transfer_executor_count}/{transfer_case_count} executor-equivalent across transfer packets, "
                f"and {all_exact_count}/{all_case_count} strict when the H2q origin packet is included."
            ),
        },
        {
            "finding_id": "h2r_avoids_h2h_regression_pattern",
            "finding": (
                "The explicit H2h regression guards are clean: H2r ties H2j/H2e on H2b and H1x while beating H2h "
                f"by {h2b_vs_h2h['delta_exact_rate']} exact-rate on H2b and "
                f"{h1x_vs_h2h['delta_exact_rate']} exact-rate on H1x."
            ),
        },
        {
            "finding_id": "h2r_closes_older_unsaturated_packets",
            "finding": (
                "Beyond preserving transfer gates, H2r closes older unsaturated packets: "
                f"H1y improves by {h1y_vs_h2a['delta_exact_rate']} exact-rate versus H2a, "
                f"H1o by {h1o_vs_h1s['delta_exact_rate']} versus H1s, and "
                f"H1p by {h1p_vs_h1s['delta_exact_rate']} versus H1s."
            ),
        },
        {
            "finding_id": "h2r_controller_burden_is_sparse_on_transfer",
            "finding": (
                "Transfer success is not just composed-route rewriting everywhere. Aggregate intervention counts are "
                f"{json.dumps(intervention_counts, sort_keys=True)}; several transfer packets saturate with zero "
                "new H2r-specific composed-route interventions."
            ),
        },
        {
            "finding_id": "h2r_next_requires_fresh_h2s",
            "finding": (
                "The current evidence supports H2r as transfer-positive on existing packets, but publication language "
                "should still require a fresh H2s composition holdout with unseen stale-selection and same-value "
                "surface decoys before calling the policy globally solved."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2r Transfer Backtest Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2r was introduced as a local repair for the H2q composition boundary. This synthesis asks the transfer "
            "question that H2h made unavoidable: does the local helper preserve older gates and harder adjacent packets, "
            "or does it trade one fixed slice for a new regression?"
        ),
        "",
        (
            f"The answer on the current packet set is positive. H2r reaches "
            f"`{manifest['transfer_exact_success_count']} / {manifest['transfer_case_count']}` strict and "
            f"`{manifest['transfer_executor_success_count']} / {manifest['transfer_case_count']}` executor-equivalent "
            "across transfer packets. Including the H2q origin packet, it is "
            f"`{manifest['all_exact_success_count']} / {manifest['all_case_count']}` strict."
        ),
        "",
        (
            "This should not be phrased as final global closure. It is transfer-positive on existing packets, including "
            "the explicit H2b/H1x regression gates, and it now justifies a fresh H2s holdout rather than more repair "
            "on the same H2q rows."
        ),
        "",
        "![H2r transfer backtest gate](figures/h2r_transfer_backtest_gate.svg)",
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
        ("h2q_origin_h2r", "H2q", "#B91C1C"),
        ("h2m_transfer_h2r", "H2m", "#047857"),
        ("h2k_transfer_h2r", "H2k", "#047857"),
        ("h2l_transfer_h2r", "H2l", "#047857"),
        ("h2f_transfer_h2r", "H2f", "#047857"),
        ("h2b_regression_h2r", "H2b", "#0891B2"),
        ("h1x_regression_h2r", "H1x", "#0891B2"),
        ("h1y_transfer_h2r", "H1y", "#7C3AED"),
        ("h1o_transfer_h2r", "H1o", "#7C3AED"),
        ("h1p_transfer_h2r", "H1p", "#7C3AED"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 980
    height = 390
    chart_left = 68
    chart_top = 72
    chart_height = 198
    bar_width = 52
    gap = 34
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2r transfer backtest saturates current packets</title>',
        '<desc id="desc">H2r reaches exact rate 1.0 on H2q and all current transfer packets.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2r transfer backtest across current replay packets</text>',
        '<line x1="68" y1="270" x2="930" y2="270" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="62" y1="{y:.1f}" x2="930" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="24" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, (profile, label, color) in enumerate(profiles):
        row = by_profile[profile]
        rate = float(row["exact_rate"])
        bar_height = rate * chart_height
        x = chart_left + index * (bar_width + gap)
        y = chart_top + chart_height - bar_height
        lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 9}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/{int(row["case_count"])}</text>'
        )
        lines.append(
            f'<text x="{x + 8}" y="298" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append(
        '<text x="32" y="340" font-family="Arial, sans-serif" font-size="13" fill="#374151">Red is the H2q origin packet; green are post-H2p/H2j transfer gates; blue are explicit H2h regression gates; purple are older unsaturated packets.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2r transfer backtest synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2r_transfer_backtest_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
