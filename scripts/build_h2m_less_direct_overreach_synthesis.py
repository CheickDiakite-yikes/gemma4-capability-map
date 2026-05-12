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
    _controller_intervention_rows,
    _interventions_for,
    _non_exact_rows,
    _packet_by_profile,
    _packet_row,
    _read_json,
    _table,
    _write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2m_less_direct_overreach_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2e_route_arbitration",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2m_less_direct_target_normalization_overreach_h2e_execute_v1",
    ),
    PacketSpec(
        "h2j_target_query_normalization_no_stale_gate",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_no_stale_gate_execute_v1",
    ),
    PacketSpec(
        "h2j_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2j_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_h2e_v1",
    ),
    ComparisonSpec(
        "h2j_vs_no_stale_gate",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_no_stale_gate_v1",
    ),
)


def build_h2m_less_direct_overreach_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    non_exact_rows = _non_exact_rows(PACKET_SPECS)
    intervention_rows = _controller_intervention_rows(PACKET_SPECS)
    family_rows = _family_rows()
    overstrip_rows = _overstrip_rows(intervention_rows, non_exact_rows)
    finding_rows = _finding_rows(packet_rows, comparison_rows, non_exact_rows, intervention_rows, overstrip_rows)

    h2e = _packet_by_profile(packet_rows, "h2e_route_arbitration")
    h2j_no_stale = _packet_by_profile(packet_rows, "h2j_target_query_normalization_no_stale_gate")
    h2j = _packet_by_profile(packet_rows, "h2j_target_query_normalization")
    h2j_vs_h2e = _comparison_by_label(comparison_rows, "h2j_vs_h2e")
    h2j_vs_no_stale = _comparison_by_label(comparison_rows, "h2j_vs_no_stale_gate")
    full_target_interventions = _interventions_for(
        intervention_rows,
        profile_label="h2j_target_query_normalization",
        intervention_kind="visual_target_query_normalization",
    )
    full_stale_interventions = _interventions_for(
        intervention_rows,
        profile_label="h2j_target_query_normalization",
        intervention_kind="visual_stale_selection_gate",
    )
    no_stale_target_interventions = _interventions_for(
        intervention_rows,
        profile_label="h2j_target_query_normalization_no_stale_gate",
        intervention_kind="visual_target_query_normalization",
    )
    no_stale_stale_interventions = _interventions_for(
        intervention_rows,
        profile_label="h2j_target_query_normalization_no_stale_gate",
        intervention_kind="visual_stale_selection_gate",
    )
    h2j_exact_case_ids = _exact_case_ids(PACKET_SPECS[-1].packet_dir)
    helpful_normalization_count = sum(
        1 for row in full_target_interventions if row["case_id"] in h2j_exact_case_ids
    )
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2e_exact_success_count": h2e["exact_success_count"],
        "h2e_executor_success_count": h2e["executor_success_count"],
        "h2j_no_stale_exact_success_count": h2j_no_stale["exact_success_count"],
        "h2j_no_stale_executor_success_count": h2j_no_stale["executor_success_count"],
        "h2j_exact_success_count": h2j["exact_success_count"],
        "h2j_executor_success_count": h2j["executor_success_count"],
        "h2j_delta_exact_vs_h2e": h2j_vs_h2e["delta_exact_rate"],
        "h2j_delta_executor_vs_h2e": h2j_vs_h2e["delta_executor_equivalence_rate"],
        "h2j_delta_exact_vs_no_stale_gate": h2j_vs_no_stale["delta_exact_rate"],
        "h2j_delta_executor_vs_no_stale_gate": h2j_vs_no_stale["delta_executor_equivalence_rate"],
        "h2e_non_exact_count": sum(1 for row in non_exact_rows if row["profile_label"] == "h2e_route_arbitration"),
        "h2j_no_stale_non_exact_count": sum(
            1
            for row in non_exact_rows
            if row["profile_label"] == "h2j_target_query_normalization_no_stale_gate"
        ),
        "h2j_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"] == "h2j_target_query_normalization"
        ),
        "target_query_normalization_count": len(full_target_interventions),
        "visual_stale_selection_gate_count": len(full_stale_interventions),
        "h2j_no_stale_target_query_normalization_count": len(no_stale_target_interventions),
        "h2j_no_stale_visual_stale_selection_gate_count": len(no_stale_stale_interventions),
        "h2j_helpful_target_query_normalization_count": helpful_normalization_count,
        "h2j_value_bearing_overstrip_count": len(overstrip_rows),
        "promotion_decision": "h2m_rejects_current_target_normalization_scope_under_less_direct_wording",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "family_rows": family_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "overstrip_rows": overstrip_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2m_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2m_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2m_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2m_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2m_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2m_overstrip_rows.csv", overstrip_rows)
    _write_csv(tables_dir / "h2m_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2m_less_direct_overreach_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _family_rows() -> list[dict[str, Any]]:
    packet_dir = ROOT / "results" / "tool_probe_replay_packets" / "20260512T_h2m_less_direct_target_normalization_overreach_dry_run_v1"
    replay_cases = _read_json(packet_dir / "replay_cases.json")
    rows: list[dict[str, Any]] = []
    family_to_targets: dict[str, list[str]] = {}
    for case in replay_cases:
        expected = (case.get("expected_calls") or [{}])[0]
        target = str(expected.get("arguments", {}).get("target_query", ""))
        family_to_targets.setdefault(case["family"], []).append(target)
    for family, targets in sorted(family_to_targets.items()):
        rows.append(
            {
                "family": family,
                "case_count": len(targets),
                "expected_target_queries": "; ".join(targets),
            }
        )
    return rows


def _overstrip_rows(
    intervention_rows: list[dict[str, Any]], non_exact_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    non_exact_by_case = {
        row["case_id"]: row
        for row in non_exact_rows
        if row["profile_label"] == "h2j_target_query_normalization"
    }
    rows: list[dict[str, Any]] = []
    for intervention in intervention_rows:
        if intervention["profile_label"] != "h2j_target_query_normalization":
            continue
        if intervention["intervention_kind"] != "visual_target_query_normalization":
            continue
        if intervention["family"] != "h2m_less_direct_value_bearing_target":
            continue
        miss = non_exact_by_case.get(intervention["case_id"])
        if not miss:
            continue
        rows.append(
            {
                "case_id": intervention["case_id"],
                "family": intervention["family"],
                "expected_target_query": miss["expected_target_query"],
                "actual_target_query": miss["actual_target_query"],
                "from_arguments": intervention["from_arguments"],
                "to_arguments": intervention["to_arguments"],
                "prompt_state_label": intervention["prompt_state_label"],
            }
        )
    return rows


def _exact_case_ids(packet_dir: Path) -> set[str]:
    return {
        row["case_id"]
        for row in _read_json(packet_dir / "live_replay_results.json")
        if row.get("replay_exact_match") is True
    }


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    overstrip_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2e = _packet_by_profile(packet_rows, "h2e_route_arbitration")
    h2j_no_stale = _packet_by_profile(packet_rows, "h2j_target_query_normalization_no_stale_gate")
    h2j = _packet_by_profile(packet_rows, "h2j_target_query_normalization")
    h2j_vs_h2e = _comparison_by_label(comparison_rows, "h2j_vs_h2e")
    h2j_vs_no_stale = _comparison_by_label(comparison_rows, "h2j_vs_no_stale_gate")
    full_target_count = len(
        _interventions_for(
            intervention_rows,
            profile_label="h2j_target_query_normalization",
            intervention_kind="visual_target_query_normalization",
        )
    )
    return [
        {
            "finding_id": "h2m_breaks_h2l_saturation",
            "finding": (
                f"H2m breaks the H2l saturation: H2j reaches {h2j['exact_success_count']}/8 exact and "
                f"executor-equivalent, H2j-no-stale also reaches {h2j_no_stale['exact_success_count']}/8, "
                f"and H2e reaches {h2e['exact_success_count']}/8 exact."
            ),
        },
        {
            "finding_id": "h2m_target_normalization_is_mixed",
            "finding": (
                f"H2j improves exact-rate over H2e by {h2j_vs_h2e['delta_exact_rate']} but does not improve "
                f"executor-equivalence over H2e ({h2j_vs_h2e['delta_executor_equivalence_rate']}); it ties the "
                f"no-stale ablation with {h2j_vs_no_stale['delta_exact_rate']} exact-rate delta."
            ),
        },
        {
            "finding_id": "h2m_exposes_overstrip",
            "finding": (
                f"H2j records {full_target_count} target-query-normalization interventions, but {len(overstrip_rows)} "
                "of them over-strip less-direct value-bearing targets to shorter component labels."
            ),
        },
        {
            "finding_id": "next_gate_should_scope_normalization",
            "finding": (
                "The next H2n move should make target-query normalization conditional on evidence that the shorter "
                "component label is explicitly requested, while preserving the H2k/H2l regression-guard repairs."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2m Less-Direct Target-Normalization Overreach Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2m removes H2l's explicit target-is wording while preserving the same value-bearing, alias, and "
            "regression-guard families. It breaks the H2l saturation: full H2j and H2j without stale-selection "
            "both fall to 3/8 strict and executor-equivalent. H2e reaches 1/8 strict and 3/8 executor-equivalent. "
            "The mechanism is mixed: H2j still repairs some contextual labels, but it also over-strips less-direct "
            "value-bearing labels such as `result badge Blocked` and `state tag Closed` into shorter component labels."
        ),
        "",
        "![H2m less-direct overreach gate](figures/h2m_less_direct_overreach_gate.svg)",
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
        "## Overstrip Rows",
        "",
        _table(payload["overstrip_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    profiles = [
        ("h2e_route_arbitration", "H2e", "#0891B2"),
        ("h2j_target_query_normalization_no_stale_gate", "H2j-no-stale", "#2563EB"),
        ("h2j_target_query_normalization", "H2j", "#1D4ED8"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 720
    height = 330
    chart_left = 94
    chart_top = 58
    chart_height = 190
    bar_width = 92
    gap = 54
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2m less-direct overreach gate</title>',
        '<desc id="desc">H2m breaks H2l saturation; H2j rows reach three of eight exact successes.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2m breaks less-direct target-normalization</text>',
        '<line x1="94" y1="248" x2="600" y2="248" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="88" y1="{y:.1f}" x2="600" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
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
            f'<text x="{x + 4}" y="276" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2m less-direct overreach synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2m_less_direct_overreach_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
