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
    _intervention_count,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h3b_saturation_breaker_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h3b_h2w_semantic_target_preservation",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h3b_saturation_breaker_h2w_execute_v1",
    ),
    PacketSpec(
        "h3b_h2z_boundary_combined",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h3b_saturation_breaker_h2z_execute_v1",
    ),
    PacketSpec(
        "h3b_h3a_boundary_combined",
        ROOT / "results" / "tool_probe_replay_live" / "20260519T_h3b_saturation_breaker_h3a_execute_v2",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h3b_h2z_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h3b_saturation_breaker_h2z_vs_h2w_v1",
    ),
    ComparisonSpec(
        "h3b_h3a_vs_h2z",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h3b_saturation_breaker_h3a_vs_h2z_v1",
    ),
    ComparisonSpec(
        "h3b_h3a_vs_h2w",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260519T_h3b_saturation_breaker_h3a_vs_h2w_v1",
    ),
)


def build_h3b_saturation_breaker_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    failure_taxonomy_rows = _failure_taxonomy_rows(PACKET_SPECS)
    case_matrix_rows = _case_matrix_rows(PACKET_SPECS)
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        family_rows=family_rows,
        failure_taxonomy_rows=failure_taxonomy_rows,
        intervention_rows=intervention_rows,
    )

    h2w = _packet_by_profile(packet_rows, "h3b_h2w_semantic_target_preservation")
    h2z = _packet_by_profile(packet_rows, "h3b_h2z_boundary_combined")
    h3a = _packet_by_profile(packet_rows, "h3b_h3a_boundary_combined")
    h2z_vs_h2w = _comparison_by_label(comparison_rows, "h3b_h2z_vs_h2w")
    h3a_vs_h2z = _comparison_by_label(comparison_rows, "h3b_h3a_vs_h2z")
    h3a_vs_h2w = _comparison_by_label(comparison_rows, "h3b_h3a_vs_h2w")
    h3a_failures = [row for row in failure_taxonomy_rows if row["profile_label"] == "h3b_h3a_boundary_combined"]
    h3a_unexpected_tool_calls = _failure_count(h3a_failures, "unexpected_tool_call")
    h3a_wrong_tool_count = _failure_count(h3a_failures, "wrong_tool")
    h3a_argument_mismatch_count = _failure_count(h3a_failures, "argument_mismatch")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "phase": "h3b_saturation_breaker_synthesis",
        "source_packet": "results/tool_probe_replay_packets/20260519T_h3b_saturation_breaker_dry_run_v1",
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h3b_case_count": h3a["case_count"],
        "h2w_exact_success_count": h2w["exact_success_count"],
        "h2w_executor_success_count": h2w["executor_success_count"],
        "h2z_exact_success_count": h2z["exact_success_count"],
        "h2z_executor_success_count": h2z["executor_success_count"],
        "h3a_exact_success_count": h3a["exact_success_count"],
        "h3a_executor_success_count": h3a["executor_success_count"],
        "h2z_delta_exact_vs_h2w": h2z_vs_h2w["delta_exact_rate"],
        "h2z_delta_executor_vs_h2w": h2z_vs_h2w["delta_executor_equivalence_rate"],
        "h3a_delta_exact_vs_h2z": h3a_vs_h2z["delta_exact_rate"],
        "h3a_delta_executor_vs_h2z": h3a_vs_h2z["delta_executor_equivalence_rate"],
        "h3a_delta_exact_vs_h2w": h3a_vs_h2w["delta_exact_rate"],
        "h3a_delta_executor_vs_h2w": h3a_vs_h2w["delta_executor_equivalence_rate"],
        "h3a_unexpected_tool_call_count": h3a_unexpected_tool_calls,
        "h3a_wrong_tool_count": h3a_wrong_tool_count,
        "h3a_argument_mismatch_count": h3a_argument_mismatch_count,
        "h3a_controller_intervention_count": sum(
            1 for row in intervention_rows if row["profile_label"] == "h3b_h3a_boundary_combined"
        ),
        "current_ladder_zero_delta": all(
            row["delta_exact_rate"] == 0.0 and row["delta_executor_equivalence_rate"] == 0.0
            for row in comparison_rows
        ),
        "h3b_breaks_current_ladder": h3a["exact_success_count"] < h3a["case_count"]
        and h3a["executor_success_count"] < h3a["case_count"],
        "promotion_decision": "do_not_add_old_helper_tuning; design H4 approval/latest-instruction and H3c semantic-generalization controls",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "family_rows": family_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "failure_taxonomy_rows": failure_taxonomy_rows,
        "case_matrix_rows": case_matrix_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h3b_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h3b_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h3b_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h3b_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h3b_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h3b_failure_taxonomy.csv", failure_taxonomy_rows)
    _write_csv(tables_dir / "h3b_case_matrix.csv", case_matrix_rows)
    _write_csv(tables_dir / "h3b_findings.csv", finding_rows)
    _write_svg(figures_dir / "h3b_saturation_breaker_family_pressure.svg", family_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _failure_taxonomy_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        counts: dict[str, int] = {}
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            mode = str(result.get("replay_failure_mode", ""))
            counts[mode] = counts.get(mode, 0) + 1
        for mode, count in sorted(counts.items()):
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "failure_mode": mode,
                    "case_count": count,
                    "share": count / sum(counts.values()) if counts else 0.0,
                }
            )
    return rows


def _case_matrix_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    by_profile: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        by_profile[spec.profile_label] = {
            str(row["case_id"]): row for row in _read_json(spec.packet_dir / "live_replay_results.json")
        }
    case_ids = sorted({case_id for rows in by_profile.values() for case_id in rows})
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        h3a = by_profile["h3b_h3a_boundary_combined"][case_id]
        row = {
            "case_id": case_id,
            "family": h3a.get("family", ""),
            "source_failure_mode": h3a.get("source_failure_mode", ""),
        }
        for profile, cases in by_profile.items():
            result = cases[case_id]
            prefix = profile.removeprefix("h3b_")
            row[f"{prefix}_exact"] = result.get("replay_exact_match")
            row[f"{prefix}_executor_equivalence"] = result.get("replay_executor_equivalence_match")
            row[f"{prefix}_failure_mode"] = result.get("replay_failure_mode", "")
        rows.append(row)
    return rows


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    failure_taxonomy_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2w = _packet_by_profile(packet_rows, "h3b_h2w_semantic_target_preservation")
    h2z = _packet_by_profile(packet_rows, "h3b_h2z_boundary_combined")
    h3a = _packet_by_profile(packet_rows, "h3b_h3a_boundary_combined")
    h3a_vs_h2w = _comparison_by_label(comparison_rows, "h3b_h3a_vs_h2w")
    h3a_families = [row for row in family_rows if row["profile_label"] == "h3b_h3a_boundary_combined"]
    family_summary = "; ".join(
        f"{row['family']} {row['exact_success_count']}/{row['case_count']} strict, "
        f"{row['executor_success_count']}/{row['case_count']} executor"
        for row in h3a_families
    )
    h3a_failures = [row for row in failure_taxonomy_rows if row["profile_label"] == "h3b_h3a_boundary_combined"]
    unexpected_tool_calls = _failure_count(h3a_failures, "unexpected_tool_call")
    h3a_intervention_count = sum(
        1 for row in intervention_rows if row["profile_label"] == "h3b_h3a_boundary_combined"
    )
    h3a_negative_value_interventions = _intervention_count(
        [row for row in intervention_rows if row["profile_label"] == "h3b_h3a_boundary_combined"],
        "visual_negative_value_component_target_preservation",
    )
    return [
        {
            "finding_id": "h3b_breaks_h3a_saturation",
            "finding": (
                f"H3b drops the current H3a candidate to {h3a['exact_success_count']}/{h3a['case_count']} strict "
                f"and {h3a['executor_success_count']}/{h3a['case_count']} executor-equivalent, breaking the prior "
                "H3/H2y/back-compat top-line saturation."
            ),
        },
        {
            "finding_id": "current_controller_ladder_has_zero_h3b_delta",
            "finding": (
                f"H2w, H2z, and H3a all score {h2w['exact_success_count']}/{h2w['case_count']} strict and "
                f"{h2w['executor_success_count']}/{h2w['case_count']} executor-equivalent; H3a-vs-H2w delta is "
                f"{h3a_vs_h2w['delta_exact_rate']} exact and {h3a_vs_h2w['delta_executor_equivalence_rate']} executor."
            ),
        },
        {
            "finding_id": "h3b_family_surface_is_not_uniform",
            "finding": (
                "Family scores show where the new pressure lives: "
                f"{family_summary}."
            ),
        },
        {
            "finding_id": "approval_stop_is_a_true_live_operator_boundary",
            "finding": (
                f"The four approval-stop rows are now scored as {unexpected_tool_calls} unexpected-tool-call failures "
                "with zero executor credit, after replay-live was fixed to preserve serialized no-tool expectations."
            ),
        },
        {
            "finding_id": "old_h3a_helpers_do_not_explain_h3b",
            "finding": (
                f"H3a records {h3a_intervention_count} controller intervention rows on H3b, including "
                f"{h3a_negative_value_interventions} negative-value preservation interventions, but those do not "
                "move aggregate score versus H2w/H2z. The next helpers should target new mechanisms, not tune the old ladder."
            ),
        },
    ]


def _failure_count(rows: list[dict[str, Any]], failure_mode: str) -> int:
    return sum(int(row["case_count"]) for row in rows if row["failure_mode"] == failure_mode)


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H3b Saturation-Breaker Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H3b is the first executed saturation breaker after H3a passed H3, H2y transfer, and the broad H2w-era "
            "transfer/back-compat gate. It is deliberately closer to frontier agentic benchmark pressure: mixed "
            "workflow state, latest-instruction retargeting, negative-value generalization, and no-tool approval-stop "
            "contracts are scored through the CLI replay-live surface."
        ),
        "",
        (
            f"On the 24-case packet, H2w, H2z, and H3a all reach "
            f"`{manifest['h3a_exact_success_count']} / {manifest['h3b_case_count']}` strict and "
            f"`{manifest['h3a_executor_success_count']} / {manifest['h3b_case_count']}` executor-equivalent. The "
            "zero-delta comparison is the key attribution result: the current controller ladder does not solve H3b."
        ),
        "",
        (
            "The most important correction in this slice is that approval-stop rows now preserve serialized no-tool "
            "expectations. Those four cases are `unexpected_tool_call` failures, not executor-equivalent paraphrases."
        ),
        "",
        "![H3b saturation-breaker family pressure](figures/h3b_saturation_breaker_family_pressure.svg)",
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
        "## Failure Taxonomy",
        "",
        _table(payload["failure_taxonomy_rows"]),
        "",
        "## Non-Exact Rows",
        "",
        _table(payload["non_exact_rows"]),
        "",
        "## Controller Intervention Rows",
        "",
        _table(payload["intervention_rows"]),
        "",
        "## Case Matrix",
        "",
        _table(payload["case_matrix_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _write_svg(path: Path, family_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in family_rows if row["profile_label"] == "h3b_h3a_boundary_combined"]
    width = 1080
    height = 420
    chart_left = 84
    chart_top = 72
    chart_height = 220
    group_width = 150
    bar_width = 38
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H3b saturation-breaker family pressure</title>',
        '<desc id="desc">H3a passes stale-origin paraphrase rows but fails approval-stop rows and only partially passes negative-value and latest-instruction rows.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="36" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H3b breaks current controller saturation by family</text>',
        '<line x1="84" y1="292" x2="1010" y2="292" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="78" y1="{y:.1f}" x2="1010" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="38" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, row in enumerate(rows):
        x = chart_left + index * group_width
        exact_height = float(row["exact_rate"]) * chart_height
        executor_height = float(row["executor_rate"]) * chart_height
        exact_y = chart_top + chart_height - exact_height
        executor_y = chart_top + chart_height - executor_height
        lines.append(f'<rect x="{x}" y="{exact_y:.1f}" width="{bar_width}" height="{exact_height:.1f}" fill="#C2410C"/>')
        lines.append(
            f'<rect x="{x + bar_width + 8}" y="{executor_y:.1f}" width="{bar_width}" height="{executor_height:.1f}" fill="#2563EB" opacity="0.65"/>'
        )
        lines.append(
            f'<text x="{x - 2}" y="{exact_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["exact_success_count"])}/4</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 6}" y="{executor_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{int(row["executor_success_count"])}/4</text>'
        )
        label = str(row["family"]).replace("h3b_", "").replace("h4_", "").replace("_", " ")
        lines.append(
            f'<text x="{x - 24}" y="322" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#111827">{label[:22]}</text>'
        )
        lines.append(
            f'<text x="{x - 24}" y="338" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#111827">{label[22:44]}</text>'
        )
    lines.extend(
        [
            '<rect x="780" y="365" width="18" height="12" fill="#C2410C"/>',
            '<text x="804" y="376" font-family="Arial, sans-serif" font-size="12" fill="#374151">strict exact</text>',
            '<rect x="900" y="365" width="18" height="12" fill="#2563EB" opacity="0.65"/>',
            '<text x="924" y="376" font-family="Arial, sans-serif" font-size="12" fill="#374151">executor-equivalent</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H3b saturation-breaker synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h3b_saturation_breaker_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
