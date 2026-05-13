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
from build_h2r_composed_route_gating_synthesis import (
    _controller_intervention_rows,
    _intervention_counts_for,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2t_overreach_independence_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2t_h2e_route_arbitration",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2t_overreach_independence_h2e_execute_v1",
    ),
    PacketSpec(
        "h2t_h2j_target_query_normalization",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2t_overreach_independence_h2j_execute_v1",
    ),
    PacketSpec(
        "h2t_h2o_value_bearing_synthesis",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2t_overreach_independence_h2o_execute_v1",
    ),
    PacketSpec(
        "h2t_h2p_contextual_surface_alias_routing",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2t_overreach_independence_h2p_execute_v1",
    ),
    PacketSpec(
        "h2t_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2t_overreach_independence_h2r_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2t_h2r_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2t_overreach_independence_h2r_vs_h2e_v1",
    ),
    ComparisonSpec(
        "h2t_h2r_vs_h2j",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2t_overreach_independence_h2r_vs_h2j_v1",
    ),
    ComparisonSpec(
        "h2t_h2r_vs_h2o",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2t_overreach_independence_h2r_vs_h2o_v1",
    ),
    ComparisonSpec(
        "h2t_h2r_vs_h2p",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2t_overreach_independence_h2r_vs_h2p_v1",
    ),
)


def build_h2t_overreach_independence_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    bad_normalization_rows = _bad_normalization_rows(PACKET_SPECS)
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        non_exact_rows=non_exact_rows,
        intervention_rows=intervention_rows,
        bad_normalization_rows=bad_normalization_rows,
    )

    h2e = _packet_by_profile(packet_rows, "h2t_h2e_route_arbitration")
    h2j = _packet_by_profile(packet_rows, "h2t_h2j_target_query_normalization")
    h2o = _packet_by_profile(packet_rows, "h2t_h2o_value_bearing_synthesis")
    h2p = _packet_by_profile(packet_rows, "h2t_h2p_contextual_surface_alias_routing")
    h2r = _packet_by_profile(packet_rows, "h2t_h2r_composed_route_gating")
    h2r_vs_h2e = _comparison_by_label(comparison_rows, "h2t_h2r_vs_h2e")
    h2r_vs_h2j = _comparison_by_label(comparison_rows, "h2t_h2r_vs_h2j")
    h2r_vs_h2o = _comparison_by_label(comparison_rows, "h2t_h2r_vs_h2o")
    h2r_vs_h2p = _comparison_by_label(comparison_rows, "h2t_h2r_vs_h2p")
    h2r_counts = _intervention_counts_for(intervention_rows, profile_label="h2t_h2r_composed_route_gating")
    h2r_bad_normalization_rows = [
        row for row in bad_normalization_rows if row["profile_label"] == "h2t_h2r_composed_route_gating"
    ]
    h2e_negation_exact = _exact_count_for_family(
        profile_label="h2t_h2e_route_arbitration",
        packet_specs=PACKET_SPECS,
        family="h2t_negation_scope_guard",
    )
    h2r_negation_exact = _exact_count_for_family(
        profile_label="h2t_h2r_composed_route_gating",
        packet_specs=PACKET_SPECS,
        family="h2t_negation_scope_guard",
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2t_case_count": int(h2r["case_count"]),
        "h2t_h2e_exact_success_count": h2e["exact_success_count"],
        "h2t_h2e_executor_success_count": h2e["executor_success_count"],
        "h2t_h2j_exact_success_count": h2j["exact_success_count"],
        "h2t_h2j_executor_success_count": h2j["executor_success_count"],
        "h2t_h2o_exact_success_count": h2o["exact_success_count"],
        "h2t_h2o_executor_success_count": h2o["executor_success_count"],
        "h2t_h2p_exact_success_count": h2p["exact_success_count"],
        "h2t_h2p_executor_success_count": h2p["executor_success_count"],
        "h2t_h2r_exact_success_count": h2r["exact_success_count"],
        "h2t_h2r_executor_success_count": h2r["executor_success_count"],
        "h2t_h2r_delta_exact_vs_h2e": h2r_vs_h2e["delta_exact_rate"],
        "h2t_h2r_delta_executor_vs_h2e": h2r_vs_h2e["delta_executor_equivalence_rate"],
        "h2t_h2r_delta_exact_vs_h2j": h2r_vs_h2j["delta_exact_rate"],
        "h2t_h2r_delta_exact_vs_h2o": h2r_vs_h2o["delta_exact_rate"],
        "h2t_h2r_delta_exact_vs_h2p": h2r_vs_h2p["delta_exact_rate"],
        "h2t_h2r_target_query_normalization_count": h2r_counts.get("visual_target_query_normalization", 0),
        "h2t_bad_normalization_count_all_profiles": len(bad_normalization_rows),
        "h2t_h2r_bad_normalization_count": len(h2r_bad_normalization_rows),
        "h2t_negation_scope_h2e_exact_count": h2e_negation_exact,
        "h2t_negation_scope_h2r_exact_count": h2r_negation_exact,
        "promotion_decision": "h2t_breaks_h2r_requires_h2u_negation_aware_normalization",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "family_rows": family_rows,
        "non_exact_rows": non_exact_rows,
        "intervention_rows": intervention_rows,
        "bad_normalization_rows": bad_normalization_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2t_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2t_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2t_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2t_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2t_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2t_bad_normalization_rows.csv", bad_normalization_rows)
    _write_csv(tables_dir / "h2t_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2t_overreach_independence_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _exact_count_for_family(
    *, profile_label: str, packet_specs: tuple[PacketSpec, ...], family: str
) -> int:
    spec = next(spec for spec in packet_specs if spec.profile_label == profile_label)
    return sum(
        1
        for row in _read_json(spec.packet_dir / "live_replay_results.json")
        if row.get("family") == family and row.get("replay_exact_match") is True
    )


def _bad_normalization_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            entries = metadata.get("visual_target_query_normalization", []) if isinstance(metadata, dict) else []
            if not isinstance(entries, list):
                continue
            expected_call = _first_call(probe.get("expected_calls"))
            raw_call = _raw_call(probe.get("raw_model_output"))
            actual_call = _first_call(probe.get("actual_calls"))
            if not expected_call or not raw_call or not actual_call:
                continue
            raw_was_exact = raw_call == expected_call
            final_is_exact = actual_call == expected_call
            if not raw_was_exact or final_is_exact:
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                rows.append(
                    {
                        "profile_label": spec.profile_label,
                        "case_id": result["case_id"],
                        "family": result.get("family", ""),
                        "expected_tool": expected_call.get("name", ""),
                        "expected_target_query": _target_query(expected_call),
                        "raw_target_query": _target_query(raw_call),
                        "actual_target_query": _target_query(actual_call),
                        "prompt_state_label": entry.get("prompt_state_label", ""),
                        "from_arguments": _compact_json(entry.get("from_arguments", {})),
                        "to_arguments": _compact_json(entry.get("to_arguments", {})),
                    }
                )
    return rows


def _first_call(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    return None


def _raw_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return _first_call(parsed)


def _target_query(call: dict[str, Any]) -> str:
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("target_query", ""))


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    bad_normalization_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2e = _packet_by_profile(packet_rows, "h2t_h2e_route_arbitration")
    h2j = _packet_by_profile(packet_rows, "h2t_h2j_target_query_normalization")
    h2o = _packet_by_profile(packet_rows, "h2t_h2o_value_bearing_synthesis")
    h2p = _packet_by_profile(packet_rows, "h2t_h2p_contextual_surface_alias_routing")
    h2r = _packet_by_profile(packet_rows, "h2t_h2r_composed_route_gating")
    h2r_vs_h2e = _comparison_by_label(comparison_rows, "h2t_h2r_vs_h2e")
    h2r_vs_h2p = _comparison_by_label(comparison_rows, "h2t_h2r_vs_h2p")
    h2r_counts = _intervention_counts_for(intervention_rows, profile_label="h2t_h2r_composed_route_gating")
    h2r_non_exact = [row for row in non_exact_rows if row["profile_label"] == "h2t_h2r_composed_route_gating"]
    h2r_bad_normalization_rows = [
        row for row in bad_normalization_rows if row["profile_label"] == "h2t_h2r_composed_route_gating"
    ]
    h2e_negation_misses = [
        row
        for row in non_exact_rows
        if row["profile_label"] == "h2t_h2e_route_arbitration"
        and row["family"] == "h2t_negation_scope_guard"
    ]
    h2r_negation_misses = [
        row
        for row in h2r_non_exact
        if row["family"] == "h2t_negation_scope_guard"
    ]
    return [
        {
            "finding_id": "h2t_breaks_h2r_topline_saturation",
            "finding": (
                f"H2r reaches {h2r['exact_success_count']}/{h2r['case_count']} strict and "
                f"{h2r['executor_success_count']}/{h2r['case_count']} executor-equivalent on H2t; "
                f"H2p, H2o, and H2j also reach {h2p['exact_success_count']}/10, "
                f"{h2o['exact_success_count']}/10, and {h2j['exact_success_count']}/10."
            ),
        },
        {
            "finding_id": "h2t_exposes_h2e_tradeoff",
            "finding": (
                f"H2e reaches {h2e['exact_success_count']}/10 strict and "
                f"{h2e['executor_success_count']}/10 executor-equivalent. H2r gains "
                f"{h2r_vs_h2e['delta_exact_rate']} exact-rate versus H2e but loses "
                f"{h2r_vs_h2e['delta_executor_equivalence_rate']} executor-equivalence-rate."
            ),
        },
        {
            "finding_id": "h2t_later_helpers_do_not_add_signal",
            "finding": (
                f"H2r ties H2p on H2t with delta {h2r_vs_h2p['delta_exact_rate']} exact-rate; the overreach "
                "signal is shared by target-query normalization and the later synthesis/routing stack."
            ),
        },
        {
            "finding_id": "h2t_bad_normalization_is_controller_induced",
            "finding": (
                f"H2r records {len(h2r_bad_normalization_rows)} rows where the raw model emitted the expected target "
                "but controller normalization rewrote it to a prompt-state label. The H2r non-exact rows are "
                f"{len(h2r_negation_misses)} negation-scope cases."
            ),
        },
        {
            "finding_id": "h2t_h2e_preserves_negation_scope",
            "finding": (
                f"H2e has {len(h2e_negation_misses)} negation-scope misses while H2r has "
                f"{len(h2r_negation_misses)}; this isolates the regression to the normalization helper rather "
                "than Gemma's raw local MLX call on those rows."
            ),
        },
        {
            "finding_id": "h2t_next_requires_h2u",
            "finding": (
                "H2t should promote an H2u intervention: target-query normalization must be negation-aware and "
                "must not rewrite an exact current-surface label to a note/caption label introduced only as "
                f"context. H2r used {h2r_counts.get('visual_target_query_normalization', 0)} target normalizations on H2t."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2t Overreach Independence Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2t is the first post-H2s holdout designed to break top-line saturation by separating helpful "
            "target normalization from overreach. It keeps low-score/value exactness pressure, but adds "
            "negation-scope rows where a note or caption names a decoy component that should not become the target."
        ),
        "",
        (
            f"H2r reaches `{manifest['h2t_h2r_exact_success_count']} / {manifest['h2t_case_count']}` strict and "
            f"`{manifest['h2t_h2r_executor_success_count']} / {manifest['h2t_case_count']}` executor-equivalent. "
            "H2p, H2o, and H2j tie that score, while H2e reaches `6 / 10` strict but `9 / 10` "
            "executor-equivalent. The H2e/H2r split is the important result: H2r preserves more literal exactness "
            "on low-score/value cases, but H2e avoids the negation-scope controller rewrite."
        ),
        "",
        (
            f"There are `{manifest['h2t_h2r_bad_normalization_count']}` H2r rows where raw MLX Gemma emitted the "
            "expected target and the controller rewrote it to a prompt-state label. H2u should therefore patch the "
            "controller, not the model prompt: normalization needs a negation-aware guard."
        ),
        "",
        "![H2t overreach independence gate](figures/h2t_overreach_independence_gate.svg)",
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
        "## Bad Normalization Rows",
        "",
        _table(payload["bad_normalization_rows"]),
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
        ("h2t_h2e_route_arbitration", "H2e", "#1D4ED8"),
        ("h2t_h2j_target_query_normalization", "H2j", "#0F766E"),
        ("h2t_h2o_value_bearing_synthesis", "H2o", "#115E59"),
        ("h2t_h2p_contextual_surface_alias_routing", "H2p", "#B45309"),
        ("h2t_h2r_composed_route_gating", "H2r", "#7C2D12"),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 880
    height = 390
    chart_left = 92
    chart_top = 78
    chart_height = 196
    bar_width = 58
    group_gap = 46
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2t overreach independence gate</title>',
        '<desc id="desc">H2r, H2p, H2o, and H2j tie at exact rate 0.8, while H2e reaches exact rate 0.6 and executor-equivalence rate 0.9.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2t breaks H2r by exposing normalization overreach</text>',
        '<line x1="92" y1="274" x2="810" y2="274" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="86" y1="{y:.1f}" x2="810" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="46" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, (profile, label, color) in enumerate(profiles):
        row = by_profile[profile]
        x = chart_left + index * ((bar_width * 2) + group_gap)
        exact_height = float(row["exact_rate"]) * chart_height
        executor_height = float(row["executor_rate"]) * chart_height
        exact_y = chart_top + chart_height - exact_height
        executor_y = chart_top + chart_height - executor_height
        lines.append(f'<rect x="{x}" y="{exact_y:.1f}" width="{bar_width}" height="{exact_height:.1f}" fill="{color}"/>')
        lines.append(
            f'<rect x="{x + bar_width + 4}" y="{executor_y:.1f}" width="{bar_width}" height="{executor_height:.1f}" fill="#9CA3AF"/>'
        )
        lines.append(
            f'<text x="{x + 7}" y="{exact_y - 8:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#111827">{int(row["exact_success_count"])}/10</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 12}" y="{executor_y - 8:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#111827">{int(row["executor_success_count"])}/10</text>'
        )
        lines.append(
            f'<text x="{x + 40}" y="306" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{label}</text>'
        )
    lines.append('<rect x="548" y="328" width="12" height="12" fill="#7C2D12"/>')
    lines.append('<text x="568" y="339" font-family="Arial, sans-serif" font-size="12" fill="#374151">strict exactness</text>')
    lines.append('<rect x="672" y="328" width="12" height="12" fill="#9CA3AF"/>')
    lines.append('<text x="692" y="339" font-family="Arial, sans-serif" font-size="12" fill="#374151">executor-equivalence</text>')
    lines.append(
        '<text x="32" y="362" font-family="Arial, sans-serif" font-size="13" fill="#374151">H2t shows a controller tradeoff: later helpers keep low-score exactness but over-normalize note/caption negation rows.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2t overreach independence synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2t_overreach_independence_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
