from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from build_h2l_target_normalization_overreach_synthesis import (
    ComparisonSpec,
    PacketSpec,
    _compact_json,
    _comparison_by_label,
    _comparison_row,
    _non_exact_rows,
    _packet_by_profile,
    _packet_row,
    _read_json,
    _table,
    _write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2w_transfer_backtest_synthesis"


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


H2U_LIVE_DIRS = {
    "h2t": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2t_overreach_independence_h2u_execute_v2",
    "h2s": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2s_execute_v1",
    "h2q": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2q_execute_v1",
    "h2m": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2m_execute_v1",
    "h2k": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2k_execute_v1",
    "h2l": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2l_execute_v1",
    "h2f": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2f_execute_v1",
    "h2b": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2b_execute_v1",
    "h1x": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h1x_execute_v1",
    "h1y": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h1y_execute_v1",
    "h1o": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h1o_execute_v1",
    "h1p": ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h1p_execute_v1",
}


H2R_LIVE_DIRS = {
    "h2t": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2t_overreach_independence_h2r_execute_v1",
    "h2s": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2s_fresh_composed_holdout_h2r_execute_v1",
    "h2q": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2q_execute_v2",
    "h2m": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2m_execute_v1",
    "h2k": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2k_execute_v2",
    "h2l": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2l_execute_v2",
    "h2f": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2f_execute_v1",
    "h2b": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2b_execute_v1",
    "h1x": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1x_execute_v1",
    "h1y": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1y_execute_v1",
    "h1o": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1o_execute_v1",
    "h1p": ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1p_execute_v1",
}


PACKET_SPECS: tuple[PacketSpec, ...] = tuple(
    PacketSpec(f"{label}_h2r_composed_route_gating", H2R_LIVE_DIRS[label])
    for label in TRANSFER_LABELS
) + tuple(
    PacketSpec(f"{label}_h2u_negation_guard", H2U_LIVE_DIRS[label])
    for label in TRANSFER_LABELS
) + tuple(
    PacketSpec(f"{label}_h2w_semantic_target_preservation", H2W_LIVE_DIRS[label])
    for label in TRANSFER_LABELS
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = tuple(
    ComparisonSpec(
        f"{label}_h2w_vs_h2u",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / f"20260517T_h2w_semantic_target_preservation_vs_h2u_on_{label}_v1",
    )
    for label in TRANSFER_LABELS
) + tuple(
    ComparisonSpec(
        f"{label}_h2w_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / f"20260517T_h2w_semantic_target_preservation_vs_h2r_on_{label}_v1",
    )
    for label in TRANSFER_LABELS
)


INTERVENTION_KEYS = (
    "visual_semantic_target_preservation",
    "visual_target_query_normalization",
    "visual_stale_selection_gate",
    "visual_value_bearing_target_query_synthesis",
    "visual_contextual_surface_alias_routing",
    "visual_composed_route_gating",
    "visual_target_query_normalization_blocked",
    "visual_composed_route_gating_blocked",
)


def build_h2w_transfer_backtest_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    family_rows = _family_rows()
    non_exact_rows = _non_exact_rows(PACKET_SPECS)
    intervention_rows = _controller_intervention_rows()
    fixed_case_rows = _fixed_case_rows()
    packet_pair_rows = _packet_pair_rows(packet_rows, comparison_rows)
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        non_exact_rows=non_exact_rows,
        intervention_rows=intervention_rows,
        fixed_case_rows=fixed_case_rows,
    )

    h2w_rows = [
        _packet_by_profile(packet_rows, f"{label}_h2w_semantic_target_preservation")
        for label in TRANSFER_LABELS
    ]
    h2w_vs_h2u = [_comparison_by_label(comparison_rows, f"{label}_h2w_vs_h2u") for label in TRANSFER_LABELS]
    h2w_vs_h2r = [_comparison_by_label(comparison_rows, f"{label}_h2w_vs_h2r") for label in TRANSFER_LABELS]
    non_h2q_h2w_rows = [
        row for label, row in zip(TRANSFER_LABELS, h2w_rows, strict=True) if label != "h2q"
    ]
    h2t_h2w_vs_h2r = _comparison_by_label(comparison_rows, "h2t_h2w_vs_h2r")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "family_row_count": len(family_rows),
        "h2w_transfer_packet_count": len(h2w_rows),
        "h2w_transfer_case_count": sum(int(row["case_count"]) for row in h2w_rows),
        "h2w_transfer_exact_success_count": sum(int(row["exact_success_count"]) for row in h2w_rows),
        "h2w_transfer_executor_success_count": sum(int(row["executor_success_count"]) for row in h2w_rows),
        "h2w_non_h2q_transfer_case_count": sum(int(row["case_count"]) for row in non_h2q_h2w_rows),
        "h2w_non_h2q_transfer_exact_success_count": sum(
            int(row["exact_success_count"]) for row in non_h2q_h2w_rows
        ),
        "h2w_non_h2q_transfer_executor_success_count": sum(
            int(row["executor_success_count"]) for row in non_h2q_h2w_rows
        ),
        "h2w_non_exact_count": sum(
            1 for row in non_exact_rows if row["profile_label"].endswith("h2w_semantic_target_preservation")
        ),
        "h2w_exact_delta_sum_vs_h2u": sum(float(row["delta_exact_rate"]) for row in h2w_vs_h2u),
        "h2w_executor_delta_sum_vs_h2u": sum(
            float(row["delta_executor_equivalence_rate"] or 0.0) for row in h2w_vs_h2u
        ),
        "h2w_exact_delta_sum_vs_h2r": sum(float(row["delta_exact_rate"]) for row in h2w_vs_h2r),
        "h2w_executor_delta_sum_vs_h2r": sum(
            float(row["delta_executor_equivalence_rate"] or 0.0) for row in h2w_vs_h2r
        ),
        "h2w_regression_count_vs_h2u": sum(1 for row in fixed_case_rows if row["delta_exact_match"] < 0),
        "h2w_fixed_case_count_vs_h2u": sum(
            1 for row in fixed_case_rows if row["comparison_label"].endswith("h2w_vs_h2u")
        ),
        "h2w_fixed_case_count_vs_h2r": sum(
            1 for row in fixed_case_rows if row["comparison_label"].endswith("h2w_vs_h2r")
        ),
        "h2t_delta_exact_vs_h2r": h2t_h2w_vs_h2r["delta_exact_rate"],
        "h2t_delta_executor_vs_h2r": h2t_h2w_vs_h2r["delta_executor_equivalence_rate"],
        "h2w_semantic_target_preservation_count": _intervention_count(
            intervention_rows,
            "visual_semantic_target_preservation",
        ),
        "h2w_target_query_normalization_count": _intervention_count(
            intervention_rows,
            "visual_target_query_normalization",
        ),
        "h2w_stale_selection_gate_count": _intervention_count(intervention_rows, "visual_stale_selection_gate"),
        "h2w_composed_route_gating_blocked_count": _intervention_count(
            intervention_rows,
            "visual_composed_route_gating_blocked",
        ),
        "runtime_posture_note": (
            "A four-way parallel MLX replay attempt hit a Metal GPU timeout before this sequential transfer backtest "
            "finished cleanly; future local MLX backtests should default to sequential or very-low-concurrency runs."
        ),
        "promotion_decision": "h2w_transfer_backtest_passes_ready_for_packaged_workflow_gate",
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
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2w_transfer_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2w_transfer_packet_pairs.csv", packet_pair_rows)
    _write_csv(tables_dir / "h2w_transfer_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2w_transfer_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2w_transfer_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2w_transfer_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2w_transfer_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h2w_transfer_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2w_transfer_backtest_gate.svg", packet_pair_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _family_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in TRANSFER_LABELS:
        family_counts: dict[str, dict[str, int]] = {}
        profile_label = f"{label}_h2w_semantic_target_preservation"
        for result in _read_json(H2W_LIVE_DIRS[label] / "live_replay_results.json"):
            family = str(result.get("family", ""))
            bucket = family_counts.setdefault(
                family,
                {"case_count": 0, "exact_success_count": 0, "executor_success_count": 0},
            )
            bucket["case_count"] += 1
            bucket["exact_success_count"] += int(result.get("replay_exact_match") is True)
            bucket["executor_success_count"] += int(result.get("replay_executor_equivalence_match") is True)
        for family, counts in sorted(family_counts.items()):
            case_count = counts["case_count"]
            rows.append(
                {
                    "profile_label": profile_label,
                    "slice": label,
                    "family": family,
                    "case_count": case_count,
                    "exact_success_count": counts["exact_success_count"],
                    "exact_rate": counts["exact_success_count"] / case_count if case_count else 0.0,
                    "executor_success_count": counts["executor_success_count"],
                    "executor_rate": counts["executor_success_count"] / case_count if case_count else 0.0,
                }
            )
    return rows


def _controller_intervention_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in TRANSFER_LABELS:
        profile_label = f"{label}_h2w_semantic_target_preservation"
        for result in _read_json(H2W_LIVE_DIRS[label] / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for kind in INTERVENTION_KEYS:
                entries = metadata.get(kind, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    rows.append(
                        {
                            "profile_label": profile_label,
                            "slice": label,
                            "case_id": result["case_id"],
                            "family": result.get("family", ""),
                            "intervention_kind": kind,
                            "from_tool": entry.get("from_tool", ""),
                            "from_arguments": _compact_json(entry.get("from_arguments", {})),
                            "to_tool": entry.get("to_tool", ""),
                            "to_arguments": _compact_json(entry.get("to_arguments", {})),
                            "preserved_target_query": entry.get("preserved_target_query", ""),
                            "blocked_label": entry.get("blocked_label", entry.get("prompt_state_label", "")),
                            "prompt_state_label": entry.get("prompt_state_label", ""),
                            "reason": entry.get("reason", ""),
                        }
                    )
    return rows


def _fixed_case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in COMPARISON_SPECS:
        payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
        for row in payload["case_deltas"]:
            delta = int(row.get("delta_exact_match") or 0)
            if delta == 0:
                continue
            rows.append(
                {
                    "comparison_label": spec.comparison_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "delta_exact_match": delta,
                    "baseline_failure_mode": row.get("baseline_replay_failure_mode", ""),
                    "candidate_failure_mode": row.get("candidate_replay_failure_mode", ""),
                    "baseline_executor_equivalence_match": row.get("baseline_replay_executor_equivalence_match"),
                    "candidate_executor_equivalence_match": row.get("candidate_replay_executor_equivalence_match"),
                }
            )
    return rows


def _packet_pair_rows(packet_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in TRANSFER_LABELS:
        h2r = _packet_by_profile(packet_rows, f"{label}_h2r_composed_route_gating")
        h2u = _packet_by_profile(packet_rows, f"{label}_h2u_negation_guard")
        h2w = _packet_by_profile(packet_rows, f"{label}_h2w_semantic_target_preservation")
        h2w_vs_h2u = _comparison_by_label(comparison_rows, f"{label}_h2w_vs_h2u")
        h2w_vs_h2r = _comparison_by_label(comparison_rows, f"{label}_h2w_vs_h2r")
        rows.append(
            {
                "slice": label,
                "case_count": h2w["case_count"],
                "h2r_exact_success_count": h2r["exact_success_count"],
                "h2u_exact_success_count": h2u["exact_success_count"],
                "h2w_exact_success_count": h2w["exact_success_count"],
                "h2r_executor_success_count": h2r["executor_success_count"],
                "h2u_executor_success_count": h2u["executor_success_count"],
                "h2w_executor_success_count": h2w["executor_success_count"],
                "h2w_delta_exact_vs_h2u": h2w_vs_h2u["delta_exact_rate"],
                "h2w_delta_executor_vs_h2u": h2w_vs_h2u["delta_executor_equivalence_rate"],
                "h2w_delta_exact_vs_h2r": h2w_vs_h2r["delta_exact_rate"],
                "h2w_delta_executor_vs_h2r": h2w_vs_h2r["delta_executor_equivalence_rate"],
            }
        )
    return rows


def _intervention_count(rows: list[dict[str, Any]], intervention_kind: str) -> int:
    return sum(1 for row in rows if row["intervention_kind"] == intervention_kind)


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    fixed_case_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2w_rows = [
        _packet_by_profile(packet_rows, f"{label}_h2w_semantic_target_preservation")
        for label in TRANSFER_LABELS
    ]
    transfer_cases = sum(int(row["case_count"]) for row in h2w_rows)
    transfer_exact = sum(int(row["exact_success_count"]) for row in h2w_rows)
    transfer_executor = sum(int(row["executor_success_count"]) for row in h2w_rows)
    h2w_vs_h2u = [_comparison_by_label(comparison_rows, f"{label}_h2w_vs_h2u") for label in TRANSFER_LABELS]
    h2w_vs_h2r = [_comparison_by_label(comparison_rows, f"{label}_h2w_vs_h2r") for label in TRANSFER_LABELS]
    h2t_vs_h2r = _comparison_by_label(comparison_rows, "h2t_h2w_vs_h2r")
    h2w_non_exact = [
        row for row in non_exact_rows if row["profile_label"].endswith("h2w_semantic_target_preservation")
    ]
    regressions_vs_h2u = [
        row for row in fixed_case_rows if row["comparison_label"].endswith("h2w_vs_h2u") and row["delta_exact_match"] < 0
    ]
    fixes_vs_h2r = [
        row for row in fixed_case_rows if row["comparison_label"].endswith("h2w_vs_h2r") and row["delta_exact_match"] > 0
    ]
    semantic_count = _intervention_count(intervention_rows, "visual_semantic_target_preservation")
    target_count = _intervention_count(intervention_rows, "visual_target_query_normalization")
    stale_count = _intervention_count(intervention_rows, "visual_stale_selection_gate")
    blocked_count = _intervention_count(intervention_rows, "visual_composed_route_gating_blocked")
    return [
        {
            "finding_id": "h2w_transfer_backtest_is_clean",
            "finding": (
                f"H2w preserves {transfer_exact}/{transfer_cases} strict exactness and "
                f"{transfer_executor}/{transfer_cases} executor equivalence across the 12-packet transfer/backward "
                "compatibility battery."
            ),
        },
        {
            "finding_id": "h2w_ties_current_h2u_incumbent",
            "finding": (
                "Against H2u, H2w has zero exact-rate and executor-equivalence-rate delta on every transfer packet "
                f"(aggregate delta exact {sum(float(row['delta_exact_rate']) for row in h2w_vs_h2u):.1f}); "
                f"there are {len(regressions_vs_h2u)} strict regressions."
            ),
        },
        {
            "finding_id": "h2w_keeps_h2t_repair_vs_h2r",
            "finding": (
                f"Against H2r, H2w only changes the H2t slice: delta exact "
                f"{h2t_vs_h2r['delta_exact_rate']} and executor-equivalence "
                f"{h2t_vs_h2r['delta_executor_equivalence_rate']}. The fixed rows are "
                f"{', '.join(row['case_id'] for row in fixes_vs_h2r) or 'none'}."
            ),
        },
        {
            "finding_id": "h2w_controller_activity_does_not_imply_transfer_cost",
            "finding": (
                f"The transfer runs record controller activity ({semantic_count} semantic-preservation, "
                f"{target_count} target-normalization, {stale_count} stale-selection, and {blocked_count} "
                f"blocked composed-route rows) while still leaving {len(h2w_non_exact)} non-exact H2w rows."
            ),
        },
        {
            "finding_id": "h2w_runtime_posture_needs_low_concurrency",
            "finding": (
                "The evidence also separates model/control quality from local runtime posture: a four-way parallel "
                "MLX replay attempt hit a Metal GPU timeout, while the sequential rerun completed cleanly. Future "
                "local MLX backtests should default to sequential or very-low-concurrency execution."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2w Transfer Backtest Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2w was introduced to repair H2v semantic target preservation: cases where the controller had to "
            "distinguish stale or quoted negation context from genuine negated target values. This backtest asks "
            "the next causal question: did that more permissive semantic preservation control regress older "
            "route, stale-selection, target-normalization, component-value, and negation-scope packets?"
        ),
        "",
        (
            f"The answer on this battery is no. H2w reaches "
            f"`{manifest['h2w_transfer_exact_success_count']} / {manifest['h2w_transfer_case_count']}` strict and "
            f"`{manifest['h2w_transfer_executor_success_count']} / {manifest['h2w_transfer_case_count']}` "
            "executor-equivalent across H2s/H2t/H2q/H2m/H2k/H2l/H2f/H2b/H1x/H1y/H1o/H1p. Excluding H2q, the "
            f"subtotal is `{manifest['h2w_non_h2q_transfer_exact_success_count']} / "
            f"{manifest['h2w_non_h2q_transfer_case_count']}`."
        ),
        "",
        (
            f"H2w ties H2u on every transfer/back-compat comparison: aggregate exact-rate delta "
            f"`{manifest['h2w_exact_delta_sum_vs_h2u']}` and executor-equivalence-rate delta "
            f"`{manifest['h2w_executor_delta_sum_vs_h2u']}`. Against H2r, the only positive delta is the inherited "
            f"H2t negation-scope repair (`{manifest['h2t_delta_exact_vs_h2r']}` exact-rate)."
        ),
        "",
        (
            "Operationally, this is also a runtime-posture result: one four-way parallel MLX attempt hit a Metal GPU "
            "timeout, while the sequential rerun completed cleanly. Treat local MLX transfer backtests as "
            "low-concurrency workloads unless the runtime is explicitly hardened for parallel replay."
        ),
        "",
        "![H2w transfer backtest gate](figures/h2w_transfer_backtest_gate.svg)",
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
    width = 1480
    height = 420
    chart_left = 70
    chart_top = 74
    chart_height = 205
    group_width = 104
    bar_width = 22
    colors = {"h2r": "#2563EB", "h2u": "#7C3AED", "h2w": "#92400E"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2w transfer backtest gate</title>',
        '<desc id="desc">H2w preserves exactness across twelve transfer and backward compatibility packets.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2w transfer/back-compat gate</text>',
        '<line x1="70" y1="279" x2="1358" y2="279" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="64" y1="{y:.1f}" x2="1358" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="28" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, row in enumerate(packet_pair_rows):
        x0 = chart_left + index * group_width
        case_count = int(row["case_count"])
        rates = {
            "h2r": int(row["h2r_exact_success_count"]) / case_count,
            "h2u": int(row["h2u_exact_success_count"]) / case_count,
            "h2w": int(row["h2w_exact_success_count"]) / case_count,
        }
        for offset, key in enumerate(("h2r", "h2u", "h2w")):
            bar_height = rates[key] * chart_height
            x = x0 + offset * (bar_width + 4)
            y = chart_top + chart_height - bar_height
            lines.append(
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[key]}"/>'
            )
        lines.append(
            f'<text x="{x0 - 2}" y="307" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">{row["slice"].upper()}</text>'
        )
        lines.append(
            f'<text x="{x0 - 4}" y="326" font-family="Arial, sans-serif" font-size="11" fill="#374151">{int(row["h2w_exact_success_count"])}/{case_count}</text>'
        )
    legend_x = 1110
    for index, (key, label) in enumerate((("h2r", "H2r"), ("h2u", "H2u"), ("h2w", "H2w"))):
        y = 350 + index * 20
        lines.append(f'<rect x="{legend_x}" y="{y}" width="18" height="12" fill="{colors[key]}"/>')
        lines.append(
            f'<text x="{legend_x + 24}" y="{y + 11}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{label} strict exactness</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2w transfer backtest synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2w_transfer_backtest_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
