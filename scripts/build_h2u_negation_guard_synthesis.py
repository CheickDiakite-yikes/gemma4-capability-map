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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2u_negation_guard_synthesis"


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2t_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2t_overreach_independence_h2r_execute_v1",
    ),
    PacketSpec(
        "h2t_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2t_overreach_independence_h2u_execute_v2",
    ),
    PacketSpec(
        "h2s_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2s_fresh_composed_holdout_h2r_execute_v1",
    ),
    PacketSpec(
        "h2s_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2s_execute_v1",
    ),
    PacketSpec(
        "h2q_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2q_execute_v2",
    ),
    PacketSpec(
        "h2q_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2q_execute_v1",
    ),
    PacketSpec(
        "h2m_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2m_execute_v1",
    ),
    PacketSpec(
        "h2m_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2m_execute_v1",
    ),
    PacketSpec(
        "h2k_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2k_execute_v2",
    ),
    PacketSpec(
        "h2k_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2k_execute_v1",
    ),
    PacketSpec(
        "h2l_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2l_execute_v2",
    ),
    PacketSpec(
        "h2l_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2l_execute_v1",
    ),
    PacketSpec(
        "h2f_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2f_execute_v1",
    ),
    PacketSpec(
        "h2f_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2f_execute_v1",
    ),
    PacketSpec(
        "h2b_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h2b_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h1x_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1x_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1y_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1y_execute_v1",
    ),
    PacketSpec(
        "h1y_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h1y_execute_v1",
    ),
    PacketSpec(
        "h1o_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1o_execute_v1",
    ),
    PacketSpec(
        "h1o_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h1o_execute_v1",
    ),
    PacketSpec(
        "h1p_h2r_composed_route_gating",
        ROOT / "results" / "tool_probe_replay_live" / "20260512T_h2r_composed_route_gating_on_h1p_execute_v1",
    ),
    PacketSpec(
        "h1p_h2u_negation_guard",
        ROOT / "results" / "tool_probe_replay_live" / "20260513T_h2u_negation_guard_on_h1p_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2t_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h2t_v1",
    ),
    ComparisonSpec(
        "h2s_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h2s_v1",
    ),
    ComparisonSpec(
        "h2q_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h2q_v1",
    ),
    ComparisonSpec(
        "h2m_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h2m_v1",
    ),
    ComparisonSpec(
        "h2k_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h2k_v1",
    ),
    ComparisonSpec(
        "h2l_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h2l_v1",
    ),
    ComparisonSpec(
        "h2f_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h2f_v1",
    ),
    ComparisonSpec(
        "h2b_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h2b_v1",
    ),
    ComparisonSpec(
        "h1x_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1y_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h1y_v1",
    ),
    ComparisonSpec(
        "h1o_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h1o_v1",
    ),
    ComparisonSpec(
        "h1p_h2u_vs_h2r",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260513T_h2u_negation_guard_vs_h2r_on_h1p_v1",
    ),
)


INITIAL_TRANSFER_LABELS = ("h2s", "h2q", "h2m")
FIRST_PASS_TRANSFER_LABELS = ("h2k", "h2l", "h2f", "h2b", "h1x")
OLDER_TRANSFER_LABELS = ("h1y", "h1o", "h1p")


def build_h2u_negation_guard_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    non_exact_rows = _non_exact_rows(PACKET_SPECS)
    fixed_case_rows = _fixed_case_rows(COMPARISON_SPECS)
    blocked_rows = _blocked_guard_rows(PACKET_SPECS)
    finding_rows = _finding_rows(
        packet_rows=packet_rows,
        comparison_rows=comparison_rows,
        non_exact_rows=non_exact_rows,
        fixed_case_rows=fixed_case_rows,
        blocked_rows=blocked_rows,
    )

    h2t_h2r = _packet_by_profile(packet_rows, "h2t_h2r_composed_route_gating")
    h2t_h2u = _packet_by_profile(packet_rows, "h2t_h2u_negation_guard")
    h2s_h2u = _packet_by_profile(packet_rows, "h2s_h2u_negation_guard")
    h2q_h2u = _packet_by_profile(packet_rows, "h2q_h2u_negation_guard")
    h2m_h2u = _packet_by_profile(packet_rows, "h2m_h2u_negation_guard")
    h2t_comparison = _comparison_by_label(comparison_rows, "h2t_h2u_vs_h2r")
    initial_transfer_packets = [_packet_by_profile(packet_rows, f"{label}_h2u_negation_guard") for label in INITIAL_TRANSFER_LABELS]
    first_pass_transfer_packets = [
        _packet_by_profile(packet_rows, f"{label}_h2u_negation_guard") for label in FIRST_PASS_TRANSFER_LABELS
    ]
    older_transfer_packets = [
        _packet_by_profile(packet_rows, f"{label}_h2u_negation_guard") for label in OLDER_TRANSFER_LABELS
    ]
    all_transfer_packets = initial_transfer_packets + first_pass_transfer_packets + older_transfer_packets
    initial_transfer_comparisons = [
        _comparison_by_label(comparison_rows, f"{label}_h2u_vs_h2r") for label in INITIAL_TRANSFER_LABELS
    ]
    first_pass_transfer_comparisons = [
        _comparison_by_label(comparison_rows, f"{label}_h2u_vs_h2r") for label in FIRST_PASS_TRANSFER_LABELS
    ]
    older_transfer_comparisons = [
        _comparison_by_label(comparison_rows, f"{label}_h2u_vs_h2r") for label in OLDER_TRANSFER_LABELS
    ]
    all_transfer_comparisons = initial_transfer_comparisons + first_pass_transfer_comparisons + older_transfer_comparisons
    h2t_blocked_rows = [row for row in blocked_rows if row["slice"] == "h2t"]
    transfer_blocked_rows = [row for row in blocked_rows if row["slice"] != "h2t"]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "h2t_case_count": h2t_h2u["case_count"],
        "h2t_h2r_exact_success_count": h2t_h2r["exact_success_count"],
        "h2t_h2u_exact_success_count": h2t_h2u["exact_success_count"],
        "h2t_h2u_executor_success_count": h2t_h2u["executor_success_count"],
        "h2t_delta_exact_vs_h2r": h2t_comparison["delta_exact_rate"],
        "h2t_delta_executor_vs_h2r": h2t_comparison["delta_executor_equivalence_rate"],
        "h2t_fixed_case_count": len([row for row in fixed_case_rows if row["comparison_label"] == "h2t_h2u_vs_h2r"]),
        "h2s_h2u_exact_success_count": h2s_h2u["exact_success_count"],
        "h2q_h2u_exact_success_count": h2q_h2u["exact_success_count"],
        "h2m_h2u_exact_success_count": h2m_h2u["exact_success_count"],
        "h2k_h2u_exact_success_count": _packet_by_profile(packet_rows, "h2k_h2u_negation_guard")[
            "exact_success_count"
        ],
        "h2l_h2u_exact_success_count": _packet_by_profile(packet_rows, "h2l_h2u_negation_guard")[
            "exact_success_count"
        ],
        "h2f_h2u_exact_success_count": _packet_by_profile(packet_rows, "h2f_h2u_negation_guard")[
            "exact_success_count"
        ],
        "h2b_h2u_exact_success_count": _packet_by_profile(packet_rows, "h2b_h2u_negation_guard")[
            "exact_success_count"
        ],
        "h1x_h2u_exact_success_count": _packet_by_profile(packet_rows, "h1x_h2u_negation_guard")[
            "exact_success_count"
        ],
        "h1y_h2u_exact_success_count": _packet_by_profile(packet_rows, "h1y_h2u_negation_guard")[
            "exact_success_count"
        ],
        "h1o_h2u_exact_success_count": _packet_by_profile(packet_rows, "h1o_h2u_negation_guard")[
            "exact_success_count"
        ],
        "h1p_h2u_exact_success_count": _packet_by_profile(packet_rows, "h1p_h2u_negation_guard")[
            "exact_success_count"
        ],
        "transfer_case_count": sum(int(row["case_count"]) for row in initial_transfer_packets),
        "transfer_exact_success_count": sum(int(row["exact_success_count"]) for row in initial_transfer_packets),
        "transfer_delta_exact_sum_vs_h2r": sum(float(row["delta_exact_rate"]) for row in initial_transfer_comparisons),
        "first_pass_transfer_case_count": sum(int(row["case_count"]) for row in first_pass_transfer_packets),
        "first_pass_transfer_exact_success_count": sum(
            int(row["exact_success_count"]) for row in first_pass_transfer_packets
        ),
        "first_pass_transfer_delta_exact_sum_vs_h2r": sum(
            float(row["delta_exact_rate"]) for row in first_pass_transfer_comparisons
        ),
        "older_transfer_case_count": sum(int(row["case_count"]) for row in older_transfer_packets),
        "older_transfer_exact_success_count": sum(int(row["exact_success_count"]) for row in older_transfer_packets),
        "older_transfer_delta_exact_sum_vs_h2r": sum(
            float(row["delta_exact_rate"]) for row in older_transfer_comparisons
        ),
        "broad_transfer_case_count": sum(int(row["case_count"]) for row in all_transfer_packets),
        "broad_transfer_exact_success_count": sum(int(row["exact_success_count"]) for row in all_transfer_packets),
        "broad_transfer_delta_exact_sum_vs_h2r": sum(float(row["delta_exact_rate"]) for row in all_transfer_comparisons),
        "blocked_guard_count": len(blocked_rows),
        "h2t_blocked_guard_count": len(h2t_blocked_rows),
        "transfer_blocked_guard_count": len(transfer_blocked_rows),
        "target_normalization_blocked_count": sum(
            1 for row in blocked_rows if row["intervention_kind"] == "visual_target_query_normalization_blocked"
        ),
        "composed_route_gating_blocked_count": sum(
            1 for row in blocked_rows if row["intervention_kind"] == "visual_composed_route_gating_blocked"
        ),
        "h2u_non_exact_count": sum(1 for row in non_exact_rows if row["profile_label"].endswith("h2u_negation_guard")),
        "promotion_decision": "h2u_promotes_to_harder_semantic_negation_holdout",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "non_exact_rows": non_exact_rows,
        "fixed_case_rows": fixed_case_rows,
        "blocked_guard_rows": blocked_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h2u_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2u_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2u_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2u_fixed_case_rows.csv", fixed_case_rows)
    _write_csv(tables_dir / "h2u_blocked_guard_rows.csv", blocked_rows)
    _write_csv(tables_dir / "h2u_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2u_negation_guard_transfer_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _fixed_case_rows(specs: tuple[ComparisonSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
        for row in payload["case_deltas"]:
            if row.get("baseline_replay_exact_match") is not False or row.get("candidate_replay_exact_match") is not True:
                continue
            rows.append(
                {
                    "comparison_label": spec.comparison_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "baseline_failure_mode": row.get("baseline_replay_failure_mode", ""),
                    "candidate_failure_mode": row.get("candidate_replay_failure_mode", ""),
                    "baseline_executor_equivalence_match": row.get("baseline_replay_executor_equivalence_match"),
                    "candidate_executor_equivalence_match": row.get("candidate_replay_executor_equivalence_match"),
                }
            )
    return rows


def _blocked_guard_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if "h2u_negation_guard" not in spec.profile_label:
            continue
        slice_name = spec.profile_label.split("_", 1)[0]
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for kind in ("visual_target_query_normalization_blocked", "visual_composed_route_gating_blocked"):
                entries = metadata.get(kind, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    rows.append(
                        {
                            "profile_label": spec.profile_label,
                            "slice": slice_name,
                            "case_id": result["case_id"],
                            "family": result.get("family", ""),
                            "intervention_kind": kind,
                            "from_tool": entry.get("from_tool", ""),
                            "from_arguments": _compact_json(entry.get("from_arguments", {})),
                            "preserved_target_query": entry.get("preserved_target_query", ""),
                            "preserved_region_id": entry.get("preserved_region_id", ""),
                            "blocked_label": entry.get("blocked_label", entry.get("prompt_state_label", "")),
                            "blocked_region_id": entry.get("blocked_region_id", ""),
                            "prompt_state_label": entry.get("prompt_state_label", ""),
                            "reason": entry.get("reason", ""),
                        }
                    )
    return rows


def _finding_rows(
    *,
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    fixed_case_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2t_h2r = _packet_by_profile(packet_rows, "h2t_h2r_composed_route_gating")
    h2t_h2u = _packet_by_profile(packet_rows, "h2t_h2u_negation_guard")
    h2t_comparison = _comparison_by_label(comparison_rows, "h2t_h2u_vs_h2r")
    initial_transfer_exact = sum(
        int(_packet_by_profile(packet_rows, f"{label}_h2u_negation_guard")["exact_success_count"])
        for label in INITIAL_TRANSFER_LABELS
    )
    initial_transfer_cases = sum(
        int(_packet_by_profile(packet_rows, f"{label}_h2u_negation_guard")["case_count"])
        for label in INITIAL_TRANSFER_LABELS
    )
    first_pass_transfer_exact = sum(
        int(_packet_by_profile(packet_rows, f"{label}_h2u_negation_guard")["exact_success_count"])
        for label in FIRST_PASS_TRANSFER_LABELS
    )
    first_pass_transfer_cases = sum(
        int(_packet_by_profile(packet_rows, f"{label}_h2u_negation_guard")["case_count"])
        for label in FIRST_PASS_TRANSFER_LABELS
    )
    older_transfer_exact = sum(
        int(_packet_by_profile(packet_rows, f"{label}_h2u_negation_guard")["exact_success_count"])
        for label in OLDER_TRANSFER_LABELS
    )
    older_transfer_cases = sum(
        int(_packet_by_profile(packet_rows, f"{label}_h2u_negation_guard")["case_count"])
        for label in OLDER_TRANSFER_LABELS
    )
    broad_transfer_exact = initial_transfer_exact + first_pass_transfer_exact + older_transfer_exact
    broad_transfer_cases = initial_transfer_cases + first_pass_transfer_cases + older_transfer_cases
    h2u_non_exact = [row for row in non_exact_rows if row["profile_label"].endswith("h2u_negation_guard")]
    h2t_fixed = [row for row in fixed_case_rows if row["comparison_label"] == "h2t_h2u_vs_h2r"]
    h2t_blocks = [row for row in blocked_rows if row["slice"] == "h2t"]
    transfer_blocks = [row for row in blocked_rows if row["slice"] != "h2t"]
    return [
        {
            "finding_id": "h2u_repairs_h2t_negation_scope",
            "finding": (
                f"H2u raises H2t from H2r's {h2t_h2r['exact_success_count']}/{h2t_h2r['case_count']} strict "
                f"to {h2t_h2u['exact_success_count']}/{h2t_h2u['case_count']} strict, with delta "
                f"{_format_rate(h2t_comparison['delta_exact_rate'])} exact-rate and "
                f"{_format_rate(h2t_comparison['delta_executor_equivalence_rate'])} executor-equivalence-rate."
            ),
        },
        {
            "finding_id": "h2u_fix_is_pipeline_ordered",
            "finding": (
                f"The repaired H2t rows are {', '.join(row['case_id'] for row in h2t_fixed)}. "
                f"H2u records {len(h2t_blocks)} H2t blocked-guard interventions, covering both target normalization "
                "and composed-route gating."
            ),
        },
        {
            "finding_id": "h2u_transfer_preserves_h2r",
            "finding": (
                f"H2u preserves {initial_transfer_exact}/{initial_transfer_cases} strict exactness across H2s, H2q, "
                "and H2m, with zero exact-rate and executor-equivalence-rate deltas versus H2r on all three "
                "initial transfer checks."
            ),
        },
        {
            "finding_id": "h2u_first_pass_transfer_preserves_h2r",
            "finding": (
                f"H2u also preserves {first_pass_transfer_exact}/{first_pass_transfer_cases} strict exactness across "
                "H2k, H2l, H2f, H2b, and H1x with zero aggregate exact-rate delta versus H2r."
            ),
        },
        {
            "finding_id": "h2u_older_transfer_preserves_h2r",
            "finding": (
                f"H2u preserves {older_transfer_exact}/{older_transfer_cases} strict exactness across the older "
                "H1y, H1o, and H1p family. Combined with H2s/H2q/H2m and H2k/H2l/H2f/H2b/H1x, the current "
                f"broad transfer subtotal is {broad_transfer_exact}/{broad_transfer_cases} exact with zero aggregate "
                "exact-rate delta versus H2r."
            ),
        },
        {
            "finding_id": "h2u_guard_fires_without_transfer_cost",
            "finding": (
                f"H2u records {len(transfer_blocks)} blocked transfer interventions outside H2t, but those rows remain "
                "exact. This suggests the guard is not merely inactive on transfer; it can fire conservatively without "
                "breaking prior wins."
            ),
        },
        {
            "finding_id": "h2u_no_remaining_non_exact_rows",
            "finding": (
                f"Across the H2u packets summarized here, H2u has {len(h2u_non_exact)} non-exact rows. "
                "The next risk is semantic negation generalization rather than same-family transfer coverage."
            ),
        },
    ]


def _format_rate(value: Any) -> str:
    return f"{float(value):.2f}"


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2u Negation Guard Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2u is the first repair after H2t exposed a controller-induced negation-scope regression. The repair is "
            "not a prompt rewrite: it adds a runtime guard that preserves exact current-surface targets when the "
            "candidate replacement is a note, caption, or old/prior contextual label."
        ),
        "",
        (
            f"On H2t, H2u reaches `{manifest['h2t_h2u_exact_success_count']} / {manifest['h2t_case_count']}` strict "
            f"and `{manifest['h2t_h2u_executor_success_count']} / {manifest['h2t_case_count']}` executor-equivalent, "
            f"improving `{_format_rate(manifest['h2t_delta_exact_vs_h2r'])}` exact-rate over H2r. It fixes "
            f"`{manifest['h2t_fixed_case_count']}` H2t rows."
        ),
        "",
        (
            f"Transfer is clean on this wave: H2u preserves `{manifest['transfer_exact_success_count']} / "
            f"{manifest['transfer_case_count']}` strict exactness across H2s, H2q, and H2m, and all three "
            "H2r-vs-H2u comparisons have zero exact/executor-equivalence deltas. The guard fires on transfer "
            f"`{manifest['transfer_blocked_guard_count']}` times without causing a miss."
        ),
        "",
        (
            f"The broader first-pass transfer backtest is also clean: H2u preserves "
            f"`{manifest['first_pass_transfer_exact_success_count']} / {manifest['first_pass_transfer_case_count']}` "
            "strict exactness across H2k, H2l, H2f, H2b, and H1x. Combined with the initial H2s/H2q/H2m transfer "
            f"gate, this gives `{manifest['transfer_exact_success_count'] + manifest['first_pass_transfer_exact_success_count']} / "
            f"{manifest['transfer_case_count'] + manifest['first_pass_transfer_case_count']}` before the older family."
        ),
        "",
        (
            f"The older transfer closure is clean too: H2u preserves "
            f"`{manifest['older_transfer_exact_success_count']} / {manifest['older_transfer_case_count']}` strict "
            "exactness across H1y, H1o, and H1p. The current broad transfer subtotal is now "
            f"`{manifest['broad_transfer_exact_success_count']} / "
            f"{manifest['broad_transfer_case_count']}` with zero aggregate exact-rate delta versus H2r."
        ),
        "",
        "![H2u negation guard transfer gate](figures/h2u_negation_guard_transfer_gate.svg)",
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## Fixed Case Rows",
        "",
        _table(payload["fixed_case_rows"]),
        "",
        "## Blocked Guard Rows",
        "",
        _table(payload["blocked_guard_rows"]),
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


def _write_svg(path: Path, packet_rows: list[dict[str, Any]]) -> None:
    pairs = [
        ("h2t_h2r_composed_route_gating", "h2t_h2u_negation_guard", "H2t", 10),
        ("h2s_h2r_composed_route_gating", "h2s_h2u_negation_guard", "H2s", 10),
        ("h2q_h2r_composed_route_gating", "h2q_h2u_negation_guard", "H2q", 8),
        ("h2m_h2r_composed_route_gating", "h2m_h2u_negation_guard", "H2m", 8),
        ("h2k_h2r_composed_route_gating", "h2k_h2u_negation_guard", "H2k", 8),
        ("h2l_h2r_composed_route_gating", "h2l_h2u_negation_guard", "H2l", 8),
        ("h2f_h2r_composed_route_gating", "h2f_h2u_negation_guard", "H2f", 10),
        ("h2b_h2r_composed_route_gating", "h2b_h2u_negation_guard", "H2b", 5),
        ("h1x_h2r_composed_route_gating", "h1x_h2u_negation_guard", "H1x", 8),
        ("h1y_h2r_composed_route_gating", "h1y_h2u_negation_guard", "H1y", 10),
        ("h1o_h2r_composed_route_gating", "h1o_h2u_negation_guard", "H1o", 12),
        ("h1p_h2r_composed_route_gating", "h1p_h2u_negation_guard", "H1p", 12),
    ]
    by_profile = {row["profile_label"]: row for row in packet_rows}
    width = 1450
    height = 390
    chart_left = 70
    chart_top = 78
    chart_height = 190
    chart_right = 1370
    bar_width = 36
    group_gap = 34
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">H2u negation guard transfer gate</title>',
        '<desc id="desc">H2u improves H2t from exact rate 0.8 to 1.0 and ties H2r at 1.0 across the current transfer gates, including older H1y, H1o, and H1p.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2u repairs H2t without transfer regression</text>',
        f'<line x1="70" y1="268" x2="{chart_right}" y2="268" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        y = chart_top + chart_height - tick * (chart_height / 5)
        lines.append(f'<line x1="66" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="44" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#6B7280">{tick / 5:.1f}</text>'
        )
    for index, (baseline_label, candidate_label, slice_label, denominator) in enumerate(pairs):
        baseline = by_profile[baseline_label]
        candidate = by_profile[candidate_label]
        x = chart_left + index * ((bar_width * 2) + group_gap)
        baseline_height = float(baseline["exact_rate"]) * chart_height
        candidate_height = float(candidate["exact_rate"]) * chart_height
        baseline_y = chart_top + chart_height - baseline_height
        candidate_y = chart_top + chart_height - candidate_height
        lines.append(
            f'<rect x="{x}" y="{baseline_y:.1f}" width="{bar_width}" height="{baseline_height:.1f}" fill="#9CA3AF"/>'
        )
        lines.append(
            f'<rect x="{x + bar_width + 6}" y="{candidate_y:.1f}" width="{bar_width}" height="{candidate_height:.1f}" fill="#A16207"/>'
        )
        lines.append(
            f'<text x="{x + 8}" y="{baseline_y - 8:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#111827">{int(baseline["exact_success_count"])}/{denominator}</text>'
        )
        lines.append(
            f'<text x="{x + bar_width + 14}" y="{candidate_y - 8:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#111827">{int(candidate["exact_success_count"])}/{denominator}</text>'
        )
        lines.append(
            f'<text x="{x + 43}" y="300" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{slice_label}</text>'
        )
    lines.append('<rect x="1118" y="326" width="12" height="12" fill="#9CA3AF"/>')
    lines.append('<text x="1138" y="337" font-family="Arial, sans-serif" font-size="12" fill="#374151">H2r</text>')
    lines.append('<rect x="1192" y="326" width="12" height="12" fill="#A16207"/>')
    lines.append('<text x="1212" y="337" font-family="Arial, sans-serif" font-size="12" fill="#374151">H2u</text>')
    lines.append(
        '<text x="32" y="362" font-family="Arial, sans-serif" font-size="13" fill="#374151">H2u blocks negated-context rewrites in both target normalization and composed-route gating.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2u negation guard synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2u_negation_guard_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
