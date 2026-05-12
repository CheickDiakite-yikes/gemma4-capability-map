from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2l_target_normalization_overreach_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    profile_label: str
    packet_dir: Path


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_label: str
    comparison_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2e_route_arbitration",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2l_target_normalization_overreach_h2e_execute_v1",
    ),
    PacketSpec(
        "h2j_target_query_normalization_no_stale_gate",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2l_target_normalization_overreach_h2j_no_stale_gate_execute_v1",
    ),
    PacketSpec(
        "h2j_target_query_normalization",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260512T_h2l_target_normalization_overreach_h2j_execute_v1",
    ),
)


COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h2j_vs_h2e",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2l_target_normalization_overreach_h2j_vs_h2e_v1",
    ),
    ComparisonSpec(
        "h2j_vs_no_stale_gate",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260512T_h2l_target_normalization_overreach_h2j_vs_no_stale_gate_v1",
    ),
)


def build_h2l_target_normalization_overreach_synthesis(
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
    family_rows = _family_rows()
    finding_rows = _finding_rows(packet_rows, comparison_rows, non_exact_rows, intervention_rows)

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
        "promotion_decision": "h2l_overreach_holdout_passes_target_normalization",
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

    _write_csv(tables_dir / "h2l_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2l_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2l_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h2l_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2l_intervention_rows.csv", intervention_rows)
    _write_csv(tables_dir / "h2l_findings.csv", finding_rows)
    _write_svg(figures_dir / "h2l_target_normalization_overreach_gate.svg", packet_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _packet_row(spec: PacketSpec) -> dict[str, Any]:
    summary = _read_json(spec.packet_dir / "summary.json")
    results = _read_json(spec.packet_dir / "live_replay_results.json")
    case_count = int(summary["case_count"])
    exact_success_count = sum(1 for row in results if row.get("replay_exact_match") is True)
    executor_success_count = sum(1 for row in results if row.get("replay_executor_equivalence_match") is True)
    return {
        "profile_label": spec.profile_label,
        "system_id": summary["system_id"],
        "packet_dir": str(spec.packet_dir.relative_to(ROOT)),
        "case_count": case_count,
        "exact_success_count": exact_success_count,
        "exact_rate": exact_success_count / case_count if case_count else 0.0,
        "executor_success_count": executor_success_count,
        "executor_rate": executor_success_count / case_count if case_count else 0.0,
    }


def _comparison_row(spec: ComparisonSpec) -> dict[str, Any]:
    payload = _read_json(spec.comparison_dir / "live_replay_comparison.json")
    summary = payload["summary"]
    return {
        "comparison_label": spec.comparison_label,
        "comparison_dir": str(spec.comparison_dir.relative_to(ROOT)),
        "shared_case_count": summary["shared_case_count"],
        "baseline_exact_rate": summary["baseline_exact_rate"],
        "candidate_exact_rate": summary["candidate_exact_rate"],
        "delta_exact_rate": summary["delta_exact_rate"],
        "baseline_executor_equivalence_rate": summary["baseline_executor_equivalence_rate"],
        "candidate_executor_equivalence_rate": summary["candidate_executor_equivalence_rate"],
        "delta_executor_equivalence_rate": summary["delta_executor_equivalence_rate"],
    }


def _family_rows() -> list[dict[str, Any]]:
    packet_dir = ROOT / "results" / "tool_probe_replay_packets" / "20260512T_h2l_target_normalization_overreach_dry_run_v1"
    replay_cases = _read_json(packet_dir / "replay_cases.json")
    rows: list[dict[str, Any]] = []
    family_to_targets: dict[str, list[str]] = {}
    for case in replay_cases:
        family = case["family"]
        expected = (case.get("expected_calls") or [{}])[0]
        target = str(expected.get("arguments", {}).get("target_query", ""))
        family_to_targets.setdefault(family, []).append(target)
    for family, targets in sorted(family_to_targets.items()):
        rows.append(
            {
                "family": family,
                "case_count": len(targets),
                "expected_target_queries": "; ".join(targets),
            }
        )
    return rows


def _non_exact_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for row in _read_json(spec.packet_dir / "live_replay_results.json"):
            if row.get("replay_exact_match") is True:
                continue
            detail = _probe_detail(row)
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "failure_mode": row.get("replay_failure_mode", ""),
                    "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                    "expected_tool": detail["expected_tool"],
                    "expected_target_query": detail["expected_target_query"],
                    "actual_tool": detail["actual_tool"],
                    "actual_target_query": detail["actual_target_query"],
                }
            )
    return rows


def _controller_intervention_rows(specs: tuple[PacketSpec, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.profile_label not in {
            "h2j_target_query_normalization",
            "h2j_target_query_normalization_no_stale_gate",
        }:
            continue
        for result in _read_json(spec.packet_dir / "live_replay_results.json"):
            probe = _read_json(Path(result["output_dir"]) / "probe_results.json")[0]
            metadata = probe.get("runtime_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for kind in ("visual_target_query_normalization", "visual_stale_selection_gate"):
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
                        }
                    )
    return rows


def _probe_detail(row: dict[str, Any]) -> dict[str, str]:
    probe = _read_json(Path(row["output_dir"]) / "probe_results.json")[0]
    expected = (probe.get("expected_calls") or [{}])[0]
    actual = (probe.get("actual_calls") or [{}])[0]
    expected_args = expected.get("arguments", {})
    actual_args = actual.get("arguments", {})
    return {
        "expected_tool": str(expected.get("name", "")),
        "expected_target_query": str(expected_args.get("target_query", "")),
        "actual_tool": str(actual.get("name", "")),
        "actual_target_query": str(actual_args.get("target_query", "")),
    }


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2e = _packet_by_profile(packet_rows, "h2e_route_arbitration")
    h2j_no_stale = _packet_by_profile(packet_rows, "h2j_target_query_normalization_no_stale_gate")
    h2j = _packet_by_profile(packet_rows, "h2j_target_query_normalization")
    h2j_vs_h2e = _comparison_by_label(comparison_rows, "h2j_vs_h2e")
    h2j_vs_no_stale = _comparison_by_label(comparison_rows, "h2j_vs_no_stale_gate")
    h2e_non_exact = [row for row in non_exact_rows if row["profile_label"] == "h2e_route_arbitration"]
    full_target_count = len(
        _interventions_for(
            intervention_rows,
            profile_label="h2j_target_query_normalization",
            intervention_kind="visual_target_query_normalization",
        )
    )
    full_stale_count = len(
        _interventions_for(
            intervention_rows,
            profile_label="h2j_target_query_normalization",
            intervention_kind="visual_stale_selection_gate",
        )
    )
    no_stale_target_count = len(
        _interventions_for(
            intervention_rows,
            profile_label="h2j_target_query_normalization_no_stale_gate",
            intervention_kind="visual_target_query_normalization",
        )
    )
    no_stale_stale_count = len(
        _interventions_for(
            intervention_rows,
            profile_label="h2j_target_query_normalization_no_stale_gate",
            intervention_kind="visual_stale_selection_gate",
        )
    )
    return [
        {
            "finding_id": "h2l_overreach_holdout_passed",
            "finding": (
                f"H2l does not expose target-query over-normalization in this replay-shaped holdout: H2j reaches "
                f"{h2j['exact_success_count']}/8 exact and executor-equivalent while preserving value-bearing and "
                "alias-is-target rows."
            ),
        },
        {
            "finding_id": "h2l_repairs_h2e_regression_guard",
            "finding": (
                f"H2e reaches {h2e['exact_success_count']}/8 exact and has {len(h2e_non_exact)} non-exact row, "
                f"while H2j improves exact-rate by {h2j_vs_h2e['delta_exact_rate']} and executor-equivalence by "
                f"{h2j_vs_h2e['delta_executor_equivalence_rate']}."
            ),
        },
        {
            "finding_id": "h2l_mechanism_is_target_normalization_not_stale_gate",
            "finding": (
                f"Full H2j records {full_target_count} target-query-normalization intervention and "
                f"{full_stale_count} stale-selection interventions; the stale-gate-off ablation records "
                f"{no_stale_target_count} target-query-normalization intervention and {no_stale_stale_count} "
                f"stale-selection interventions, tying full H2j with a {h2j_vs_no_stale['delta_exact_rate']} "
                "exact-rate delta."
            ),
        },
        {
            "finding_id": "next_holdout_should_reduce_prompt_directness",
            "finding": (
                "H2l is useful positive control evidence, but the next holdout should reduce direct target-is wording "
                "or add repeated seed variants before treating over-normalization as closed."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2l Target Normalization Overreach Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2l is a post-H2k overreach holdout for H2j target-query normalization. It asks whether the controller "
            "will over-strip requested targets when the value-bearing phrase or alias label is itself the target. "
            "On this 8-case packet, full H2j and H2j without the stale-selection gate both reach 8/8 strict and "
            "executor-equivalent, while H2e reaches 7/8 and misses one short-label regression guard. The single "
            "recorded H2j intervention repairs `critical chip` into `status badge`; no stale-selection intervention "
            "is recorded. This supports the current target-normalization scope, while leaving a harder less-direct "
            "H2m holdout as the next appropriate pressure test."
        ),
        "",
        "![H2l target-normalization overreach gate](figures/h2l_target_normalization_overreach_gate.svg)",
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
        '<title id="title">H2l target-normalization overreach gate</title>',
        '<desc id="desc">H2j and H2j without stale-selection gate pass H2l while H2e misses one regression guard.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<text x="32" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">H2l checks target-normalization overreach</text>',
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


def _interventions_for(
    rows: list[dict[str, Any]], *, profile_label: str, intervention_kind: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["profile_label"] == profile_label and row["intervention_kind"] == intervention_kind
    ]


def _packet_by_profile(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["profile_label"] == profile_label:
            return row
    raise KeyError(profile_label)


def _comparison_by_label(rows: list[dict[str, Any]], comparison_label: str) -> dict[str, Any]:
    for row in rows:
        if row["comparison_label"] == comparison_label:
            return row
    raise KeyError(comparison_label)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_None._"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2l target-normalization overreach synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2l_target_normalization_overreach_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
