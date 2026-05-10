from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1y_routed_residual_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    profile_label: str
    live_packet_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1y_routed_residual_no_directive_execute_v1",
    ),
    PacketSpec(
        "component_label_guard_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1y_routed_residual_component_label_guard_execute_v1",
    ),
    PacketSpec(
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1y_routed_residual_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "routed_residual_guard_v16",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1y_routed_residual_routed_residual_guard_execute_v1",
    ),
)

COMPARISON_DIRS: tuple[Path, ...] = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1y_component_label_guard_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1y_component_residual_guard_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1y_routed_residual_guard_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1y_component_residual_guard_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1y_routed_residual_guard_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1y_routed_residual_guard_vs_component_residual_guard_v1",
)


def build_h1y_routed_residual_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    family_rows = _family_rows()
    comparison_rows = [_comparison_row(path) for path in COMPARISON_DIRS]
    non_exact_rows = _non_exact_rows()
    finding_rows = _finding_rows(packet_rows, family_rows, comparison_rows, non_exact_rows)
    no_directive = _row_by_label(packet_rows, "no_directive")
    v11 = _row_by_label(packet_rows, "component_label_guard_v11")
    v12 = _row_by_label(packet_rows, "component_residual_guard_v12")
    v16 = _row_by_label(packet_rows, "routed_residual_guard_v16")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "profile_count": len(packet_rows),
        "case_count": int(v11["case_count"]),
        "no_directive_exact_success_count": int(no_directive["exact_success_count"]),
        "v11_exact_success_count": int(v11["exact_success_count"]),
        "v11_executor_success_count": int(v11["executor_success_count"]),
        "v12_exact_success_count": int(v12["exact_success_count"]),
        "v12_executor_success_count": int(v12["executor_success_count"]),
        "v16_exact_success_count": int(v16["exact_success_count"]),
        "v16_executor_success_count": int(v16["executor_success_count"]),
        "comparison_count": len(comparison_rows),
        "finding_count": len(finding_rows),
        "promotion_decision": "do_not_promote_v16_design_v17_selection_origin_guard",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "family_rows": family_rows,
        "comparison_rows": comparison_rows,
        "non_exact_rows": non_exact_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1y_routed_residual_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1y_routed_residual_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h1y_routed_residual_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h1y_routed_residual_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h1y_routed_residual_findings.csv", finding_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _packet_row(spec: PacketSpec) -> dict[str, Any]:
    summary = _read_json(spec.live_packet_dir / "summary.json")
    results = _read_json(spec.live_packet_dir / "live_replay_results.json")
    case_count = int(summary["case_count"])
    exact_success_count = sum(1 for row in results if row.get("replay_exact_match") is True)
    executor_success_count = sum(1 for row in results if row.get("replay_executor_equivalence_match") is True)
    return {
        "profile_label": spec.profile_label,
        "system_id": summary["system_id"],
        "packet_dir": str(spec.live_packet_dir.relative_to(ROOT)),
        "case_count": case_count,
        "exact_success_count": exact_success_count,
        "exact_rate": exact_success_count / case_count if case_count else 0.0,
        "executor_success_count": executor_success_count,
        "executor_rate": executor_success_count / case_count if case_count else 0.0,
        "argument_mismatch_count": sum(1 for row in results if row.get("replay_failure_mode") == "argument_mismatch"),
        "wrong_tool_count": sum(1 for row in results if row.get("replay_failure_mode") == "wrong_tool"),
        "no_tool_call_count": sum(1 for row in results if row.get("replay_failure_mode") == "no_tool_call"),
    }


def _family_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
        results = _read_json(spec.live_packet_dir / "live_replay_results.json")
        for family in sorted({str(row["family"]) for row in results}):
            family_results = [row for row in results if row["family"] == family]
            case_count = len(family_results)
            exact_success_count = sum(1 for row in family_results if row.get("replay_exact_match") is True)
            executor_success_count = sum(
                1 for row in family_results if row.get("replay_executor_equivalence_match") is True
            )
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "family": family,
                    "case_count": case_count,
                    "exact_success_count": exact_success_count,
                    "exact_rate": exact_success_count / case_count if case_count else 0.0,
                    "executor_success_count": executor_success_count,
                    "executor_rate": executor_success_count / case_count if case_count else 0.0,
                }
            )
    return rows


def _comparison_row(path: Path) -> dict[str, Any]:
    payload = _read_json(path / "live_replay_comparison.json")
    summary = payload["summary"]
    return {
        "comparison_dir": str(path.relative_to(ROOT)),
        "baseline_system_id": summary["baseline_system_id"],
        "candidate_system_id": summary["candidate_system_id"],
        "shared_case_count": summary["shared_case_count"],
        "baseline_exact_rate": summary["baseline_exact_rate"],
        "candidate_exact_rate": summary["candidate_exact_rate"],
        "delta_exact_rate": summary["delta_exact_rate"],
        "baseline_executor_equivalence_rate": summary["baseline_executor_equivalence_rate"],
        "candidate_executor_equivalence_rate": summary["candidate_executor_equivalence_rate"],
        "delta_executor_equivalence_rate": summary["delta_executor_equivalence_rate"],
    }


def _non_exact_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
        results = _read_json(spec.live_packet_dir / "live_replay_results.json")
        for row in results:
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
                    "expected_arguments": detail["expected_arguments"],
                    "actual_tool": detail["actual_tool"],
                    "actual_arguments": detail["actual_arguments"],
                    "actual_region_ids": detail["actual_region_ids"],
                    "output_dir": row.get("output_dir", ""),
                }
            )
    return rows


def _probe_detail(row: dict[str, Any]) -> dict[str, str]:
    output_dir = row.get("output_dir")
    if not output_dir:
        return _empty_probe_detail()
    probe_path = Path(str(output_dir)) / "probe_results.json"
    if not probe_path.exists():
        return _empty_probe_detail()
    probe_rows = _read_json(probe_path)
    if not probe_rows:
        return _empty_probe_detail()
    probe = probe_rows[0]
    expected_calls = probe.get("expected_calls") or []
    actual_calls = probe.get("actual_calls") or []
    actual_execution = probe.get("actual_execution") or []
    expected = expected_calls[0] if expected_calls else {}
    actual = actual_calls[0] if actual_calls else {}
    region_ids = []
    if actual_execution:
        output = actual_execution[-1].get("output") or {}
        region_ids = output.get("region_ids") or []
    return {
        "expected_tool": str(expected.get("name", "")),
        "expected_arguments": json.dumps(expected.get("arguments", {}), sort_keys=True),
        "actual_tool": str(actual.get("name", "")),
        "actual_arguments": json.dumps(actual.get("arguments", {}), sort_keys=True),
        "actual_region_ids": ",".join(str(region_id) for region_id in region_ids),
    }


def _empty_probe_detail() -> dict[str, str]:
    return {
        "expected_tool": "",
        "expected_arguments": "",
        "actual_tool": "",
        "actual_arguments": "",
        "actual_region_ids": "",
    }


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    no_directive = _row_by_label(packet_rows, "no_directive")
    v11 = _row_by_label(packet_rows, "component_label_guard_v11")
    v12 = _row_by_label(packet_rows, "component_residual_guard_v12")
    v16 = _row_by_label(packet_rows, "routed_residual_guard_v16")
    v12_vs_v11 = _comparison_by_dir(comparison_rows, "component_residual_guard_vs_component_label_guard")
    v16_vs_v11 = _comparison_by_dir(comparison_rows, "routed_residual_guard_vs_component_label_guard")
    v16_vs_v12 = _comparison_by_dir(comparison_rows, "routed_residual_guard_vs_component_residual_guard")
    v12_surface = _family_row(family_rows, "component_residual_guard_v12", "h1y_preserve_surface_value")
    v16_surface = _family_row(family_rows, "routed_residual_guard_v16", "h1y_preserve_surface_value")
    v16_failures = ", ".join(
        row["case_id"] for row in non_exact_rows if row["profile_label"] == "routed_residual_guard_v16"
    )
    return [
        {
            "finding_id": "h1y_is_harder_than_no_directive",
            "finding": (
                f"No-directive reaches {no_directive['exact_success_count']}/10 exact and "
                f"{no_directive['executor_success_count']}/10 executor-equivalent. The mixed packet is a genuine "
                "tool-use breaker rather than a saturated readiness row."
            ),
        },
        {
            "finding_id": "v11_partial_default_remains_useful",
            "finding": (
                f"Component-label guard v11 reaches {v11['exact_success_count']}/10 exact and "
                f"{v11['executor_success_count']}/10 executor-equivalent. It preserves surface-value holdouts and "
                "the activation row, but misses all three stale-field route rows."
            ),
        },
        {
            "finding_id": "v12_best_local_but_still_noisy",
            "finding": (
                f"Component-residual guard v12 is the local H1y winner at {v12['exact_success_count']}/10 exact "
                f"and {v12['executor_success_count']}/10 executor-equivalent, a "
                f"+{v12_vs_v11['delta_exact_rate']:.3f} exact-rate delta over v11. It still only reaches "
                f"{v12_surface['exact_success_count']}/2 on surface-value holdouts, so broad residual wording "
                "keeps the old transfer risk alive."
            ),
        },
        {
            "finding_id": "v16_route_text_is_not_enough",
            "finding": (
                f"Routed residual guard v16 ties v11 at {v16['exact_success_count']}/10 exact and "
                f"{v16['executor_success_count']}/10 executor-equivalent, with "
                f"{v16_vs_v11['delta_exact_rate']:.3f} exact-rate delta over v11 and "
                f"{v16_vs_v12['delta_exact_rate']:.3f} versus v12. It drops to "
                f"{v16_surface['exact_success_count']}/2 on surface-value holdouts; non-exact rows: {v16_failures}."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "Do not promote v16. The next profile should target selection-origin and component-phrase "
                "precedence directly: forbid refine_selection on user-mentioned stale ids without a prior tool "
                "result, prefer explicit 'label is/component is' phrases, preserve 'locate X exactly' code labels, "
                "and drop wrapper words like lifecycle or operation without replacing component labels by values."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H1y Routed Residual Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H1y tests the routed-helper hypothesis directly. The packet mixes stale-field routes, "
            "nonstandard component classes, code labels, ordinary surface-value holdouts, and one activation row. "
            "No-directive reaches `0 / 10`, v11 reaches `5 / 10`, v12 reaches `7 / 10`, and v16 reaches `5 / 10`. "
            "The negative v16 result matters: route wording alone did not preserve v11 while capturing v12's gains."
        ),
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Family Rows",
        "",
        _table(payload["family_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
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


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_None._"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"`{value:.5f}`"
    if isinstance(value, (int, bool)):
        return f"`{str(value).lower() if isinstance(value, bool) else value}`"
    return str(value)


def _row_by_label(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["profile_label"] == profile_label:
            return row
    raise KeyError(profile_label)


def _family_row(rows: list[dict[str, Any]], profile_label: str, family: str) -> dict[str, Any]:
    for row in rows:
        if row["profile_label"] == profile_label and row["family"] == family:
            return row
    raise KeyError((profile_label, family))


def _comparison_by_dir(rows: list[dict[str, Any]], pattern: str) -> dict[str, Any]:
    for row in rows:
        if pattern in row["comparison_dir"]:
            return row
    raise KeyError(pattern)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H1y routed-residual synthesis packet.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1y_routed_residual_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
