from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1x_v11_breaker_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    profile_label: str
    live_packet_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1x_v11_breaker_no_directive_execute_v1",
    ),
    PacketSpec(
        "component_label_guard_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1x_v11_breaker_component_label_guard_execute_v1",
    ),
    PacketSpec(
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1x_v11_breaker_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "code_label_exact_guard_v15",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1x_v11_breaker_code_label_exact_guard_execute_v1",
    ),
)

COMPARISON_DIRS: tuple[Path, ...] = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1x_component_label_guard_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1x_component_residual_guard_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1x_code_label_exact_guard_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1x_component_residual_guard_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1x_code_label_exact_guard_vs_component_label_guard_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1x_code_label_exact_guard_vs_component_residual_guard_v1",
)


def build_h1x_v11_breaker_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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
    v15 = _row_by_label(packet_rows, "code_label_exact_guard_v15")
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
        "v15_exact_success_count": int(v15["exact_success_count"]),
        "v15_executor_success_count": int(v15["executor_success_count"]),
        "comparison_count": len(comparison_rows),
        "finding_count": len(finding_rows),
        "promotion_decision": "component_residual_guard_is_h1x_local_winner_keep_transfer_gate",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "family_rows": family_rows,
        "comparison_rows": comparison_rows,
        "non_exact_rows": non_exact_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1x_v11_breaker_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1x_v11_breaker_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h1x_v11_breaker_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h1x_v11_breaker_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h1x_v11_breaker_findings.csv", finding_rows)
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
        "executable_paraphrase_count": sum(
            1 for row in results if row.get("replay_failure_mode") == "executable_paraphrase"
        ),
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
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "failure_mode": row.get("replay_failure_mode", ""),
                    "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                    "output_dir": row.get("output_dir", ""),
                }
            )
    return rows


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    no_directive = _row_by_label(packet_rows, "no_directive")
    v11 = _row_by_label(packet_rows, "component_label_guard_v11")
    v12 = _row_by_label(packet_rows, "component_residual_guard_v12")
    v15 = _row_by_label(packet_rows, "code_label_exact_guard_v15")
    v12_vs_no = _comparison_by_dir(comparison_rows, "component_residual_guard_vs_no_directive")
    v12_vs_v11 = _comparison_by_dir(comparison_rows, "component_residual_guard_vs_component_label_guard")
    v15_surface = _family_row(family_rows, "code_label_exact_guard_v15", "h1x_oblique_surface_value")
    v11_stale = _family_row(family_rows, "component_label_guard_v11", "h1x_oblique_stale_field")
    v15_non_exact = ", ".join(
        row["case_id"] for row in non_exact_rows if row["profile_label"] == "code_label_exact_guard_v15"
    )
    return [
        {
            "finding_id": "h1x_breaks_no_directive",
            "finding": (
                f"No-directive reaches {no_directive['exact_success_count']}/8 exact and "
                f"{no_directive['executor_success_count']}/8 executor-equivalent; it only solves the "
                "activation/no-call rows and fails the oblique stale-field, surface-value, and nonstandard-class rows."
            ),
        },
        {
            "finding_id": "h1x_breaks_v11_saturation",
            "finding": (
                f"Component-label guard v11 drops to {v11['exact_success_count']}/8 exact and "
                f"{v11['executor_success_count']}/8 executor-equivalent; the miss is concentrated in "
                f"oblique stale-field routing at {v11_stale['exact_success_count']}/2."
            ),
        },
        {
            "finding_id": "v12_local_winner",
            "finding": (
                f"Component-residual guard v12 reaches {v12['exact_success_count']}/8 exact and "
                f"{v12['executor_success_count']}/8 executor-equivalent, a +{v12_vs_no['delta_exact_rate']:.3f} "
                f"exact-rate delta over no-directive and +{v12_vs_v11['delta_exact_rate']:.3f} over v11 on H1x."
            ),
        },
        {
            "finding_id": "v15_over_narrows_again",
            "finding": (
                f"Code-label exact guard v15 reaches {v15['exact_success_count']}/8 exact and "
                f"{v15['executor_success_count']}/8 executor-equivalent. It is only "
                f"{v15_surface['exact_success_count']}/2 strict exact on oblique surface-value rows; non-exact rows: "
                f"{v15_non_exact}."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "Treat H1x as evidence for a routed residual helper, not a global default replacement. The next "
                "hard slice should retest v12 against the old transfer packets and a new mixed packet with "
                "oblique stale-field plus surface-value rows in the same workflow family."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H1x V11-Breaker Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H1x is the first focused post-H1w packet that breaks v11 saturation. No-directive reaches "
            "`2 / 8`, v11 reaches `7 / 8`, v12 reaches `8 / 8`, and v15 reaches `6 / 8` strict exact "
            "with `7 / 8` executor-equivalent. The result strengthens the routed-helper hypothesis: residual "
            "wording is locally useful on oblique stale-field rows, while code-label exactness remains too narrow."
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
    parser = argparse.ArgumentParser(description="Build the H1x v11-breaker synthesis packet.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1x_v11_breaker_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
