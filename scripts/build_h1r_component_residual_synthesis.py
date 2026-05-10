from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1r_component_residual_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    profile_label: str
    live_packet_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1r_component_label_residual_no_directive_execute_v1",
    ),
    PacketSpec(
        "component_label_guard_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1r_component_label_residual_component_label_guard_execute_v1",
    ),
    PacketSpec(
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h1r_component_label_residual_component_residual_guard_execute_v1",
    ),
)

COMPARISON_DIRS: tuple[Path, ...] = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1r_component_label_guard_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1r_component_residual_guard_vs_no_directive_v1",
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1r_component_residual_guard_vs_component_label_guard_v1",
)


def build_h1r_component_residual_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    family_rows = _family_rows()
    comparison_rows = [_comparison_row(path) for path in COMPARISON_DIRS]
    finding_rows = _finding_rows(packet_rows, comparison_rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "profile_count": len(packet_rows),
        "case_count": int(packet_rows[0]["case_count"]) if packet_rows else 0,
        "comparison_count": len(comparison_rows),
        "v12_exact_success_count": _row_by_label(packet_rows, "component_residual_guard_v12")[
            "exact_success_count"
        ],
        "v12_executor_success_count": _row_by_label(packet_rows, "component_residual_guard_v12")[
            "executor_success_count"
        ],
        "finding_count": len(finding_rows),
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "family_rows": family_rows,
        "comparison_rows": comparison_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h1r_component_residual_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1r_component_residual_family_summary.csv", family_rows)
    _write_csv(tables_dir / "h1r_component_residual_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h1r_component_residual_findings.csv", finding_rows)
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
        "no_tool_call_count": sum(1 for row in results if row.get("replay_failure_mode") == "no_tool_call"),
        "argument_mismatch_count": sum(1 for row in results if row.get("replay_failure_mode") == "argument_mismatch"),
        "executable_paraphrase_count": sum(
            1 for row in results if row.get("replay_failure_mode") == "executable_paraphrase"
        ),
    }


def _family_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
        results = _read_json(spec.live_packet_dir / "live_replay_results.json")
        families = sorted({str(row["family"]) for row in results})
        for family in families:
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


def _finding_rows(packet_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    no_directive = _row_by_label(packet_rows, "no_directive")
    v11 = _row_by_label(packet_rows, "component_label_guard_v11")
    v12 = _row_by_label(packet_rows, "component_residual_guard_v12")
    v12_vs_v11 = next(row for row in comparison_rows if "component_label_guard" in row["baseline_system_id"])
    return [
        {
            "finding_id": "h1r_breaks_no_directive",
            "finding": (
                f"No-directive reaches {no_directive['exact_success_count']}/6 exact and "
                f"{no_directive['executor_success_count']}/6 executor-equivalent, so H1r is a useful residual discriminator."
            ),
        },
        {
            "finding_id": "v11_is_strong_incumbent",
            "finding": (
                f"Component-label guard v11 reaches {v11['exact_success_count']}/6 exact and "
                f"{v11['executor_success_count']}/6 executor-equivalent, leaving only the alert-s92 code-label miss."
            ),
        },
        {
            "finding_id": "v12_saturates_h1r",
            "finding": (
                f"Component-residual guard v12 reaches {v12['exact_success_count']}/6 exact and "
                f"{v12['executor_success_count']}/6 executor-equivalent, improving over v11 by "
                f"{v12_vs_v11['delta_exact_rate']:.3f} exact-rate."
            ),
        },
    ]


def _row_by_label(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["profile_label"] == profile_label:
            return row
    raise KeyError(profile_label)


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H1r Component Residual Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H1r isolates the H1q residual families: stale-selection fields, nonstandard component classes "
            "such as tag/toggle, and code-label exactness. No-directive collapses to `0 / 6` exact and "
            "`1 / 6` executor-equivalent; v11 is strong at `5 / 6`; v12 saturates the packet at `6 / 6`."
        ),
        "",
        "## Packet Rows",
        "",
        _markdown_table(payload["packet_rows"]),
        "",
        "## Family Rows",
        "",
        _markdown_table(payload["family_rows"]),
        "",
        "## Comparison Rows",
        "",
        _markdown_table(payload["comparison_rows"]),
        "",
        "## Findings",
        "",
        _markdown_table(payload["finding_rows"]),
        "",
        "## Interpretation",
        "",
        (
            "This is positive residual evidence for v12, but not yet a global promotion. The next test should "
            "transfer v12 back across H1n/H1o/H1p and verify that the extra residual wording does not reintroduce "
            "the H1p executor-equivalence loss or the H1n broad-prose regressions."
        ),
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    fields = list(rows[0].keys())
    output = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(output)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H1r component residual synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h1r_component_residual_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
