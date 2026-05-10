from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2c_transfer_probe_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    packet_label: str
    profile_label: str
    packet_dir: Path


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_label: str
    comparison_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "h2b_residual_fit",
        "h2c_scoped_residual_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "component_label_guard_v11",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1x_v11_breaker_component_label_guard_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "component_residual_guard_v12",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h1x_v11_breaker_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "h2a_stale_selection_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1",
    ),
    PacketSpec(
        "h1x_transfer",
        "h2c_scoped_residual_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2c_scoped_residual_gate_on_h1x_execute_v1",
    ),
)

COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        "h1x_h2c_vs_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2c_scoped_residual_gate_vs_component_label_guard_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x_h2c_vs_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2c_scoped_residual_gate_vs_component_residual_guard_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x_h2c_vs_h2a",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2c_scoped_residual_gate_vs_h2a_on_h1x_v1",
    ),
    ComparisonSpec(
        "h1x_h2c_vs_no_directive",
        ROOT
        / "results"
        / "tool_probe_replay_live_comparisons"
        / "20260510T_h2c_scoped_residual_gate_vs_no_directive_on_h1x_v1",
    ),
)


def build_h2c_transfer_probe_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    comparison_rows = [_comparison_row(spec) for spec in COMPARISON_SPECS]
    h2c_h1x_non_exact_rows = _non_exact_rows(
        "h1x_transfer",
        "h2c_scoped_residual_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2c_scoped_residual_gate_on_h1x_execute_v1",
    )
    finding_rows = _finding_rows(packet_rows, comparison_rows, h2c_h1x_non_exact_rows)
    h2c_h2b = _row_by_label(packet_rows, "h2b_residual_fit", "h2c_scoped_residual_gate")
    h2c_h1x = _row_by_label(packet_rows, "h1x_transfer", "h2c_scoped_residual_gate")
    h2c_vs_h2a = _comparison_by_label(comparison_rows, "h1x_h2c_vs_h2a")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "packet_row_count": len(packet_rows),
        "comparison_count": len(comparison_rows),
        "h2c_h2b_exact_success_count": int(h2c_h2b["exact_success_count"]),
        "h2c_h2b_executor_success_count": int(h2c_h2b["executor_success_count"]),
        "h2c_h1x_exact_success_count": int(h2c_h1x["exact_success_count"]),
        "h2c_h1x_executor_success_count": int(h2c_h1x["executor_success_count"]),
        "h1x_delta_exact_vs_h2a": h2c_vs_h2a["delta_exact_rate"],
        "h1x_delta_executor_vs_h2a": h2c_vs_h2a["delta_executor_equivalence_rate"],
        "promotion_decision": "reject_global_h2c_build_h2d_class_preserving_route",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "comparison_rows": comparison_rows,
        "h2c_h1x_non_exact_rows": h2c_h1x_non_exact_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h2c_transfer_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2c_transfer_comparison_summary.csv", comparison_rows)
    _write_csv(tables_dir / "h2c_transfer_h1x_non_exact_rows.csv", h2c_h1x_non_exact_rows)
    _write_csv(tables_dir / "h2c_transfer_findings.csv", finding_rows)
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
        "packet_label": spec.packet_label,
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


def _non_exact_rows(packet_label: str, profile_label: str, packet_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_json(packet_dir / "live_replay_results.json"):
        if row.get("replay_exact_match") is True:
            continue
        detail = _probe_detail(row)
        rows.append(
            {
                "packet_label": packet_label,
                "profile_label": profile_label,
                "case_id": row["case_id"],
                "family": row.get("family", ""),
                "failure_mode": row.get("replay_failure_mode", ""),
                "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                "expected_tool": detail["expected_tool"],
                "expected_arguments": detail["expected_arguments"],
                "actual_tool": detail["actual_tool"],
                "actual_arguments": detail["actual_arguments"],
            }
        )
    return rows


def _probe_detail(row: dict[str, Any]) -> dict[str, str]:
    probe_path = Path(row["output_dir"]) / "probe_results.json"
    probe = _read_json(probe_path)[0]
    expected = (probe.get("expected_calls") or [{}])[0]
    actual = (probe.get("actual_calls") or [{}])[0]
    return {
        "expected_tool": str(expected.get("name", "")),
        "expected_arguments": json.dumps(expected.get("arguments", {}), sort_keys=True),
        "actual_tool": str(actual.get("name", "")),
        "actual_arguments": json.dumps(actual.get("arguments", {}), sort_keys=True),
    }


def _finding_rows(
    packet_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    non_exact_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    h2c_h2b = _row_by_label(packet_rows, "h2b_residual_fit", "h2c_scoped_residual_gate")
    h2c_h1x = _row_by_label(packet_rows, "h1x_transfer", "h2c_scoped_residual_gate")
    h2a_h1x = _row_by_label(packet_rows, "h1x_transfer", "h2a_stale_selection_gate")
    v12_h1x = _row_by_label(packet_rows, "h1x_transfer", "component_residual_guard_v12")
    miss = non_exact_rows[0] if non_exact_rows else {}
    return [
        {
            "finding_id": "h2c_local_fit_does_not_transfer_cleanly_to_h1x",
            "finding": (
                f"H2c is {h2c_h2b['exact_success_count']}/5 on H2b but only "
                f"{h2c_h1x['exact_success_count']}/8 on H1x, while H2a is "
                f"{h2a_h1x['exact_success_count']}/8 and v12 is {v12_h1x['exact_success_count']}/8."
            ),
        },
        {
            "finding_id": "h2c_regression_is_component_class_swap",
            "finding": (
                f"The H1x miss is {miss.get('case_id', '')}: expected {miss.get('expected_arguments', '')}, "
                f"but H2c produced {miss.get('actual_arguments', '')}. This is a class-swap overfit from "
                "`result pill` to `result chip`."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "Build H2d as a class-preserving scoped residual route: keep exact role-plus-component phrases "
                "from the prompt, but never substitute a component class learned from the H2b fit packet."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2c Transfer Probe Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2c saturates H2b locally but fails the first held-out H1x transfer check. The failure is precise: "
            "it turns the expected `result chip` into `result pill`, indicating class-swap overfit from the H2b "
            "fit row. The next candidate should preserve the exact component class named in the prompt."
        ),
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Comparison Rows",
        "",
        _table(payload["comparison_rows"]),
        "",
        "## H2c H1x Non-Exact Rows",
        "",
        _table(payload["h2c_h1x_non_exact_rows"]),
        "",
        "## Findings",
        "",
        _table(payload["finding_rows"]),
        "",
    ]
    return "\n".join(lines)


def _row_by_label(rows: list[dict[str, Any]], packet_label: str, profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["packet_label"] == packet_label and row["profile_label"] == profile_label:
            return row
    raise KeyError((packet_label, profile_label))


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H2c transfer probe synthesis report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2c_transfer_probe_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
