from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h2b_residual_exactness_synthesis"


@dataclass(frozen=True)
class PacketSpec:
    profile_label: str
    packet_dir: Path


PACKET_SPECS: tuple[PacketSpec, ...] = (
    PacketSpec(
        "no_directive",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2b_residual_exactness_no_directive_execute_v1",
    ),
    PacketSpec(
        "component_label_guard_v11",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2b_residual_exactness_component_label_guard_execute_v1",
    ),
    PacketSpec(
        "component_value_guard_v9",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2b_residual_exactness_component_value_guard_execute_v1",
    ),
    PacketSpec(
        "component_residual_guard_v12",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2b_residual_exactness_component_residual_guard_execute_v1",
    ),
    PacketSpec(
        "code_label_exact_guard_v15",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260510T_h2b_residual_exactness_code_label_exact_guard_execute_v1",
    ),
    PacketSpec(
        "h2a_stale_selection_gate",
        ROOT / "results" / "tool_probe_replay_live" / "20260510T_h2b_residual_exactness_h2a_execute_v1",
    ),
)


def build_h2b_residual_exactness_synthesis(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_rows = [_packet_row(spec) for spec in PACKET_SPECS]
    case_rows = _case_rows()
    non_exact_rows = _non_exact_rows()
    finding_rows = _finding_rows(packet_rows, non_exact_rows)
    v12 = _row_by_label(packet_rows, "component_residual_guard_v12")
    v9 = _row_by_label(packet_rows, "component_value_guard_v9")
    h2a = _row_by_label(packet_rows, "h2a_stale_selection_gate")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "profile_count": len(packet_rows),
        "case_count": int(v12["case_count"]),
        "v12_exact_success_count": int(v12["exact_success_count"]),
        "v12_executor_success_count": int(v12["executor_success_count"]),
        "v9_exact_success_count": int(v9["exact_success_count"]),
        "v9_executor_success_count": int(v9["executor_success_count"]),
        "h2a_exact_success_count": int(h2a["exact_success_count"]),
        "h2a_executor_success_count": int(h2a["executor_success_count"]),
        "strict_winner": "component_residual_guard_v12",
        "executor_winners": ["component_residual_guard_v12", "component_value_guard_v9"],
        "promotion_decision": "do_not_globalize_v12_use_h2c_scoped_residual_route",
    }
    payload = {
        "manifest": manifest,
        "packet_rows": packet_rows,
        "case_rows": case_rows,
        "non_exact_rows": non_exact_rows,
        "finding_rows": finding_rows,
    }
    _write_csv(tables_dir / "h2b_residual_exactness_packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h2b_residual_exactness_case_matrix.csv", case_rows)
    _write_csv(tables_dir / "h2b_residual_exactness_non_exact_rows.csv", non_exact_rows)
    _write_csv(tables_dir / "h2b_residual_exactness_findings.csv", finding_rows)
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
        "no_tool_call_count": sum(1 for row in results if row.get("replay_failure_mode") == "no_tool_call"),
        "argument_mismatch_count": sum(1 for row in results if row.get("replay_failure_mode") == "argument_mismatch"),
        "executable_paraphrase_count": sum(
            1 for row in results if row.get("replay_failure_mode") == "executable_paraphrase"
        ),
    }


def _case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
        for row in _read_json(spec.packet_dir / "live_replay_results.json"):
            rows.append(
                {
                    "profile_label": spec.profile_label,
                    "case_id": row["case_id"],
                    "family": row.get("family", ""),
                    "source_failure_mode": row.get("source_failure_mode", ""),
                    "replay_failure_mode": row.get("replay_failure_mode", ""),
                    "exact_match": row.get("replay_exact_match"),
                    "executor_equivalence_match": row.get("replay_executor_equivalence_match"),
                }
            )
    return rows


def _non_exact_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PACKET_SPECS:
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
                    "expected_arguments": detail["expected_arguments"],
                    "actual_tool": detail["actual_tool"],
                    "actual_arguments": detail["actual_arguments"],
                    "actual_region_ids": detail["actual_region_ids"],
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


def _finding_rows(packet_rows: list[dict[str, Any]], non_exact_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    no_directive = _row_by_label(packet_rows, "no_directive")
    v11 = _row_by_label(packet_rows, "component_label_guard_v11")
    v9 = _row_by_label(packet_rows, "component_value_guard_v9")
    v12 = _row_by_label(packet_rows, "component_residual_guard_v12")
    v15 = _row_by_label(packet_rows, "code_label_exact_guard_v15")
    h2a = _row_by_label(packet_rows, "h2a_stale_selection_gate")
    v12_failures = ", ".join(row["case_id"] for row in non_exact_rows if row["profile_label"] == "component_residual_guard_v12")
    v9_failures = ", ".join(row["case_id"] for row in non_exact_rows if row["profile_label"] == "component_value_guard_v9")
    v15_failures = ", ".join(row["case_id"] for row in non_exact_rows if row["profile_label"] == "code_label_exact_guard_v15")
    return [
        {
            "finding_id": "h2b_is_a_real_residual_breaker",
            "finding": (
                f"No-directive reaches {no_directive['exact_success_count']}/5 strict and "
                f"{no_directive['executor_success_count']}/5 executor-equivalent; v11 reaches "
                f"{v11['exact_success_count']}/5 strict and {v11['executor_success_count']}/5 executor-equivalent. "
                "The packet preserves pressure instead of washing out the residual mechanism."
            ),
        },
        {
            "finding_id": "v12_is_strict_winner",
            "finding": (
                f"Component-residual guard v12 reaches {v12['exact_success_count']}/5 strict and "
                f"{v12['executor_success_count']}/5 executor-equivalent, the best strict score on H2b. "
                f"Its remaining miss is: {v12_failures}."
            ),
        },
        {
            "finding_id": "v9_ties_executor_but_not_exact",
            "finding": (
                f"Component-value guard v9 reaches {v9['exact_success_count']}/5 strict and "
                f"{v9['executor_success_count']}/5 executor-equivalent, tying v12 on executor-equivalence but "
                f"missing strict exactness on: {v9_failures}."
            ),
        },
        {
            "finding_id": "v15_solves_code_not_component_class",
            "finding": (
                f"Code-label exact guard v15 reaches {v15['exact_success_count']}/5 strict and "
                f"{v15['executor_success_count']}/5 executor-equivalent. It fixes the code-label rows plus result pill, "
                f"but misses the component-class rows: {v15_failures}."
            ),
        },
        {
            "finding_id": "h2a_is_not_residual_exactness_solution",
            "finding": (
                f"H2a reaches {h2a['exact_success_count']}/5 strict and {h2a['executor_success_count']}/5 "
                "executor-equivalent on H2b. It remains useful for stale-selection mediation, but does not solve "
                "the alias/code-label residual by itself."
            ),
        },
        {
            "finding_id": "next_slice",
            "finding": (
                "Do not globalize v12 despite the H2b win; H1s already showed transfer cost. The next move is H2c: "
                "a scoped residual route/factor that activates v12-like language only for exact alias/code-label "
                "contexts while preserving H2a's stale-selection controller gate."
            ),
        },
    ]


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H2b Residual Exactness Synthesis",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Summary",
        "",
        (
            "H2b composes the five residual cases left by the H2a transfer gate. V12 is the strict winner at "
            "`4 / 5` and `4 / 5` executor-equivalent. V9 ties executor-equivalence at `4 / 5` but only reaches "
            "`3 / 5` strict. V15 fixes the two code-label rows plus result pill but misses both component-class "
            "rows. H2a itself falls to `0 / 5` strict and `3 / 5` executor-equivalent, confirming it is a stale-"
            "selection helper, not an alias exactness solution."
        ),
        "",
        "## Packet Rows",
        "",
        _table(payload["packet_rows"]),
        "",
        "## Case Matrix",
        "",
        _table(payload["case_rows"]),
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


def _row_by_label(rows: list[dict[str, Any]], profile_label: str) -> dict[str, Any]:
    for row in rows:
        if row["profile_label"] == profile_label:
            return row
    raise KeyError(profile_label)


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
    parser = argparse.ArgumentParser(description="Build the H2b residual exactness synthesis packet.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h2b_residual_exactness_synthesis(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
