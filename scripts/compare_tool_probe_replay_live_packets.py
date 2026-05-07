from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "tool_probe_replay_live_comparisons"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two live exact tool-probe replay packets.")
    parser.add_argument("baseline_packet")
    parser.add_argument("candidate_packet")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_tool_probe_replay_live_packets(
        baseline_packet=Path(args.baseline_packet),
        candidate_packet=Path(args.candidate_packet),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(comparison["summary"], indent=2, ensure_ascii=False))


def compare_tool_probe_replay_live_packets(
    *,
    baseline_packet: Path,
    candidate_packet: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    baseline_manifest = _read_json(baseline_packet / "manifest.json")
    candidate_manifest = _read_json(candidate_packet / "manifest.json")
    baseline_summary = _read_json(baseline_packet / "summary.json")
    candidate_summary = _read_json(candidate_packet / "summary.json")
    baseline_rows = {str(row["case_id"]): row for row in _read_json(baseline_packet / "live_replay_results.json")}
    candidate_rows = {str(row["case_id"]): row for row in _read_json(candidate_packet / "live_replay_results.json")}
    shared_case_ids = sorted(set(baseline_rows) & set(candidate_rows))

    case_deltas = []
    for case_id in shared_case_ids:
        baseline = baseline_rows[case_id]
        candidate = candidate_rows[case_id]
        case_deltas.append(
            {
                "case_id": case_id,
                "family": candidate.get("family") or baseline.get("family", ""),
                "source_failure_mode": candidate.get("source_failure_mode") or baseline.get("source_failure_mode", ""),
                "baseline_replay_exact_match": bool(baseline.get("replay_exact_match")),
                "candidate_replay_exact_match": bool(candidate.get("replay_exact_match")),
                "delta_exact_match": _bool_delta(candidate.get("replay_exact_match"), baseline.get("replay_exact_match")),
                "baseline_replay_executable_match": _optional_bool(baseline.get("replay_executable_match")),
                "candidate_replay_executable_match": _optional_bool(candidate.get("replay_executable_match")),
                "delta_executable_match": _optional_bool_delta(
                    candidate.get("replay_executable_match"),
                    baseline.get("replay_executable_match"),
                ),
                "baseline_replay_failure_mode": baseline.get("replay_failure_mode", ""),
                "candidate_replay_failure_mode": candidate.get("replay_failure_mode", ""),
                "baseline_actual_call_count": int(baseline.get("replay_actual_call_count") or 0),
                "candidate_actual_call_count": int(candidate.get("replay_actual_call_count") or 0),
                "delta_actual_call_count": int(candidate.get("replay_actual_call_count") or 0)
                - int(baseline.get("replay_actual_call_count") or 0),
            }
        )

    summary = {
        "baseline_packet": str(baseline_packet.resolve()),
        "candidate_packet": str(candidate_packet.resolve()),
        "baseline_packet_run_id": baseline_manifest.get("packet_run_id", baseline_packet.name),
        "candidate_packet_run_id": candidate_manifest.get("packet_run_id", candidate_packet.name),
        "baseline_system_id": baseline_manifest.get("system_id", ""),
        "candidate_system_id": candidate_manifest.get("system_id", ""),
        "shared_case_count": len(shared_case_ids),
        "baseline_exact_rate": float(baseline_summary.get("exact_rate") or 0.0),
        "candidate_exact_rate": float(candidate_summary.get("exact_rate") or 0.0),
        "delta_exact_rate": float(candidate_summary.get("exact_rate") or 0.0)
        - float(baseline_summary.get("exact_rate") or 0.0),
        "shared_executable_case_count": sum(
            1
            for row in case_deltas
            if row["baseline_replay_executable_match"] is not None
            and row["candidate_replay_executable_match"] is not None
        ),
        "baseline_executable_rate": _optional_rate(
            sum(1 for row in case_deltas if row["baseline_replay_executable_match"] is True),
            sum(
                1
                for row in case_deltas
                if row["baseline_replay_executable_match"] is not None
                and row["candidate_replay_executable_match"] is not None
            ),
        ),
        "candidate_executable_rate": _optional_rate(
            sum(1 for row in case_deltas if row["candidate_replay_executable_match"] is True),
            sum(
                1
                for row in case_deltas
                if row["baseline_replay_executable_match"] is not None
                and row["candidate_replay_executable_match"] is not None
            ),
        ),
        "case_delta_count": len(case_deltas),
    }
    if summary["baseline_executable_rate"] is not None and summary["candidate_executable_rate"] is not None:
        summary["delta_executable_rate"] = summary["candidate_executable_rate"] - summary["baseline_executable_rate"]
    else:
        summary["delta_executable_rate"] = None
    target = output_dir or DEFAULT_OUTPUT_ROOT / f"{candidate_packet.name}_vs_{baseline_packet.name}"
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "live_replay_comparison.json", {"summary": summary, "case_deltas": case_deltas})
    _write_csv(target / "live_replay_case_deltas.csv", case_deltas)
    (target / "live_replay_summary.md").write_text(_summary_markdown(summary, case_deltas), encoding="utf-8")
    return {
        "output_dir": str(target.resolve()),
        "summary": summary,
        "case_deltas": case_deltas,
    }


def _summary_markdown(summary: dict[str, Any], case_deltas: list[dict[str, Any]]) -> str:
    lines = [
        "# Live Exact Replay Comparison",
        "",
        f"- Baseline system: `{summary['baseline_system_id']}`",
        f"- Candidate system: `{summary['candidate_system_id']}`",
        f"- Baseline exact rate: `{summary['baseline_exact_rate']}`",
        f"- Candidate exact rate: `{summary['candidate_exact_rate']}`",
        f"- Delta exact rate: `{summary['delta_exact_rate']}`",
        f"- Baseline executable rate: `{summary['baseline_executable_rate']}`",
        f"- Candidate executable rate: `{summary['candidate_executable_rate']}`",
        f"- Delta executable rate: `{summary['delta_executable_rate']}`",
        "",
        "| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline calls | candidate calls | delta calls | candidate failure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in case_deltas:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_id"]),
                    str(row["family"]),
                    str(row["baseline_replay_exact_match"]),
                    str(row["candidate_replay_exact_match"]),
                    str(row["baseline_replay_executable_match"]),
                    str(row["candidate_replay_executable_match"]),
                    str(row["baseline_actual_call_count"]),
                    str(row["candidate_actual_call_count"]),
                    str(row["delta_actual_call_count"]),
                    str(row["candidate_replay_failure_mode"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _bool_delta(candidate: Any, baseline: Any) -> int:
    return int(bool(candidate)) - int(bool(baseline))


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    return bool(value)


def _optional_bool_delta(candidate: Any, baseline: Any) -> int | None:
    candidate_bool = _optional_bool(candidate)
    baseline_bool = _optional_bool(baseline)
    if candidate_bool is None or baseline_bool is None:
        return None
    return int(candidate_bool) - int(baseline_bool)


def _optional_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
