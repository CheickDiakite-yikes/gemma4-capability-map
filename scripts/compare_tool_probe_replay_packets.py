from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "tool_probe_replay_comparisons"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two exact tool-probe replay packets.")
    parser.add_argument("baseline_packet")
    parser.add_argument("candidate_packet")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_tool_probe_replay_packets(
        baseline_packet=Path(args.baseline_packet),
        candidate_packet=Path(args.candidate_packet),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(comparison["summary"], indent=2, ensure_ascii=False))


def compare_tool_probe_replay_packets(
    *,
    baseline_packet: Path,
    candidate_packet: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    baseline_summary = _read_json(baseline_packet / "summary.json")
    candidate_summary = _read_json(candidate_packet / "summary.json")
    baseline_rows = {str(row["case_id"]): row for row in _read_json(baseline_packet / "replay_results.json")}
    candidate_rows = {str(row["case_id"]): row for row in _read_json(candidate_packet / "replay_results.json")}
    shared_case_ids = sorted(set(baseline_rows) & set(candidate_rows))
    case_rows = [_case_delta(case_id, baseline_rows[case_id], candidate_rows[case_id]) for case_id in shared_case_ids]
    family_rows = _family_rows(case_rows)
    summary = {
        "baseline_packet": str(baseline_packet.resolve()),
        "candidate_packet": str(candidate_packet.resolve()),
        "baseline_replay_system_id": baseline_summary.get("replay_system_id", ""),
        "candidate_replay_system_id": candidate_summary.get("replay_system_id", ""),
        "shared_case_count": len(shared_case_ids),
        "baseline_exact_match_rate": baseline_summary.get("replay_exact_match_rate", 0.0),
        "candidate_exact_match_rate": candidate_summary.get("replay_exact_match_rate", 0.0),
        "delta_exact_match_rate": float(candidate_summary.get("replay_exact_match_rate") or 0.0)
        - float(baseline_summary.get("replay_exact_match_rate") or 0.0),
        "case_delta_count": sum(1 for row in case_rows if int(row["delta_exact_match"]) != 0),
    }
    target = output_dir or DEFAULT_OUTPUT_ROOT / f"{candidate_packet.name}_vs_{baseline_packet.name}"
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "replay_comparison.json", {"summary": summary, "case_deltas": case_rows, "family_deltas": family_rows})
    _write_csv(target / "replay_case_deltas.csv", case_rows)
    _write_csv(target / "replay_family_deltas.csv", family_rows)
    return {
        "output_dir": str(target.resolve()),
        "summary": summary,
        "case_deltas": case_rows,
        "family_deltas": family_rows,
    }


def _case_delta(case_id: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": candidate.get("family") or baseline.get("family", ""),
        "baseline_failure_mode": baseline.get("replay_failure_mode", ""),
        "candidate_failure_mode": candidate.get("replay_failure_mode", ""),
        "baseline_exact_match": bool(baseline.get("replay_exact_match")),
        "candidate_exact_match": bool(candidate.get("replay_exact_match")),
        "delta_exact_match": int(bool(candidate.get("replay_exact_match"))) - int(bool(baseline.get("replay_exact_match"))),
        "baseline_executable_match": baseline.get("replay_executable_match"),
        "candidate_executable_match": candidate.get("replay_executable_match"),
        "baseline_actual_call_count": int(baseline.get("replay_actual_call_count") or 0),
        "candidate_actual_call_count": int(candidate.get("replay_actual_call_count") or 0),
        "delta_actual_call_count": int(candidate.get("replay_actual_call_count") or 0)
        - int(baseline.get("replay_actual_call_count") or 0),
    }


def _family_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["family"])].append(row)
    rows: list[dict[str, Any]] = []
    for family, family_cases in sorted(grouped.items()):
        total = len(family_cases)
        baseline_exact = sum(1 for row in family_cases if row["baseline_exact_match"])
        candidate_exact = sum(1 for row in family_cases if row["candidate_exact_match"])
        rows.append(
            {
                "family": family,
                "case_count": total,
                "baseline_exact_count": baseline_exact,
                "candidate_exact_count": candidate_exact,
                "delta_exact_count": candidate_exact - baseline_exact,
                "baseline_exact_rate": baseline_exact / total if total else 0.0,
                "candidate_exact_rate": candidate_exact / total if total else 0.0,
                "delta_exact_rate": (candidate_exact - baseline_exact) / total if total else 0.0,
            }
        )
    return rows


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
