from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_prompt_contract_probe_packet(packet_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    packet_path = Path(packet_dir)
    target = Path(output_dir) if output_dir else packet_path
    target.mkdir(parents=True, exist_ok=True)
    candidate_rows = _read_csv(packet_path / "candidate_summary.csv")
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for row in candidate_rows:
        case_deltas = _case_deltas(row)
        failure_counts = Counter(str(case.get("candidate_failure_mode", "")) for case in case_deltas if case.get("candidate_failure_mode"))
        improved_case_count = sum(1 for case in case_deltas if _int(case.get("delta_exact_match")) > 0 or _int(case.get("delta_executable_match")) > 0)
        regressed_case_count = sum(1 for case in case_deltas if _int(case.get("delta_exact_match")) < 0 or _int(case.get("delta_executable_match")) < 0)
        summary_row = {
            "system_id": row.get("system_id", ""),
            "tool_prompt_contract_id": row.get("tool_prompt_contract_id", ""),
            "tool_catalog_profile_id": row.get("tool_catalog_profile_id", ""),
            "exact_match_rate": row.get("exact_match_rate", ""),
            "executable_match_rate": row.get("executable_match_rate", ""),
            "delta_exact_vs_contracted": row.get("delta_exact_vs_contracted", ""),
            "delta_exact_vs_no_directive": row.get("delta_exact_vs_no_directive", ""),
            "probe_gate": row.get("probe_gate", ""),
            "improved_case_count": improved_case_count,
            "regressed_case_count": regressed_case_count,
            "dominant_failure_mode": failure_counts.most_common(1)[0][0] if failure_counts else "",
            "failure_modes": ";".join(f"{mode}:{count}" for mode, count in sorted(failure_counts.items())),
            "recommendation": _recommendation(row),
        }
        summary_rows.append(summary_row)
        for failure_mode, count in sorted(failure_counts.items()):
            failure_rows.append(
                {
                    "system_id": row.get("system_id", ""),
                    "tool_prompt_contract_id": row.get("tool_prompt_contract_id", ""),
                    "tool_catalog_profile_id": row.get("tool_catalog_profile_id", ""),
                    "failure_mode": failure_mode,
                    "count": count,
                }
            )

    summary_payload = {
        "packet_dir": str(packet_path.resolve()),
        "candidate_count": len(summary_rows),
        "candidate_gate_summary": summary_rows,
        "failure_mode_counts": failure_rows,
    }
    _write_csv(target / "candidate_gate_summary.csv", summary_rows)
    _write_csv(target / "candidate_failure_mode_counts.csv", failure_rows)
    (target / "candidate_gate_summary.md").write_text(_markdown_summary(summary_payload), encoding="utf-8")
    (target / "candidate_gate_summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an executed MLX prompt-contract probe packet.")
    parser.add_argument("packet_dir")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize_prompt_contract_probe_packet(args.packet_dir, output_dir=args.output_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _case_deltas(row: dict[str, str]) -> list[dict[str, str]]:
    comparison_path = row.get("no_directive_comparison_path", "")
    if not comparison_path:
        return []
    case_delta_path = Path(comparison_path).with_name("probe_case_deltas.csv")
    return _read_csv(case_delta_path)


def _recommendation(row: dict[str, str]) -> str:
    exact_rate = _float(row.get("exact_match_rate"))
    executable_rate = _float(row.get("executable_match_rate"))
    exact_delta_vs_no_directive = _float(row.get("delta_exact_vs_no_directive"))
    if exact_rate >= 0.5:
        return "strong_probe_candidate"
    if exact_delta_vs_no_directive > 0.0:
        return "weak_exact_gain"
    if executable_rate >= 1.0:
        return "visual_executable_gain_only"
    return "no_probe_gain"


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Prompt-Contract Probe Candidate Gate Summary",
        "",
        f"Packet: `{payload['packet_dir']}`",
        "",
        "| contract | exact | executable | delta exact vs no-directive | improved cases | dominant failure | recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["candidate_gate_summary"]:
        lines.append(
            "| {contract} | {exact} | {executable} | {delta} | {improved} | {failure} | {recommendation} |".format(
                contract=row["tool_prompt_contract_id"],
                exact=row["exact_match_rate"],
                executable=row["executable_match_rate"],
                delta=row["delta_exact_vs_no_directive"],
                improved=row["improved_case_count"],
                failure=row["dominant_failure_mode"],
                recommendation=row["recommendation"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `weak_exact_gain` means the candidate recovered at least one exact probe case over no-directive, but remains far below contracted MLX.",
            "- `visual_executable_gain_only` means the candidate recovered the visual executor target without improving exact JSON copy rate.",
            "- Candidates should move to H1i only as mechanism probes, not as assumed replacements for the final tool-turn directive.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _int(value: Any) -> int:
    if value in {None, ""}:
        return 0
    return int(value)


def _float(value: Any) -> float:
    if value in {None, ""}:
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
