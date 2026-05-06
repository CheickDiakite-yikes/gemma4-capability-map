from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def analyze_runtime_live_smoke_packet(packet_dir: str | Path) -> dict[str, Any]:
    packet_path = Path(packet_dir)
    summary = _read_json(packet_path / "summary.json", default={})
    sessions = _read_json(packet_path / "sessions.json", default=[])
    findings = _read_json(packet_path / "controller_findings.json", default=[])
    policy_blocks = _read_json(packet_path / "policy_blocks.json", default=[])
    repeat_count = int(summary.get("repeat_count") or _max_repeat_index(sessions, findings, policy_blocks) or 1)
    repair_family_counts = _repair_family_counts(findings)
    policy_block_counts = _policy_block_counts(policy_blocks)
    workflow_stability = _workflow_stability(
        sessions=sessions,
        findings=findings,
        policy_blocks=policy_blocks,
        repeat_count=repeat_count,
    )
    return {
        "packet_dir": str(packet_path.resolve()),
        "run_group_id": summary.get("run_group_id", packet_path.name.removesuffix("_runtime_live_smoke_packet")),
        "workflow_count": int(summary.get("workflow_count") or len({row.get("workflow_id") for row in sessions})),
        "repeat_count": repeat_count,
        "session_count": int(summary.get("session_count") or len(sessions)),
        "failed_sessions": int(summary.get("failed_sessions") or 0),
        "status_counts": summary.get("status_counts", dict(Counter(str(row.get("status")) for row in sessions))),
        "role_readiness_avg": summary.get("role_readiness_avg"),
        "strict_interface_avg": summary.get("strict_interface_avg"),
        "recovered_execution_avg": summary.get("recovered_execution_avg"),
        "controller_repair_avg": summary.get("controller_repair_avg"),
        "argument_repair_avg": summary.get("argument_repair_avg"),
        "controller_fallback_avg": summary.get("controller_fallback_avg"),
        "raw_planning_clean_rate_avg": summary.get("raw_planning_clean_rate_avg"),
        "controller_finding_count": len(findings),
        "policy_block_count": len(policy_blocks),
        "approval_count": int(summary.get("approval_count") or _sum_int(sessions, "approval_count")),
        "stable_repair_family_count": sum(1 for row in repair_family_counts if len(row["repeat_indexes"]) >= repeat_count),
        "stable_policy_block_family_count": sum(1 for row in policy_block_counts if len(row["repeat_indexes"]) >= repeat_count),
        "repair_family_counts": repair_family_counts,
        "policy_block_counts": policy_block_counts,
        "workflow_stability": workflow_stability,
    }


def write_runtime_packet_analysis(packet_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    packet_path = Path(packet_dir)
    output_path = Path(output_dir) if output_dir else packet_path
    output_path.mkdir(parents=True, exist_ok=True)
    analysis = analyze_runtime_live_smoke_packet(packet_path)
    paths = {
        "analysis": str((output_path / "runtime_packet_analysis.json").resolve()),
        "repair_family_counts": str((output_path / "runtime_repair_family_counts.csv").resolve()),
        "policy_block_counts": str((output_path / "runtime_policy_block_counts.csv").resolve()),
        "workflow_stability": str((output_path / "runtime_workflow_stability.csv").resolve()),
    }
    _write_json(Path(paths["analysis"]), analysis)
    _write_csv(Path(paths["repair_family_counts"]), analysis["repair_family_counts"])
    _write_csv(Path(paths["policy_block_counts"]), analysis["policy_block_counts"])
    _write_csv(Path(paths["workflow_stability"]), analysis["workflow_stability"])
    return paths


def _repair_family_counts(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        repair_family = "+".join(str(note) for note in finding.get("repair_notes") or ["unknown"])
        key = (str(finding.get("workflow_id", "")), str(finding.get("task_id", "")), repair_family)
        grouped[key].append(finding)
    rows: list[dict[str, Any]] = []
    for (workflow_id, task_id, repair_family), group in sorted(grouped.items()):
        rows.append(
            {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "repair_family": repair_family,
                "count": len(group),
                "repeat_indexes": _sorted_unique(row.get("repeat_index") for row in group),
                "session_ids": [str(row.get("session_id")) for row in group],
                "raw_output_examples": _unique_examples(group, "raw_outputs"),
            }
        )
    return rows


def _policy_block_counts(policy_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in policy_blocks:
        key = (
            str(block.get("workflow_id", "")),
            str(block.get("submission_gate", "")),
            str(block.get("action", "")),
        )
        grouped[key].append(block)
    rows: list[dict[str, Any]] = []
    for (workflow_id, submission_gate, action), group in sorted(grouped.items()):
        rows.append(
            {
                "workflow_id": workflow_id,
                "submission_gate": submission_gate,
                "action": action,
                "count": len(group),
                "repeat_indexes": _sorted_unique(row.get("repeat_index") for row in group),
                "targets": _sorted_unique(row.get("target") for row in group),
                "sandbox_endpoints": _sorted_unique(row.get("sandbox_endpoint") for row in group),
                "blocked_reasons": _sorted_unique(row.get("blocked_reason") for row in group if row.get("blocked_reason")),
            }
        )
    return rows


def _workflow_stability(
    *,
    sessions: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    policy_blocks: list[dict[str, Any]],
    repeat_count: int,
) -> list[dict[str, Any]]:
    workflows = sorted({str(row.get("workflow_id")) for row in sessions})
    rows: list[dict[str, Any]] = []
    for workflow_id in workflows:
        workflow_sessions = [row for row in sessions if str(row.get("workflow_id")) == workflow_id]
        finding_patterns = _patterns_by_repeat(
            [
                (
                    row.get("repeat_index"),
                    str(row.get("task_id", "")),
                    "+".join(str(note) for note in row.get("repair_notes") or ["unknown"]),
                )
                for row in findings
                if str(row.get("workflow_id")) == workflow_id
            ],
            repeat_count=repeat_count,
        )
        policy_patterns = _patterns_by_repeat(
            [
                (
                    row.get("repeat_index"),
                    str(row.get("submission_gate", "")),
                    str(row.get("action", "")),
                )
                for row in policy_blocks
                if str(row.get("workflow_id")) == workflow_id
            ],
            repeat_count=repeat_count,
        )
        rows.append(
            {
                "workflow_id": workflow_id,
                "session_count": len(workflow_sessions),
                "status_counts": dict(Counter(str(row.get("status")) for row in workflow_sessions)),
                "finding_patterns_by_repeat": finding_patterns,
                "stable_finding_pattern": _all_patterns_match(finding_patterns),
                "policy_patterns_by_repeat": policy_patterns,
                "stable_policy_pattern": _all_patterns_match(policy_patterns),
                "controller_repair_avg": _average(row.get("controller_repair_count") for row in workflow_sessions),
                "argument_repair_avg": _average(row.get("argument_repair_count") for row in workflow_sessions),
                "controller_fallback_avg": _average(row.get("controller_fallback_count") for row in workflow_sessions),
                "raw_planning_clean_rate_avg": _average(row.get("raw_planning_clean_rate") for row in workflow_sessions),
            }
        )
    return rows


def _patterns_by_repeat(items: list[tuple[Any, str, str]], *, repeat_count: int) -> dict[str, list[str]]:
    patterns: dict[str, list[str]] = {str(index): [] for index in range(1, repeat_count + 1)}
    for repeat_index, first, second in items:
        patterns.setdefault(str(repeat_index), []).append(f"{first}:{second}")
    return {key: sorted(value) for key, value in patterns.items()}


def _all_patterns_match(patterns: dict[str, list[str]]) -> bool:
    values = list(patterns.values())
    return all(value == values[0] for value in values[1:]) if values else True


def _max_repeat_index(*groups: list[dict[str, Any]]) -> int:
    values = [int(row.get("repeat_index") or 0) for group in groups for row in group]
    return max(values) if values else 0


def _sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def _average(values: Any) -> float:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _sorted_unique(values: Any) -> list[Any]:
    return sorted({value for value in values if value is not None})


def _unique_examples(rows: list[dict[str, Any]], key: str, *, limit: int = 3) -> list[Any]:
    examples: list[Any] = []
    seen: set[str] = set()
    for row in rows:
        for value in row.get(key) or []:
            marker = json.dumps(value, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            examples.append(value)
            if len(examples) >= limit:
                return examples
    return examples


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value
