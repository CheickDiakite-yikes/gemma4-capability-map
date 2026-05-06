from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.runtime.operator import session_inspection_payload
from gemma4_capability_map.runtime.sandbox import DEFAULT_SANDBOX_POLICY_ID


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "runtime_live_smoke_packets"
DEFAULT_WORKFLOW_IDS = ["executive_visual_dashboard_review"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a packaged-workflow live runtime smoke packet.")
    parser.add_argument("--workflow-id", action="append", dest="workflow_ids", default=[])
    parser.add_argument("--system-id", default="mlx_gemma4_e2b_reasoner_only")
    parser.add_argument("--lane", default="replayable_core")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--sandbox-mode", choices=["ephemeral_copy", "disabled"], default="ephemeral_copy")
    parser.add_argument("--sandbox-policy-id", default=DEFAULT_SANDBOX_POLICY_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_packet(
        workflow_ids=args.workflow_ids or DEFAULT_WORKFLOW_IDS,
        system_id=args.system_id,
        lane=args.lane,
        output_root=Path(args.output_root),
        results_root=Path(args.results_root) if args.results_root else None,
        run_group_id=args.run_group_id,
        sandbox_mode=args.sandbox_mode,
        sandbox_policy_id=args.sandbox_policy_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("failed_sessions"):
        raise SystemExit(1)


def run_packet(
    *,
    workflow_ids: list[str],
    system_id: str,
    lane: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    results_root: Path | None = None,
    run_group_id: str | None = None,
    sandbox_mode: str = "ephemeral_copy",
    sandbox_policy_id: str = DEFAULT_SANDBOX_POLICY_ID,
    dry_run: bool = False,
    runtime: LocalAgentRuntime | None = None,
) -> dict[str, Any]:
    if runtime is None:
        runtime = LocalAgentRuntime(results_root=results_root) if results_root else LocalAgentRuntime()
    run_group_id = run_group_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_dir = output_root / f"{run_group_id}_runtime_live_smoke_packet"
    packet_dir.mkdir(parents=True, exist_ok=True)
    workflows = {row["workflow_id"]: row for row in runtime.list_workflows(lane=lane)}
    missing = [workflow_id for workflow_id in workflow_ids if workflow_id not in workflows]
    if missing:
        raise SystemExit(f"Unknown packaged workflows for lane `{lane}`: {', '.join(missing)}")

    manifest = {
        "run_group_id": run_group_id,
        "created_at": datetime.now(UTC).isoformat(),
        "system_id": system_id,
        "lane": lane,
        "sandbox_mode": sandbox_mode,
        "sandbox_policy_id": sandbox_policy_id,
        "dry_run": dry_run,
        "workflows": [workflows[workflow_id] for workflow_id in workflow_ids],
        "commands": [
            [
                "moonie-agent",
                "live",
                "--workflow-id",
                workflow_id,
                "--system-id",
                system_id,
                "--lane",
                lane,
                "--sandbox-mode",
                sandbox_mode,
                "--sandbox-policy-id",
                sandbox_policy_id,
            ]
            for workflow_id in workflow_ids
        ],
    }
    _write_json(packet_dir / "manifest.json", manifest)
    if dry_run:
        summary = {
            "run_group_id": run_group_id,
            "dry_run": True,
            "workflow_count": len(workflow_ids),
            "output_dir": str(packet_dir.resolve()),
        }
        _write_json(packet_dir / "summary.json", summary)
        return summary

    rows: list[dict[str, Any]] = []
    for workflow_id in workflow_ids:
        session = runtime.launch_session(
            workflow_id=workflow_id,
            system_id=system_id,
            lane=lane,
            background=False,
            sandbox_mode=sandbox_mode,
            sandbox_policy_id=sandbox_policy_id,
        )
        rows.append(_session_row(runtime, session.session_id))

    summary = _summary(run_group_id, packet_dir, rows)
    _write_json(packet_dir / "sessions.json", rows)
    _write_json(packet_dir / "summary.json", summary)
    _write_csv(packet_dir / "leaderboard.csv", rows)
    return summary


def _session_row(runtime: LocalAgentRuntime, session_id: str) -> dict[str, Any]:
    session = runtime.get_session(session_id)
    runtime_trace = session.runtime_trace
    scorecard = session_inspection_payload(runtime, session_id, target="scorecard").get("scorecard", {})
    metrics = dict(scorecard.get("metrics", session.metrics))
    return {
        "session_id": session.session_id,
        "workflow_id": session.workflow_id,
        "episode_id": session.episode_id,
        "system_id": session.system_id,
        "lane": session.lane,
        "status": session.status.value,
        "sandbox_mode": session.sandbox_mode,
        "sandbox_policy_id": session.sandbox_policy_id,
        "sandbox_root": session.sandbox_root,
        "sandbox_manifest_exists": bool(session.sandbox_manifest_path and Path(session.sandbox_manifest_path).exists()),
        "artifact_count": len(session.artifact_paths),
        "policy_block_count": len(session.sandbox_policy_blocks),
        "approval_count": len(session.approvals),
        "controller_finding_count": len(scorecard.get("controller_findings", [])),
        "summary_path": runtime_trace.summary_path if runtime_trace else "",
        "episode_trace_path": runtime_trace.episode_trace_path if runtime_trace else "",
        "role_readiness_score": metrics.get("role_readiness_score"),
        "strict_interface_score": metrics.get("strict_interface_score"),
        "recovered_execution_score": metrics.get("recovered_execution_score"),
        "controller_repair_count": metrics.get("controller_repair_count"),
        "argument_repair_count": metrics.get("argument_repair_count"),
        "controller_fallback_count": metrics.get("controller_fallback_count"),
        "raw_planning_clean_rate": metrics.get("raw_planning_clean_rate"),
    }


def _summary(run_group_id: str, packet_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_sessions = sum(1 for row in rows if row["status"] == "failed")
    return {
        "run_group_id": run_group_id,
        "dry_run": False,
        "workflow_count": len(rows),
        "failed_sessions": failed_sessions,
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "output_dir": str(packet_dir.resolve()),
        "role_readiness_avg": _average(row.get("role_readiness_score") for row in rows),
        "strict_interface_avg": _average(row.get("strict_interface_score") for row in rows),
        "recovered_execution_avg": _average(row.get("recovered_execution_score") for row in rows),
        "controller_repair_avg": _average(row.get("controller_repair_count") for row in rows),
        "argument_repair_avg": _average(row.get("argument_repair_count") for row in rows),
        "controller_fallback_avg": _average(row.get("controller_fallback_count") for row in rows),
        "raw_planning_clean_rate_avg": _average(row.get("raw_planning_clean_rate") for row in rows),
        "controller_finding_count": sum(int(row.get("controller_finding_count") or 0) for row in rows),
        "policy_block_count": sum(int(row.get("policy_block_count") or 0) for row in rows),
        "approval_count": sum(int(row.get("approval_count") or 0) for row in rows),
    }


def _average(values: Any) -> float:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
