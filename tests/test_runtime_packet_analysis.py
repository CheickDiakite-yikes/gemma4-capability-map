from __future__ import annotations

import json
from pathlib import Path

from gemma4_capability_map.runtime.packet_analysis import (
    analyze_runtime_live_smoke_packet,
    write_runtime_packet_analysis,
)


def test_runtime_packet_analysis_groups_repeated_families(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    _write_json(
        packet_dir / "summary.json",
        {
            "run_group_id": "packet",
            "workflow_count": 1,
            "repeat_count": 2,
            "session_count": 2,
            "failed_sessions": 0,
            "status_counts": {"awaiting_approval": 2},
            "controller_repair_avg": 1.0,
            "argument_repair_avg": 0.5,
            "controller_fallback_avg": 0.5,
            "raw_planning_clean_rate_avg": 0.0,
            "approval_count": 2,
        },
    )
    _write_json(
        packet_dir / "sessions.json",
        [
            {
                "repeat_index": 1,
                "workflow_id": "jobs_visual_form_hold",
                "status": "awaiting_approval",
                "controller_repair_count": 1.0,
                "argument_repair_count": 0.5,
                "controller_fallback_count": 0.5,
                "raw_planning_clean_rate": 0.0,
                "approval_count": 1,
            },
            {
                "repeat_index": 2,
                "workflow_id": "jobs_visual_form_hold",
                "status": "awaiting_approval",
                "controller_repair_count": 1.0,
                "argument_repair_count": 0.5,
                "controller_fallback_count": 0.5,
                "raw_planning_clean_rate": 0.0,
                "approval_count": 1,
            },
        ],
    )
    _write_json(
        packet_dir / "controller_findings.json",
        [
            {
                "repeat_index": 1,
                "workflow_id": "jobs_visual_form_hold",
                "task_id": "tool_021_jobs_cli_patch_only_latest_email_fix",
                "repair_notes": ["repaired_arguments:cli_apply_patch"],
                "session_id": "session-1",
                "raw_outputs": ["raw call"],
            },
            {
                "repeat_index": 2,
                "workflow_id": "jobs_visual_form_hold",
                "task_id": "tool_021_jobs_cli_patch_only_latest_email_fix",
                "repair_notes": ["repaired_arguments:cli_apply_patch"],
                "session_id": "session-2",
                "raw_outputs": ["raw call"],
            },
        ],
    )
    _write_json(
        packet_dir / "policy_blocks.json",
        [
            {
                "repeat_index": 1,
                "workflow_id": "jobs_visual_form_hold",
                "submission_gate": "approval_required",
                "action": "attempt_submit",
                "target": "sandbox://jobs/submit",
                "sandbox_endpoint": "https://sandbox.local/jobs/submit",
            },
            {
                "repeat_index": 2,
                "workflow_id": "jobs_visual_form_hold",
                "submission_gate": "approval_required",
                "action": "attempt_submit",
                "target": "sandbox://jobs/submit",
                "sandbox_endpoint": "https://sandbox.local/jobs/submit",
            },
        ],
    )

    analysis = analyze_runtime_live_smoke_packet(packet_dir)

    assert analysis["repeat_count"] == 2
    assert analysis["controller_finding_count"] == 2
    assert analysis["stable_repair_family_count"] == 1
    assert analysis["stable_policy_block_family_count"] == 1
    assert analysis["repair_family_counts"][0]["count"] == 2
    assert analysis["repair_family_counts"][0]["repeat_indexes"] == [1, 2]
    assert analysis["workflow_stability"][0]["stable_finding_pattern"] is True
    assert analysis["workflow_stability"][0]["stable_policy_pattern"] is True


def test_write_runtime_packet_analysis_outputs_json_and_csv(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    _write_json(packet_dir / "summary.json", {"run_group_id": "packet", "repeat_count": 1})
    _write_json(packet_dir / "sessions.json", [])
    _write_json(packet_dir / "controller_findings.json", [])
    _write_json(packet_dir / "policy_blocks.json", [])

    paths = write_runtime_packet_analysis(packet_dir)

    assert Path(paths["analysis"]).exists()
    assert Path(paths["repair_family_counts"]).exists()
    assert Path(paths["policy_block_counts"]).exists()
    assert Path(paths["workflow_stability"]).exists()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
