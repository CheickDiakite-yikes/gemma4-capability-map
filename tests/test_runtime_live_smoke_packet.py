from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gemma4_capability_map.runtime.core import LocalAgentRuntime


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_runtime_live_smoke_packet.py"
SPEC = importlib.util.spec_from_file_location("run_runtime_live_smoke_packet_script", MODULE_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(SCRIPT)


def test_runtime_live_smoke_packet_dry_run_writes_manifest(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    result = SCRIPT.run_packet(
        workflow_ids=["executive_visual_dashboard_review"],
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        output_root=tmp_path / "packets",
        run_group_id="packet_dry_run",
        dry_run=True,
        runtime=runtime,
    )

    output_dir = Path(result["output_dir"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert result["dry_run"] is True
    assert summary["workflow_count"] == 1
    assert manifest["workflows"][0]["workflow_id"] == "executive_visual_dashboard_review"
    assert manifest["commands"][0][0:4] == ["moonie-agent", "live", "--workflow-id", "executive_visual_dashboard_review"]


def test_runtime_live_smoke_packet_dry_run_records_repeat_plan(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    result = SCRIPT.run_packet(
        workflow_ids=["executive_visual_dashboard_review"],
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        output_root=tmp_path / "packets",
        run_group_id="packet_repeat_dry_run",
        repeat=3,
        dry_run=True,
        runtime=runtime,
    )

    output_dir = Path(result["output_dir"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["workflow_count"] == 1
    assert summary["repeat_count"] == 3
    assert summary["session_count"] == 3
    assert manifest["repeat_count"] == 3
    assert [row["repeat_index"] for row in manifest["launch_plan"]] == [1, 2, 3]


def test_runtime_live_smoke_packet_runs_oracle_workflow(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    result = SCRIPT.run_packet(
        workflow_ids=["executive_visual_dashboard_review"],
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        output_root=tmp_path / "packets",
        run_group_id="packet_real",
        dry_run=False,
        runtime=runtime,
    )

    output_dir = Path(result["output_dir"])
    sessions = json.loads((output_dir / "sessions.json").read_text(encoding="utf-8"))
    findings = json.loads((output_dir / "controller_findings.json").read_text(encoding="utf-8"))
    blocks = json.loads((output_dir / "policy_blocks.json").read_text(encoding="utf-8"))
    leaderboard = (output_dir / "leaderboard.csv").read_text(encoding="utf-8")
    assert result["workflow_count"] == 1
    assert result["failed_sessions"] == 0
    assert result["role_readiness_avg"] > 0.0
    assert sessions[0]["status"] == "completed"
    assert sessions[0]["sandbox_manifest_exists"] is True
    assert sessions[0]["artifact_count"] == 3
    assert isinstance(findings, list)
    assert isinstance(blocks, list)
    assert "workflow_id,episode_id" in leaderboard


def test_runtime_live_smoke_packet_repeats_oracle_workflow(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    result = SCRIPT.run_packet(
        workflow_ids=["executive_visual_dashboard_review"],
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        output_root=tmp_path / "packets",
        run_group_id="packet_repeat_real",
        repeat=2,
        dry_run=False,
        runtime=runtime,
    )

    output_dir = Path(result["output_dir"])
    sessions = json.loads((output_dir / "sessions.json").read_text(encoding="utf-8"))
    workflow_summaries = json.loads((output_dir / "workflow_summaries.json").read_text(encoding="utf-8"))
    workflow_summary_csv = (output_dir / "workflow_summary.csv").read_text(encoding="utf-8")
    assert result["workflow_count"] == 1
    assert result["repeat_count"] == 2
    assert result["session_count"] == 2
    assert result["failed_sessions"] == 0
    assert [row["repeat_index"] for row in sessions] == [1, 2]
    assert workflow_summaries[0]["workflow_id"] == "executive_visual_dashboard_review"
    assert workflow_summaries[0]["session_count"] == 2
    assert len(workflow_summaries[0]["session_ids"]) == 2
    assert "controller_fallback_avg" in workflow_summary_csv
