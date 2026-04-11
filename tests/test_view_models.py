from __future__ import annotations

import csv
import json
from pathlib import Path

from gemma4_capability_map.app.view_models import build_console_snapshot, build_session_snapshot, default_session_id
from gemma4_capability_map.reporting.knowledge_work_board import build_board_rows, write_board_exports
from gemma4_capability_map.runtime.core import LocalAgentRuntime


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "model_registry.yaml"


def _write_snapshot(
    root: Path,
    name: str,
    manifest: dict,
    summary: dict,
    leaderboard_rows: list[dict[str, object]],
) -> None:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (target / "episode_leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "episode_id",
                "role_family",
                "lane",
                "workspace_id",
                "benchmark_tags",
                "artifact_quality_score",
                "browser_workflow_score",
                "strict_interface_score",
                "recovered_execution_score",
                "revision_responsiveness",
                "memory_retention_score",
                "escalation_correctness",
                "collateral_damage_free",
                "human_time_ratio",
                "role_readiness_score",
            ],
        )
        writer.writeheader()
        for row in leaderboard_rows:
            writer.writerow(row)


def test_console_snapshot_exposes_lane_specific_workflow_cards(tmp_path: Path) -> None:
    manifest = {
        "run_group_id": "20260411T150000Z_replayable_core",
        "created_at": "20260411T150000Z",
        "lane": "replayable_core",
        "run_intent": "exploratory",
        "backend": "hf_service",
        "reasoner": "google/gemma-4-E2B-it",
        "router": "google/functiongemma-270m-it",
        "retriever": "google/embeddinggemma-300m",
        "reasoner_backend": "hf_service",
        "router_backend": "hf_service",
        "retriever_backend": "hf_service",
        "episode_count": 1,
        "episodes_path": str((ROOT / "data" / "knowledge_work" / "replayable_core" / "episodes.jsonl").resolve()),
    }
    summary = {
        "runs": 1,
        "artifact_quality_avg": 1.0,
        "browser_workflow_avg": 1.0,
        "strict_interface_avg": 1.0,
        "recovered_execution_avg": 1.0,
        "real_world_readiness_avg": 0.97,
        "escalation_correctness_avg": 1.0,
    }
    leaderboard_rows = [
        {
            "run_id": "run-pass",
            "episode_id": "ep-pass",
            "role_family": "executive_assistant",
            "lane": "replayable_core",
            "workspace_id": "ws-pass",
            "benchmark_tags": "knowledge_work_arena,replayable_core,executive_assistant",
            "artifact_quality_score": 1.0,
            "browser_workflow_score": 1.0,
            "strict_interface_score": 1.0,
            "recovered_execution_score": 1.0,
            "revision_responsiveness": 1.0,
            "memory_retention_score": 1.0,
            "escalation_correctness": 1.0,
            "collateral_damage_free": 1.0,
            "human_time_ratio": 0.1,
            "role_readiness_score": 0.97,
        }
    ]
    _write_snapshot(tmp_path / "results" / "knowledge_work", "model_backed_hf_specialists_test", manifest, summary, leaderboard_rows)
    history_dir = tmp_path / "results" / "history"
    rows = build_board_rows(tmp_path / "results" / "knowledge_work", REGISTRY_PATH)
    write_board_exports(rows, history_dir)

    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    snapshot = build_console_snapshot(runtime, history_dir / "knowledge_work_board_latest.csv")

    assert snapshot["workflow_cards_by_lane"]["replayable_core"]
    assert snapshot["workflow_cards_by_lane"]["live_web_stress"]
    assert all(card["lane"] == "live_web_stress" for card in snapshot["workflow_cards_by_lane"]["live_web_stress"])
    assert "comparison_batches" in snapshot
    assert snapshot["comparison_batches"] == []
    assert snapshot["comparison_health"]["completed"] == 1


def test_session_snapshot_surfaces_trace_recovery_state(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )
    trace_path = Path(session.runtime_trace.episode_trace_path or "")
    assert trace_path.exists()
    trace_path.unlink()

    snapshot = build_session_snapshot(runtime, session.session_id)

    assert snapshot is not None
    assert snapshot["summary"]["workflow_title"] == snapshot["session"].title
    assert snapshot["summary"]["events"] == len(snapshot["events"])
    assert snapshot["summary"]["latest_artifact_title"]
    assert snapshot["summary"]["last_activity_at"]
    assert snapshot["trace_error"]
    assert "missing" in snapshot["trace_error"].lower()
    assert snapshot["can_retry"] is True
    if snapshot["artifact_previews"]:
        artifact = snapshot["artifact_previews"][0]
        assert "file_name" in artifact
        assert "preview_excerpt" in artifact
    if snapshot["revision_cards"]:
        revision = snapshot["revision_cards"][0]
        assert "before_excerpt" in revision
        assert "after_excerpt" in revision


def test_default_session_id_prefers_pending_approval_over_completed(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    completed = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )
    awaiting = runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )

    snapshot = build_console_snapshot(runtime)

    assert default_session_id(snapshot) == awaiting.session_id
    assert completed.session_id != awaiting.session_id
