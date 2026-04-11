from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gemma4_capability_map.runtime.core import LocalAgentRuntime


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOARD_PATH = ROOT / "results" / "history" / "knowledge_work_board_latest.csv"


def load_board_rows(path: str | Path = DEFAULT_BOARD_PATH) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full_lane = [row for row in rows if row.get("run_scope") == "full_lane"]
    for row in full_lane:
        for field in (
            "real_world_readiness_avg",
            "strict_interface_avg",
            "browser_workflow_avg",
            "artifact_quality_avg",
            "recovered_execution_avg",
            "total_params_b",
            "warmup_load_ms",
            "last_request_elapsed_ms",
        ):
            value = row.get(field)
            row[field] = float(value) if value not in {"", None} else None
        row["local"] = str(row.get("local", "")).lower() == "true"
    return full_lane


def build_console_snapshot(runtime: LocalAgentRuntime, board_path: str | Path = DEFAULT_BOARD_PATH) -> dict[str, Any]:
    sessions = runtime.list_sessions()
    approvals = runtime.list_approvals()
    workflows = runtime.list_workflows()
    profiles = runtime.list_system_profiles()
    board_rows = load_board_rows(board_path)
    local_rows = [row for row in board_rows if row.get("local")]
    local_rows = sorted(local_rows, key=lambda row: (-(row.get("real_world_readiness_avg") or 0.0), row.get("display_name", "")))

    lane_cards: list[dict[str, Any]] = []
    for lane in ("replayable_core", "live_web_stress"):
        lane_subset = [row for row in local_rows if row.get("lane") == lane]
        if not lane_subset:
            continue
        leader = lane_subset[0]
        lane_cards.append(
            {
                "lane": lane,
                "label": "Replayable" if lane == "replayable_core" else "Live",
                "display_name": leader.get("display_name", ""),
                "short_label": leader.get("short_label", leader.get("display_name", "")),
                "readiness": leader.get("real_world_readiness_avg", 0.0) or 0.0,
                "strict": leader.get("strict_interface_avg", 0.0) or 0.0,
                "browser": leader.get("browser_workflow_avg", 0.0) or 0.0,
            }
        )

    return {
        "sessions": sessions,
        "approvals": approvals,
        "workflows": workflows,
        "profiles": profiles,
        "board_rows": board_rows,
        "top_local_rows": local_rows[:6],
        "lane_cards": lane_cards,
        "counts": {
            "sessions": len(sessions),
            "approvals": len(approvals),
            "running": sum(session.status.value in {"pending", "warming", "running"} for session in sessions),
            "completed": sum(session.status.value == "completed" for session in sessions),
        },
    }


def default_profile_id(snapshot: dict[str, Any]) -> str | None:
    profiles = snapshot.get("profiles", [])
    for profile in profiles:
        if profile.recommended:
            return profile.system_id
    return profiles[0].system_id if profiles else None


def default_session_id(snapshot: dict[str, Any]) -> str | None:
    sessions = snapshot.get("sessions", [])
    return sessions[0].session_id if sessions else None
