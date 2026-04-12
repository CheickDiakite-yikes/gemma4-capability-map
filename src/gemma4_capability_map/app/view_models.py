from __future__ import annotations

import csv
import json
from difflib import unified_diff
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import (
    build_comparison_batch_rows,
    build_community_signal_rows,
    build_community_signal_summary,
    build_lane_summary_rows,
    build_public_summary,
    build_runtime_profile_rows,
)
from gemma4_capability_map.runtime.core import LocalAgentRuntime


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOARD_PATH = ROOT / "results" / "history" / "knowledge_work_board_latest.csv"
DEFAULT_COMMUNITY_SIGNAL_EXPORT = ROOT / "results" / "history" / "knowledge_work_community_signals.csv"
DEFAULT_COMMUNITY_SIGNAL_REGISTRY = ROOT / "configs" / "community_signals.yaml"

NUMERIC_BOARD_FIELDS = {
    "real_world_readiness_avg",
    "strict_interface_avg",
    "browser_workflow_avg",
    "artifact_quality_avg",
    "recovered_execution_avg",
    "escalation_correctness_avg",
    "total_params_b",
    "warmup_load_ms",
    "last_request_elapsed_ms",
    "requests_completed",
    "total_cost_per_mtok",
    "input_cost_per_mtok",
    "output_cost_per_mtok",
    "reasoner_params_b",
    "router_params_b",
    "retriever_params_b",
    "episode_count",
    "pass_count",
    "refine_count",
    "fail_count",
    "controller_repair_avg",
    "argument_repair_avg",
    "controller_fallback_avg",
    "intent_override_avg",
    "raw_planning_clean_rate_avg",
}


def load_board_rows(path: str | Path = DEFAULT_BOARD_PATH) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    normalized = [_normalize_board_row(row) for row in rows]
    return normalized


def load_community_signal_rows(board_path: str | Path = DEFAULT_BOARD_PATH) -> list[dict[str, Any]]:
    board_path = Path(board_path)
    export_path = board_path.parent / DEFAULT_COMMUNITY_SIGNAL_EXPORT.name
    if export_path.exists():
        with export_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return [_normalize_community_signal_row(row) for row in rows]
    return build_community_signal_rows(DEFAULT_COMMUNITY_SIGNAL_REGISTRY)


def build_console_snapshot(runtime: LocalAgentRuntime, board_path: str | Path = DEFAULT_BOARD_PATH) -> dict[str, Any]:
    sessions = runtime.list_sessions()
    approvals = runtime.list_approvals()
    profiles = runtime.list_system_profiles()
    board_rows = load_board_rows(board_path)
    community_signal_rows = load_community_signal_rows(board_path)
    local_rows = [row for row in board_rows if row.get("local")]
    local_rows = sorted(local_rows, key=lambda row: (-(row.get("real_world_readiness_avg") or 0.0), row.get("display_name", "")))
    runtime_profiles = build_runtime_profile_rows(local_rows)
    board_summary = build_public_summary(local_rows)
    community_signal_summary = build_community_signal_summary(community_signal_rows)
    comparison_batches = build_comparison_batch_rows(local_rows)
    lane_cards = _build_lane_cards(local_rows)
    comparison_health = _comparison_health_summary(local_rows)
    workflow_cards_by_lane = {
        lane: _build_workflow_cards(runtime.list_workflows(lane=lane), sessions, approvals, runtime_profiles, local_rows)
        for lane in ("replayable_core", "live_web_stress")
    }
    workflow_cards = workflow_cards_by_lane.get("replayable_core", [])

    return {
        "sessions": sessions,
        "approvals": approvals,
        "workflows": runtime.list_workflows(lane="replayable_core"),
        "profiles": profiles,
        "board_rows": board_rows,
        "community_signal_rows": community_signal_rows,
        "top_local_rows": local_rows[:8],
        "runtime_profiles": runtime_profiles,
        "comparison_batches": comparison_batches,
        "community_signal_summary": community_signal_summary,
        "community_signal_cards": _build_community_signal_cards(community_signal_rows),
        "comparison_health": comparison_health,
        "workflow_cards": workflow_cards,
        "workflow_cards_by_lane": workflow_cards_by_lane,
        "lane_cards": lane_cards,
        "board_summary": board_summary,
        "counts": {
            "sessions": len(sessions),
            "approvals": len(approvals),
            "running": sum(session.status.value in {"pending", "warming", "running", "resuming", "retrying"} for session in sessions),
            "completed": sum(session.status.value == "completed" for session in sessions),
            "failed": sum(session.status.value == "failed" for session in sessions),
        },
    }


def build_session_snapshot(runtime: LocalAgentRuntime, session_id: str) -> dict[str, Any] | None:
    try:
        session = runtime.get_session(session_id)
    except ValueError:
        return None
    events = runtime.get_events(session_id)
    trace_payload, trace_error = _load_trace_payload(session)
    artifact_previews = _build_artifact_previews(session, trace_payload)
    revision_cards = _build_revision_cards(trace_payload)
    review_cards = _build_review_cards(trace_payload)
    browser_cards = _build_browser_cards(trace_payload)
    pending_approval = next((approval for approval in session.approvals if approval.status.value == "pending"), None)
    latest_diff = revision_cards[0]["diff_excerpt"] if revision_cards else ""
    summary = {
        "workflow_id": session.workflow_id,
        "workflow_title": session.title,
        "workflow_category": session.workflow_category,
        "lane": session.lane,
        "status": session.status.value,
        "message": session.latest_message,
        "attempt": session.attempt,
        "session_id": session.session_id,
        "system_id": session.system_id,
        "lineage_root_session_id": session.lineage_root_session_id or session.parent_session_id or session.session_id,
        "active_approval_id": session.active_approval_id,
        "tool_invocations": len(session.tool_invocations),
        "events": len(events),
        "artifacts": len(artifact_previews),
        "revisions": len(revision_cards),
        "reviews": len(review_cards),
        "approvals": len(session.approvals),
        "has_pending_approval": pending_approval is not None,
        "latest_artifact_title": session.latest_artifact_title or (artifact_previews[0]["title"] if artifact_previews else ""),
        "latest_artifact_path": session.latest_artifact_path or (artifact_previews[0]["path"] if artifact_previews else ""),
        "latest_artifact_revision": artifact_previews[0]["revision"] if artifact_previews else 0,
        "latest_revision_artifact_id": session.latest_revision_artifact_id or (revision_cards[0]["artifact_id"] if revision_cards else ""),
        "latest_review_feedback": session.latest_review_feedback or (review_cards[0]["feedback"] if review_cards else ""),
        "last_activity_at": session.last_activity_at or session.updated_at,
    }
    trace_paths = []
    if session.runtime_trace is not None:
        for label, path in (
            ("manifest.json", session.runtime_trace.manifest_path),
            ("summary.json", session.runtime_trace.summary_path),
            ("episode_trace.json", session.runtime_trace.episode_trace_path),
        ):
            if path:
                trace_paths.append({"label": label, "path": path})
    return {
        "session": session,
        "events": events,
        "artifact_previews": artifact_previews,
        "revision_cards": revision_cards,
        "review_cards": review_cards,
        "browser_cards": browser_cards,
        "pending_approval": pending_approval,
        "trace_error": trace_error,
        "trace_paths": trace_paths,
        "metric_cards": _metric_cards(session),
        "summary": summary,
        "session_summary": summary,
        "stats": {
            "events": len(events),
            "artifacts": len(artifact_previews),
            "revisions": len(revision_cards),
            "tool_invocations": len(session.tool_invocations),
            "approvals": len(session.approvals),
        },
        "latest_diff": latest_diff,
        "can_retry": session.status.value in {"completed", "failed", "denied"},
        "can_resume": session.status.value in {"interrupted"},
    }


def default_profile_id(snapshot: dict[str, Any]) -> str | None:
    profiles = snapshot.get("profiles", [])
    for profile in profiles:
        if profile.recommended:
            return profile.system_id
    return profiles[0].system_id if profiles else None


def default_session_id(snapshot: dict[str, Any]) -> str | None:
    sessions = snapshot.get("sessions", [])
    if not sessions:
        return None
    prioritized = sorted(
        sessions,
        key=lambda session: (
            _session_priority(session),
            session.last_activity_at or session.updated_at or session.created_at,
        ),
        reverse=True,
    )
    return prioritized[0].session_id


def _comparison_health_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "completed": sum(1 for row in rows if row.get("board_status") == "completed"),
        "partial": sum(1 for row in rows if row.get("board_status") == "partial"),
        "timed_out": sum(1 for row in rows if row.get("board_status") == "timed_out"),
        "failed": sum(1 for row in rows if row.get("board_status") == "failed"),
        "avg_coverage": _average_metric(rows, "coverage_ratio"),
    }


def _session_priority(session: Any) -> tuple[int, int]:
    if getattr(session, "active_approval_id", None):
        return (5, getattr(session, "last_event_sequence", 0))
    if session.status.value in {"awaiting_approval"}:
        return (4, getattr(session, "last_event_sequence", 0))
    if session.status.value in {"running", "warming", "resuming", "retrying", "pending"}:
        return (3, getattr(session, "last_event_sequence", 0))
    if session.status.value == "interrupted":
        return (2, getattr(session, "last_event_sequence", 0))
    if session.status.value == "completed":
        return (1, getattr(session, "last_event_sequence", 0))
    return (0, getattr(session, "last_event_sequence", 0))


def _average_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _normalize_board_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for field in NUMERIC_BOARD_FIELDS:
        normalized[field] = _as_float(normalized.get(field))
    normalized["local"] = str(normalized.get("local", "")).lower() == "true"
    normalized["publishable_default"] = str(normalized.get("publishable_default", "")).lower() == "true"
    return normalized


def _normalize_community_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["status"] = str(normalized.get("status", "untriaged") or "untriaged").strip().lower()
    return normalized


def _build_lane_cards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lane_summaries = build_lane_summary_rows(rows)
    cards: list[dict[str, Any]] = []
    for lane in ("replayable_core", "live_web_stress"):
        candidates = [row for row in lane_summaries if row.get("lane") == lane]
        if not candidates:
            continue
        preferred = next((row for row in candidates if row.get("run_intent") == "exploratory"), candidates[0])
        cards.append(
            {
                "lane": lane,
                "label": "Replayable" if lane == "replayable_core" else "Live",
                "run_intent": preferred.get("run_intent", ""),
                "best_display_name": preferred.get("best_display_name", ""),
                "best_readiness": preferred.get("best_readiness", 0.0) or 0.0,
                "avg_readiness": preferred.get("avg_readiness", 0.0) or 0.0,
                "avg_strict_interface": preferred.get("avg_strict_interface", 0.0) or 0.0,
                "avg_browser_workflow": preferred.get("avg_browser_workflow", 0.0) or 0.0,
                "systems": preferred.get("systems", 0),
            }
        )
    return cards


def _build_workflow_cards(
    workflows: list[dict[str, Any]],
    sessions: list[Any],
    approvals: list[Any],
    runtime_profiles: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_lookup = {(row.get("system_id", ""), row.get("run_intent", "")): row for row in runtime_profiles}
    board_lookup = {
        (
            row.get("system_id", ""),
            row.get("lane", ""),
            row.get("run_intent", ""),
        ): row
        for row in board_rows
    }
    cards: list[dict[str, Any]] = []
    for workflow in workflows:
        sessions_for_workflow = [session for session in sessions if session.workflow_id == workflow["workflow_id"]]
        latest_session = sessions_for_workflow[0] if sessions_for_workflow else None
        pending_approvals = sum(1 for approval in approvals if approval.session_id in {session.session_id for session in sessions_for_workflow})
        recommended_system = workflow.get("recommended_system_id", "")
        recommended_profile = profile_lookup.get((recommended_system, "exploratory")) or profile_lookup.get((recommended_system, "canonical"))
        lane_row = board_lookup.get((recommended_system, workflow.get("lane", ""), "exploratory")) or board_lookup.get(
            (recommended_system, workflow.get("lane", ""), "canonical")
        )
        cards.append(
            {
                "workflow_id": workflow["workflow_id"],
                "title": workflow["title"],
                "subtitle": workflow["subtitle"],
                "description": workflow["description"],
                "role_family": workflow["role_family"],
                "category": workflow["category"],
                "lane": workflow["lane"],
                "episode_id": workflow["episode_id"],
                "supports_approval": workflow["supports_approval"],
                "preview_asset": workflow["preview_asset"],
                "recommended_system_id": recommended_system,
                "recommended_short_label": (recommended_profile or {}).get("short_label", recommended_system),
                "recommended_readiness": (lane_row or {}).get("real_world_readiness_avg", (recommended_profile or {}).get("avg_readiness", 0.0)) or 0.0,
                "active_sessions": len([session for session in sessions_for_workflow if session.status.value in {"pending", "warming", "running"}]),
                "pending_approvals": pending_approvals,
                "completed_sessions": len([session for session in sessions_for_workflow if session.status.value == "completed"]),
                "latest_session_id": latest_session.session_id if latest_session else "",
                "latest_status": latest_session.status.value if latest_session else "idle",
                "latest_message": latest_session.latest_message if latest_session else workflow["description"],
                "latest_artifact_count": len(latest_session.artifact_paths) if latest_session else 0,
                "tags": workflow.get("tags", []),
            }
        )
    return sorted(cards, key=lambda card: (-card["pending_approvals"], -card["active_sessions"], card["title"].lower()))


def _build_community_signal_cards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row in rows:
        cards.append(
            {
                "claim": row.get("claim", ""),
                "source": row.get("source", ""),
                "source_date": row.get("source_date", ""),
                "status": row.get("status", "untriaged"),
                "benchmark_slice": row.get("benchmark_slice", ""),
                "why_it_matters": row.get("why_it_matters", ""),
                "moonie_hypothesis": row.get("moonie_hypothesis", ""),
                "notes": row.get("notes", ""),
                "source_url": row.get("source_url", ""),
            }
        )
    return sorted(cards, key=lambda card: (card["status"], card["source_date"], card["source"], card["claim"]))


def _load_trace_payload(session: Any) -> tuple[dict[str, Any], str | None]:
    runtime_trace = session.runtime_trace
    if runtime_trace is None or not runtime_trace.episode_trace_path:
        return {}, None
    path = Path(runtime_trace.episode_trace_path)
    if not path.exists():
        return {}, f"Trace file missing: {path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, f"Trace file unreadable: {path.name}"
    return (payload if isinstance(payload, dict) else {}), None


def _build_artifact_previews(session: Any, trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    versions = trace_payload.get("artifact_versions", [])
    latest_versions: dict[str, dict[str, Any]] = {}
    for version in versions:
        if not isinstance(version, dict):
            continue
        artifact_id = str(version.get("artifact_id", "") or "")
        revision = int(version.get("revision", 0) or 0)
        current = latest_versions.get(artifact_id)
        if current is None or revision >= int(current.get("revision", 0) or 0):
            latest_versions[artifact_id] = version

    previews: list[dict[str, Any]] = []
    for artifact_path in session.artifact_paths:
        path = Path(artifact_path)
        version = next((item for item in latest_versions.values() if Path(str(item.get("file_path", ""))).name == path.name), None)
        preview = _preview_from_path(path, fallback=str((version or {}).get("content", "")))
        previews.append(
            {
                "title": str((version or {}).get("artifact_id", path.stem)),
                "file_name": path.name,
                "path": str(path),
                "kind": preview["kind"],
                "image_path": preview.get("image_path"),
                "excerpt": preview.get("excerpt", ""),
                "preview_excerpt": preview.get("excerpt", ""),
                "revision": int((version or {}).get("revision", 0) or 0),
                "score": _as_float((version or {}).get("score")) or 0.0,
                "source_stage": str((version or {}).get("source_stage", "")),
            }
        )
    if previews:
        return previews

    for version in latest_versions.values():
        previews.append(
            {
                "title": str(version.get("artifact_id", "artifact")),
                "file_name": Path(str(version.get("file_path", "") or "")).name,
                "path": str(version.get("file_path", "") or ""),
                "kind": "text",
                "image_path": None,
                "excerpt": _excerpt(str(version.get("content", "") or ""), limit=900),
                "preview_excerpt": _excerpt(str(version.get("content", "") or ""), limit=900),
                "revision": int(version.get("revision", 0) or 0),
                "score": _as_float(version.get("score")) or 0.0,
                "source_stage": str(version.get("source_stage", "")),
            }
        )
    return previews


def _build_revision_cards(trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for version in trace_payload.get("artifact_versions", []):
        if not isinstance(version, dict):
            continue
        grouped.setdefault(str(version.get("artifact_id", "artifact")), []).append(version)

    cards: list[dict[str, Any]] = []
    for artifact_id, versions in grouped.items():
        ordered = sorted(versions, key=lambda item: int(item.get("revision", 0) or 0))
        latest = ordered[-1]
        previous = ordered[-2] if len(ordered) > 1 else None
        previous_content = str(previous.get("content", "") if previous else "")
        latest_content = str(latest.get("content", ""))
        diff_excerpt = _diff_excerpt(previous_content, latest_content)
        cards.append(
            {
                "artifact_id": artifact_id,
                "revision": int(latest.get("revision", 0) or 0),
                "score": _as_float(latest.get("score")) or 0.0,
                "source_stage": str(latest.get("source_stage", "")),
                "excerpt": _excerpt(latest_content or "", limit=800),
                "before_excerpt": _excerpt(previous_content, limit=800) if previous_content else "",
                "after_excerpt": _excerpt(latest_content, limit=800),
                "diff_excerpt": diff_excerpt,
                "file_path": str(latest.get("file_path", "") or ""),
                "has_material_diff": bool(previous_content and previous_content != latest_content),
            }
        )
    return sorted(cards, key=lambda card: (-card["revision"], card["artifact_id"]))


def _build_review_cards(trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for review in trace_payload.get("review_history", []):
        if not isinstance(review, dict):
            continue
        cards.append(
            {
                "artifact_id": str(review.get("artifact_id", "")),
                "feedback": str(review.get("feedback", "")),
                "expected_improvements": review.get("expected_improvements", []),
            }
        )
    return cards


def _build_browser_cards(trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for action in trace_payload.get("browser_actions", [])[-8:]:
        if not isinstance(action, dict):
            continue
        outcome = str(action.get("transition_outcome", "") or action.get("gate_result", "") or "pass")
        cards.append(
            {
                "stage_id": str(action.get("stage_id", "")),
                "action": str(action.get("action", "")),
                "target": str(action.get("target", "")),
                "purpose": str(action.get("purpose", "")),
                "surface": str(action.get("surface", "")),
                "outcome": outcome,
                "evidence": str(action.get("evidence", "")),
                "verification_result": str(action.get("verification_result", "")),
            }
        )
    return cards


def _preview_from_path(path: Path, fallback: str = "") -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"} and path.exists():
        return {"kind": "image", "image_path": str(path.resolve()), "excerpt": ""}
    if suffix in {".md", ".txt", ".json", ".csv", ".log", ".yaml", ".yml"} and path.exists():
        try:
            return {"kind": "text", "image_path": None, "excerpt": _excerpt(path.read_text(encoding="utf-8"), limit=900)}
        except OSError:
            pass
    if fallback:
        return {"kind": "text", "image_path": None, "excerpt": _excerpt(fallback, limit=900)}
    return {"kind": "file", "image_path": None, "excerpt": ""}


def _metric_cards(session: Any) -> list[dict[str, Any]]:
    labels = [
        ("role_readiness_score", "Readiness"),
        ("artifact_quality_score", "Artifact"),
        ("browser_workflow_score", "Browser"),
        ("strict_interface_score", "Strict"),
        ("recovered_execution_score", "Recovered"),
        ("raw_planning_clean_rate", "Plan Clean"),
        ("controller_repair_count", "Repairs"),
    ]
    cards: list[dict[str, Any]] = []
    for key, label in labels:
        if key not in session.metrics:
            continue
        cards.append({"key": key, "label": label, "value": float(session.metrics.get(key, 0.0) or 0.0)})
    return cards


def _as_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _excerpt(text: str, *, limit: int = 600) -> str:
    compact = "\n".join(line.rstrip() for line in text.strip().splitlines())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _diff_excerpt(previous: str, current: str) -> str:
    if not previous and current:
        return _excerpt(current, limit=700)
    diff = list(unified_diff(previous.splitlines(), current.splitlines(), fromfile="previous", tofile="latest", lineterm=""))
    if not diff:
        return "No material diff captured."
    return _excerpt("\n".join(diff), limit=900)
