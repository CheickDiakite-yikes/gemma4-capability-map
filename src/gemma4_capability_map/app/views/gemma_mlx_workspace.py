from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st

from gemma4_capability_map.app.theme import pill_html
from gemma4_capability_map.app.view_models import (
    build_session_snapshot,
    build_workspace_snapshot,
    default_session_id,
)
from gemma4_capability_map.runtime.core import LocalAgentRuntime


DEFAULT_BROWSER_TARGET = "https://docs.streamlit.io"
SIGNIFICANT_EVENT_KINDS = {
    "created",
    "instruction_updated",
    "warming",
    "running",
    "approval_required",
    "approval_resolved",
    "resume_requested",
    "resume_started",
    "resumed",
    "artifacts_ready",
    "completed",
    "failed",
    "interrupted",
}


def render_gemma_mlx_workspace(runtime: LocalAgentRuntime) -> None:
    snapshot = build_workspace_snapshot(runtime)
    project_cards = snapshot.get("project_cards", [])
    selected_project_id = _resolve_project_id(project_cards)
    selected_session_id = _resolve_session_id(snapshot, selected_project_id)
    selected_snapshot = build_session_snapshot(runtime, selected_session_id) if selected_session_id else None

    selected_profile_id = st.session_state.get("workspace_selected_profile") or snapshot.get("workspace_default_profile_id")
    available_profiles = {profile.system_id: profile for profile in snapshot["profiles"]}
    if selected_profile_id not in available_profiles and available_profiles:
        selected_profile_id = next(iter(available_profiles))
    st.session_state["workspace_selected_profile"] = selected_profile_id
    lane = st.session_state.get("workspace_lane") or "live_web_stress"
    st.session_state["workspace_lane"] = lane

    active_profile = available_profiles.get(selected_profile_id)
    mlx_row = snapshot.get("mlx_profile_row", {})
    specialist_row = snapshot.get("hf_specialist_profile_row", {})
    qwen_row = snapshot.get("mlx_qwen_profile_row", {})

    st.markdown(
        f"""
        <div class="workspace-window">
          <div class="workspace-window-bar">
            <div class="workspace-window-actions">
              <span class="workspace-window-dot red"></span>
              <span class="workspace-window-dot amber"></span>
              <span class="workspace-window-dot green"></span>
            </div>
            <div class="workspace-window-title">Moonie Gemma MLX Workspace</div>
            <div class="workspace-window-meta">
              <span class="workspace-meta-chip">Projects {len(project_cards)}</span>
              <span class="workspace-meta-chip">Sessions {snapshot['counts']['sessions']}</span>
              <span class="workspace-meta-chip">Needs review {snapshot['counts']['approvals']}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([0.92, 1.58, 1.22], gap="medium")

    with left:
        _render_workspace_sidebar(snapshot, project_cards, selected_project_id)

    with center:
        _render_workspace_center(runtime, snapshot, selected_snapshot, selected_project_id, selected_profile_id, lane)

    with right:
        _render_workspace_context(runtime, snapshot, selected_snapshot, active_profile, mlx_row, specialist_row, qwen_row)


def _resolve_project_id(project_cards: list[dict[str, Any]]) -> str | None:
    selected_project_id = st.session_state.get("workspace_selected_project")
    if selected_project_id and any(card["project_id"] == selected_project_id for card in project_cards):
        return selected_project_id
    if not project_cards:
        return None
    st.session_state["workspace_selected_project"] = project_cards[0]["project_id"]
    return project_cards[0]["project_id"]


def _resolve_session_id(snapshot: dict[str, Any], selected_project_id: str | None) -> str | None:
    selected_session_id = st.session_state.get("workspace_selected_session")
    sessions = snapshot["sessions"]
    if selected_session_id and any(session.session_id == selected_session_id for session in sessions):
        if not selected_project_id:
            return selected_session_id
        matching = next((session for session in sessions if session.session_id == selected_session_id), None)
        if matching and matching.project_id == selected_project_id:
            return selected_session_id
    if selected_project_id:
        project_card = next((card for card in snapshot.get("project_cards", []) if card["project_id"] == selected_project_id), None)
        if project_card and project_card["sessions"]:
            selected_session_id = project_card["sessions"][0].session_id
            st.session_state["workspace_selected_session"] = selected_session_id
            return selected_session_id
    selected_session_id = default_session_id(snapshot)
    if selected_session_id:
        st.session_state["workspace_selected_session"] = selected_session_id
    return selected_session_id


def _render_workspace_sidebar(
    snapshot: dict[str, Any],
    project_cards: list[dict[str, Any]],
    selected_project_id: str | None,
) -> None:
    st.markdown(
        """
        <div class="workspace-panel workspace-sidebar-panel">
          <div class="workspace-sidebar-top">
            <div>
              <div class="workspace-kicker">Workspace</div>
              <h2 class="workspace-sidebar-title">Local agent coworker</h2>
              <p class="workspace-sidebar-subtitle">Gemma 4 on MLX, real runtime sessions, real approvals, real artifacts.</p>
            </div>
          </div>
          <div class="workspace-nav-list">
            <div class="workspace-nav-item active">New chat</div>
            <div class="workspace-nav-item">Search</div>
            <div class="workspace-nav-item">Plugins</div>
            <div class="workspace-nav-item">Pull requests</div>
            <div class="workspace-nav-item">Automations</div>
            <div class="workspace-nav-item">Scratchpad</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="workspace-section-label">Projects</div>', unsafe_allow_html=True)
    if not project_cards:
        st.markdown(
            """
            <div class="workspace-empty-card">
              <h4>No projects yet</h4>
              <div class="workspace-muted">Start a session to create a reusable project thread around a workflow and set of constraints.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for project in project_cards:
        project_selected = project["project_id"] == selected_project_id
        st.markdown(
            f"""
            <div class="workspace-project-card {'selected' if project_selected else ''}">
              <div class="workspace-row">
                <div>
                  <h4>{project['title']}</h4>
                  <div class="workspace-muted">{project['session_count']} threads · {project['running_count']} running · {project['approval_count']} review</div>
                  <div class="workspace-muted">{project['latest_message']}</div>
                </div>
                {pill_html(project['latest_activity_at'].split('T')[0], 'completed')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Open {project['title']}",
            key=f"workspace_project_{project['project_id']}",
            use_container_width=True,
        ):
            st.session_state["workspace_selected_project"] = project["project_id"]
            st.session_state["workspace_selected_session"] = project["latest_session_id"]
            st.rerun()
        for session in project["sessions"][:4]:
            st.markdown(
                f"""
                <div class="workspace-thread-meta">
                  <div class="workspace-thread-title">{session.title}</div>
                  <div class="workspace-thread-subtitle">{session.latest_message}</div>
                  <div class="workspace-thread-subtitle">{session.workflow_category.replace('_', ' ')} · {session.lane.replace('_', ' ')} · {session.last_activity_at or session.updated_at}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                f"View thread {session.session_id[-8:]}",
                key=f"workspace_session_{session.session_id}",
                use_container_width=True,
            ):
                st.session_state["workspace_selected_project"] = session.project_id or project["project_id"]
                st.session_state["workspace_selected_session"] = session.session_id
                st.rerun()
def _render_workspace_center(
    runtime: LocalAgentRuntime,
    snapshot: dict[str, Any],
    selected_snapshot: dict[str, Any] | None,
    selected_project_id: str | None,
    selected_profile_id: str | None,
    lane: str,
) -> None:
    session = selected_snapshot["session"] if selected_snapshot else None
    lane_cards = snapshot.get("workflow_cards_by_lane", {}).get(lane, [])
    workflow_options = {card["workflow_id"]: card for card in lane_cards}
    default_workflow_id = session.workflow_id if session and session.workflow_id in workflow_options else next(iter(workflow_options), None)
    available_profiles = {profile.system_id: profile for profile in snapshot["profiles"]}

    st.markdown(
        f"""
        <div class="workspace-main-header">
          <div>
            <div class="workspace-kicker">Gemma MLX Harness</div>
            <h1 class="workspace-main-title">{session.title if session else 'New chat'}</h1>
            <p class="workspace-main-subtitle">{session.latest_message if session else 'Launch a real local Gemma 4 MLX session, inspect runtime state, and keep the browser/review context attached to the thread.'}</p>
          </div>
          <div class="workspace-main-badges">
            <span class="workspace-meta-chip">{selected_profile_id or 'no profile'}</span>
            <span class="workspace-meta-chip">{lane.replace('_', ' ')}</span>
            <span class="workspace-meta-chip">{selected_project_id or 'new project'}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if session is None:
        st.markdown(
            """
            <div class="workspace-hero">
              <div class="workspace-hero-mark">✦</div>
              <h2>Let’s build with Gemma 4 MLX</h2>
              <p>Moonie now has a dedicated local agent workspace: projects on the left, conversation in the center, browser and review context on the right.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="workspace-section-label">Conversation</div>', unsafe_allow_html=True)
        for message in _build_transcript_rows(selected_snapshot):
            st.markdown(
                f"""
                <div class="workspace-message {message['role']}">
                  <div class="workspace-message-title">{message['title']}</div>
                  <div class="workspace-message-body">{message['body']}</div>
                  <div class="workspace-message-meta">{message['meta']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="workspace-section-label">Composer</div>', unsafe_allow_html=True)
    if not available_profiles or not workflow_options:
        st.markdown(
            """
            <div class="workspace-empty-card">
              <h4>Workspace configuration incomplete</h4>
              <div class="workspace-muted">The runtime needs at least one profile and one workflow for this lane before the Gemma MLX workspace can launch a session.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    with st.form("workspace_composer"):
        project_value = st.text_input(
            "Project",
            value=selected_project_id or (session.project_id if session else default_workflow_id or "gemma-mlx"),
            placeholder="project-alpha",
        )
        profile_options = list(available_profiles.keys())
        profile_index = max(0, profile_options.index(selected_profile_id)) if selected_profile_id in profile_options else 0
        chosen_profile = st.selectbox(
            "Runtime",
            options=profile_options,
            index=profile_index,
            format_func=lambda system_id: available_profiles[system_id].short_label,
        )
        chosen_lane = st.radio("Lane", ["live_web_stress", "replayable_core"], horizontal=True, index=0 if lane == "live_web_stress" else 1)
        workflow_id = st.selectbox(
            "Workflow",
            options=list(workflow_options.keys()),
            index=max(0, list(workflow_options.keys()).index(default_workflow_id)) if default_workflow_id in workflow_options else 0,
            format_func=lambda item: workflow_options[item]["title"],
        )
        request = st.text_area(
            "Instruction",
            value="",
            height=120,
            placeholder="Ask the agent to research, browse, review artifacts, or continue the selected project with the newest constraint.",
        )
        start = st.form_submit_button("Start session", use_container_width=True)
        stream = st.form_submit_button("Tail selected session", use_container_width=True)
        resume = st.form_submit_button("Resume selected", use_container_width=True)
        retry = st.form_submit_button("Retry selected", use_container_width=True)

    st.session_state["workspace_selected_profile"] = chosen_profile
    st.session_state["workspace_lane"] = chosen_lane

    if start:
        launched = runtime.launch_session(
            workflow_id=workflow_id,
            system_id=chosen_profile,
            lane=chosen_lane,
            project_id=_normalize_project_id(project_value),
            human_request=request,
            background=True,
        )
        st.session_state["workspace_selected_project"] = launched.project_id
        st.session_state["workspace_selected_session"] = launched.session_id
        st.session_state[f"workspace_stream_cursor_{launched.session_id}"] = 0
        st.rerun()

    if stream and session is not None:
        cursor_key = f"workspace_stream_cursor_{session.session_id}"
        stream_payload = runtime.stream_session(
            session.session_id,
            after_sequence=int(st.session_state.get(cursor_key, 0)),
            timeout_s=1.5,
            poll_s=0.15,
        )
        st.session_state[cursor_key] = stream_payload["session"].last_event_sequence
        st.session_state["workspace_stream_notice"] = f"Fetched {len(stream_payload['events'])} new events from `{session.title}`."
        st.rerun()

    if resume and session is not None and selected_snapshot and selected_snapshot["can_resume"]:
        resumed = runtime.resume_session(session.session_id, note=request.strip(), background=True)
        st.session_state["workspace_selected_session"] = resumed.session_id
        st.rerun()

    if retry and session is not None and selected_snapshot and selected_snapshot["can_retry"]:
        retried = runtime.retry_session(session.session_id, note=request.strip(), background=True)
        st.session_state["workspace_selected_project"] = retried.project_id
        st.session_state["workspace_selected_session"] = retried.session_id
        st.rerun()

    stream_notice = st.session_state.get("workspace_stream_notice")
    if stream_notice:
        st.caption(stream_notice)


def _render_workspace_context(
    runtime: LocalAgentRuntime,
    snapshot: dict[str, Any],
    selected_snapshot: dict[str, Any] | None,
    active_profile: Any,
    mlx_row: dict[str, Any],
    specialist_row: dict[str, Any],
    qwen_row: dict[str, Any],
) -> None:
    session = selected_snapshot["session"] if selected_snapshot else None
    pending_approval = selected_snapshot["pending_approval"] if selected_snapshot else None
    browser_target = _browser_target(selected_snapshot)

    st.markdown('<div class="workspace-section-label">Runtime posture</div>', unsafe_allow_html=True)
    runtime_metrics = [
        ("Gemma MLX readiness", _percent(mlx_row.get("real_world_readiness_avg"))),
        ("Gemma MLX strict", _percent(mlx_row.get("strict_interface_avg"))),
        ("Gemma MLX recovered", _percent(mlx_row.get("recovered_execution_avg"))),
        ("Gemma MLX latency", _latency_text(mlx_row.get("last_request_elapsed_ms"))),
        ("HF specialists", _percent(specialist_row.get("real_world_readiness_avg"))),
        ("MLX Qwen", _percent(qwen_row.get("real_world_readiness_avg"))),
    ]
    runtime_columns = st.columns(2, gap="small")
    for index, (label, value) in enumerate(runtime_metrics):
        with runtime_columns[index % 2]:
            st.markdown(_runtime_metric_card(label, value), unsafe_allow_html=True)
    if active_profile is not None:
        st.caption(
            f"{active_profile.short_label} · {active_profile.backend} · {active_profile.deployment or 'local runtime'} · timeout {active_profile.request_timeout_seconds:.0f}s"
        )

    summary_tab, review_tab, browser_tab = st.tabs(["Summary", "Review", "Browser"])

    with summary_tab:
        if session is None:
            st.markdown(
                """
                <div class="workspace-empty-card">
                  <h4>No active session</h4>
                  <div class="workspace-muted">Launch a Gemma MLX thread to inspect metrics, approvals, artifacts, and live browser state here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="workspace-summary-card">
                  <div class="workspace-row">
                    <div>
                      <h4>{session.title}</h4>
                      <div class="workspace-muted">{session.workflow_category.replace('_', ' ')} · {session.lane.replace('_', ' ')}</div>
                      <div class="workspace-muted">Project {session.project_id or 'n/a'} · attempt {session.attempt}</div>
                    </div>
                    {pill_html(session.status.value.replace('_', ' '), session.status.value)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric_cards = selected_snapshot.get("metric_cards", [])
            if metric_cards:
                metric_columns = st.columns(2, gap="small")
                for index, card in enumerate(metric_cards):
                    value = f"{card['value'] * 100:.1f}%" if "count" not in card["key"] else f"{card['value']:.0f}"
                    with metric_columns[index % 2]:
                        st.markdown(_runtime_metric_card(card["label"], value), unsafe_allow_html=True)
            if pending_approval is not None:
                st.markdown(
                    f"""
                    <div class="workspace-approval-card">
                      <div class="workspace-row">
                        <div>
                          <h4>{pending_approval.title}</h4>
                          <div class="workspace-muted">{pending_approval.reason}</div>
                        </div>
                        {pill_html('needs review', 'awaiting_approval')}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                note = st.text_input("Approval note", key=f"workspace_approval_note_{pending_approval.approval_id}")
                approve_col, deny_col = st.columns(2, gap="small")
                with approve_col:
                    if st.button("Approve and finalize", key=f"workspace_approve_{pending_approval.approval_id}", use_container_width=True):
                        runtime.resolve_approval_by_id(pending_approval.approval_id, decision="approve", note=note, resume=True)
                        st.rerun()
                with deny_col:
                    if st.button("Deny", key=f"workspace_deny_{pending_approval.approval_id}", use_container_width=True):
                        runtime.resolve_approval_by_id(pending_approval.approval_id, decision="deny", note=note, resume=False)
                        st.rerun()

            st.markdown('<div class="workspace-section-label">Tool activity</div>', unsafe_allow_html=True)
            if session.tool_invocations:
                for tool in session.tool_invocations[-6:]:
                    args = ", ".join(f"{key}={value}" for key, value in list(tool.arguments.items())[:3]) or "no args"
                    st.markdown(
                        f"""
                        <div class="workspace-inline-card">
                          <div class="workspace-row">
                            <div>
                              <h4>{tool.tool_name}</h4>
                              <div class="workspace-muted">{args}</div>
                            </div>
                            {pill_html(tool.validator_result, tool.validator_result)}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
                    <div class="workspace-inline-card">
                      <div class="workspace-muted">No tool calls recorded yet.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with review_tab:
        if selected_snapshot is None:
            st.markdown(
                """
                <div class="workspace-empty-card">
                  <div class="workspace-muted">Artifact previews, diffs, and review rounds will appear here after the first run.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            artifact_previews = selected_snapshot["artifact_previews"] or []
            revision_cards = selected_snapshot["revision_cards"] or []
            review_cards = selected_snapshot["review_cards"] or []
            for artifact in artifact_previews[:3]:
                st.markdown(
                    f"""
                    <div class="workspace-inline-card">
                      <div class="workspace-row">
                        <div>
                          <h4>{artifact['title']}</h4>
                          <div class="workspace-muted">{artifact['file_name']}</div>
                          <div class="workspace-muted">Revision {artifact['revision']} · score {(artifact['score'] or 0.0) * 100:.1f}%</div>
                        </div>
                        {pill_html(artifact['kind'], 'completed')}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if artifact["kind"] == "image" and artifact.get("image_path"):
                    st.image(artifact["image_path"], use_container_width=True)
                elif artifact.get("preview_excerpt"):
                    st.code(artifact["preview_excerpt"], language="markdown")
            for revision in revision_cards[:2]:
                st.markdown(
                    f"""
                    <div class="workspace-inline-card">
                      <h4>{revision['artifact_id']}</h4>
                      <div class="workspace-muted">Revision {revision['revision']} · stage {revision['source_stage'] or 'n/a'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.code(revision["diff_excerpt"], language="diff")
            for review in review_cards[:2]:
                checklist = ", ".join(review["expected_improvements"]) if review["expected_improvements"] else "No explicit checklist"
                st.markdown(
                    f"""
                    <div class="workspace-inline-card">
                      <h4>{review['artifact_id']}</h4>
                      <div class="workspace-muted">{review['feedback']}</div>
                      <div class="workspace-muted">Expected: {checklist}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with browser_tab:
        target = st.text_input("Browser target", value=browser_target, key="workspace_browser_target")
        st.markdown(
            f"""
            <div class="workspace-browser-shell">
              <div class="workspace-browser-bar">
                <span class="workspace-browser-button">←</span>
                <span class="workspace-browser-button">→</span>
                <span class="workspace-browser-button">↻</span>
                <div class="workspace-browser-url">{target}</div>
              </div>
              <div class="workspace-browser-stage">
                <div class="workspace-browser-title">Browser context</div>
                <p>This pane mirrors the runtime’s browser state and review context. The first shippable slice keeps the browser shell inside Moonie and the real browser automation in the runtime trace.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if _is_http_url(target):
            st.link_button("Open target externally", target, use_container_width=True)
        browser_cards = selected_snapshot["browser_cards"] if selected_snapshot else []
        if browser_cards:
            for browser in browser_cards[:6]:
                st.markdown(
                    f"""
                    <div class="workspace-inline-card">
                      <div class="workspace-row">
                        <div>
                          <h4>{browser['action']} → {browser['target']}</h4>
                          <div class="workspace-muted">{browser['purpose'] or browser['surface']}</div>
                          <div class="workspace-muted">{browser['evidence'] or browser['verification_result']}</div>
                        </div>
                        {pill_html(browser['outcome'].replace('_', ' '), browser['outcome'])}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="workspace-inline-card">
                  <div class="workspace-muted">No browser actions recorded yet for this session.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _build_transcript_rows(selected_snapshot: dict[str, Any]) -> list[dict[str, str]]:
    session = selected_snapshot["session"]
    rows: list[dict[str, str]] = []
    for instruction in session.instruction_history[-6:]:
        rows.append(
            {
                "role": "user",
                "title": "You",
                "body": _escape_html(instruction.content),
                "meta": f"{instruction.source} · {instruction.created_at}",
                "sort_key": instruction.created_at,
            }
        )
    for event in selected_snapshot["events"]:
        if event.kind not in SIGNIFICANT_EVENT_KINDS:
            continue
        role = "agent"
        if event.kind in {"approval_required", "failed", "interrupted"}:
            role = "system"
        rows.append(
            {
                "role": role,
                "title": event.kind.replace("_", " ").title(),
                "body": _escape_html(event.message),
                "meta": f"#{event.sequence} · {event.created_at}",
                "sort_key": f"{event.created_at}:{event.sequence:04d}",
            }
        )
    ordered = sorted(rows, key=lambda row: row["sort_key"])
    return [{key: value for key, value in row.items() if key != "sort_key"} for row in ordered[-12:]]


def _browser_target(selected_snapshot: dict[str, Any] | None) -> str:
    if selected_snapshot is not None:
        browser_cards = selected_snapshot.get("browser_cards") or []
        for browser in browser_cards:
            candidate = browser.get("target") or browser.get("evidence") or ""
            if _is_http_url(candidate):
                return candidate
        session = selected_snapshot["session"]
        if session.preview_asset and Path(session.preview_asset).exists():
            return session.preview_asset
    return DEFAULT_BROWSER_TARGET


def _normalize_project_id(project_value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", project_value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "gemma-mlx"


def _runtime_metric_card(label: str, value: str) -> str:
    return f"""
    <div class="workspace-metric-card">
      <div class="workspace-metric-label">{label}</div>
      <div class="workspace-metric-value">{value}</div>
    </div>
    """


def _percent(value: Any) -> str:
    try:
        return f"{float(value or 0.0) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _latency_text(value: Any) -> str:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return "n/a"
    if numeric <= 0.0:
        return "n/a"
    return f"{numeric:.0f} ms"


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")
