from __future__ import annotations

from pathlib import Path

import streamlit as st

from gemma4_capability_map.app.theme import pill_html
from gemma4_capability_map.app.view_models import (
    build_console_snapshot,
    build_session_snapshot,
    default_profile_id,
    default_session_id,
)
from gemma4_capability_map.runtime.core import LocalAgentRuntime


def render_operator_console(runtime: LocalAgentRuntime) -> None:
    snapshot = build_console_snapshot(runtime)
    selected_session_id = st.session_state.get("operator_selected_session") or default_session_id(snapshot)
    selected_profile_id = st.session_state.get("operator_selected_profile") or default_profile_id(snapshot)
    summary = snapshot["board_summary"]
    comparison_health = snapshot["comparison_health"]
    community_signal_summary = snapshot.get("community_signal_summary", {})
    best_local = summary.get("highest_readiness_local_profile", {})
    fastest_local = summary.get("fastest_local_profile", {})

    st.markdown(
        f"""
        <div class="console-hero">
          <p class="console-title">Moonie Operator Console</p>
          <p class="console-subtitle">Dark operator shell for benchmark-backed local workflows, approvals, artifacts, and publishable runtime posture.</p>
          <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Sessions</div><div class="metric-value">{snapshot['counts']['sessions']}</div></div>
            <div class="metric-card"><div class="metric-label">Running</div><div class="metric-value">{snapshot['counts']['running']}</div></div>
            <div class="metric-card"><div class="metric-label">Needs Review</div><div class="metric-value">{snapshot['counts']['approvals']}</div></div>
            <div class="metric-card"><div class="metric-label">Failures</div><div class="metric-value">{snapshot['counts']['failed']}</div></div>
          </div>
          <div class="console-inline-grid">
            <div class="console-chip">
              <span class="console-chip-label">Top local</span>
              <span class="console-chip-value">{best_local.get('display_name', 'n/a')}</span>
            </div>
            <div class="console-chip">
              <span class="console-chip-label">Readiness</span>
              <span class="console-chip-value">{(best_local.get('avg_readiness', 0.0) or 0.0) * 100:.1f}%</span>
            </div>
            <div class="console-chip">
              <span class="console-chip-label">Fastest local</span>
              <span class="console-chip-value">{fastest_local.get('display_name', 'n/a')}</span>
            </div>
            <div class="console-chip">
              <span class="console-chip-label">Last request</span>
              <span class="console-chip-value">{(fastest_local.get('value', 0.0) or 0.0):.0f} ms</span>
            </div>
            <div class="console-chip">
              <span class="console-chip-label">Completed batches</span>
              <span class="console-chip-value">{comparison_health.get('completed', 0)}</span>
            </div>
            <div class="console-chip">
              <span class="console-chip-label">Coverage</span>
              <span class="console-chip-value">{(comparison_health.get('avg_coverage', 0.0) or 0.0) * 100:.1f}%</span>
            </div>
            <div class="console-chip">
              <span class="console-chip-label">Harnessability</span>
              <span class="console-chip-value">{summary.get('harnessability_slice_count', 0)}</span>
            </div>
            <div class="console-chip">
              <span class="console-chip-label">Direction</span>
              <span class="console-chip-value">{summary.get('direction_following_slice_count', 0)}</span>
            </div>
            <div class="console-chip">
              <span class="console-chip-label">Signals</span>
              <span class="console-chip-value">{community_signal_summary.get('row_count', 0)}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lane_columns = st.columns(max(1, len(snapshot["lane_cards"])), gap="small")
    for column, card in zip(lane_columns, snapshot["lane_cards"], strict=False):
        with column:
            st.markdown(
                f"""
                <div class="console-card lane-card">
                  <div class="console-row">
                    <div>
                      <div class="console-section-title">{card['label']} lane</div>
                      <h4>{card['best_display_name']}</h4>
                      <div class="console-muted">Best {(card['best_readiness'] or 0.0) * 100:.1f}% · Fleet avg {(card['avg_readiness'] or 0.0) * 100:.1f}%</div>
                    </div>
                    {pill_html(card['run_intent'].replace('_', ' '), 'completed')}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    toolbar_left, toolbar_right = st.columns([1, 1], gap="small")
    with toolbar_left:
        if st.button("Refresh Console", use_container_width=True):
            st.rerun()
    with toolbar_right:
        st.caption("Operator shell keeps benchmark and runtime state in one surface.")

    left, center, right = st.columns([1.1, 1.85, 1.25], gap="large")

    with left:
        st.markdown('<div class="console-shell">', unsafe_allow_html=True)
        _render_launch_panel(runtime, snapshot, selected_profile_id)
        _render_workflow_catalog(snapshot)
        _render_community_signals(snapshot)
        _render_session_nav(snapshot)
        st.markdown("</div>", unsafe_allow_html=True)

    with center:
        session_snapshot = build_session_snapshot(runtime, selected_session_id) if selected_session_id else None
        if session_snapshot is None:
            st.markdown(
                """
                <div class="console-shell">
                  <div class="console-section-title">Session Detail</div>
                  <div class="console-card">
                    <h4>No session selected</h4>
                    <div class="console-muted">Launch a workflow or choose a recent session to inspect traces, approvals, artifacts, and revisions.</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            _render_session_detail(runtime, session_snapshot)

    with right:
        st.markdown('<div class="console-shell">', unsafe_allow_html=True)
        _render_benchmark_pulse(snapshot)
        _render_approval_queue(runtime, snapshot)
        _render_runtime_profiles(snapshot)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_launch_panel(runtime: LocalAgentRuntime, snapshot: dict[str, object], selected_profile_id: str | None) -> None:
    st.markdown('<div class="console-section-title">Launch Workflow</div>', unsafe_allow_html=True)
    profiles = {profile.system_id: profile for profile in snapshot["profiles"]}  # type: ignore[index]
    profile_options = list(profiles.keys())
    if not profile_options:
        st.markdown('<div class="console-card"><div class="console-muted">No profiles or workflows are configured.</div></div>', unsafe_allow_html=True)
        return

    profile_id = st.selectbox(
        "System profile",
        options=profile_options,
        index=max(0, profile_options.index(selected_profile_id)) if selected_profile_id in profiles else 0,
        format_func=lambda system_id: profiles[system_id].short_label,
    )
    st.session_state["operator_selected_profile"] = profile_id
    lane = st.radio("Lane", ["replayable_core", "live_web_stress"], horizontal=True)
    workflows = {
        workflow["workflow_id"]: workflow
        for workflow in snapshot.get("workflow_cards_by_lane", {}).get(lane, [])  # type: ignore[union-attr]
    }
    workflow_options = list(workflows.keys())
    if not workflow_options:
        st.markdown('<div class="console-card"><div class="console-muted">No workflows are available for the selected lane.</div></div>', unsafe_allow_html=True)
        return
    workflow_id = st.selectbox(
        "Workflow",
        options=workflow_options,
        format_func=lambda item: workflows[item]["title"],
    )
    request = st.text_area("Task framing", placeholder="Operator notes, newest constraint, or approval guidance.")
    workflow = workflows[workflow_id]
    st.markdown(
        f"""
        <div class="console-card">
          <h4>{workflow['title']}</h4>
          <div class="console-muted">{workflow['description']}</div>
          <div class="console-muted">Recommended {workflow['recommended_short_label']} · readiness {(workflow['recommended_readiness'] or 0.0) * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Session", use_container_width=True):
        session = runtime.launch_session(
            workflow_id=workflow_id,
            system_id=profile_id,
            lane=lane,
            human_request=request,
            background=True,
        )
        st.session_state["operator_selected_session"] = session.session_id
        st.rerun()


def _render_workflow_catalog(snapshot: dict[str, object]) -> None:
    st.markdown('<div class="console-section-title">Workflow Lens</div>', unsafe_allow_html=True)
    workflow_cards = snapshot.get("workflow_cards_by_lane", {}).get("replayable_core", snapshot["workflow_cards"])  # type: ignore[index]
    for workflow in workflow_cards[:5]:  # type: ignore[index]
        tags = ", ".join(workflow["tags"][:3]) if workflow["tags"] else workflow["category"].replace("_", " ")
        st.markdown(
            f"""
            <div class="console-card">
              <div class="console-row">
                <div>
                  <h4>{workflow['title']}</h4>
                  <div class="console-muted">{workflow['subtitle']}</div>
                  <div class="console-muted">{tags}</div>
                </div>
                {pill_html('review' if workflow['supports_approval'] else 'fast path', 'awaiting_approval' if workflow['supports_approval'] else 'completed')}
              </div>
              <div class="console-inline-grid compact">
                <div class="console-chip"><span class="console-chip-label">Active</span><span class="console-chip-value">{workflow['active_sessions']}</span></div>
                <div class="console-chip"><span class="console-chip-label">Review</span><span class="console-chip-value">{workflow['pending_approvals']}</span></div>
                <div class="console-chip"><span class="console-chip-label">Artifacts</span><span class="console-chip-value">{workflow['latest_artifact_count']}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_community_signals(snapshot: dict[str, object]) -> None:
    st.markdown('<div class="console-section-title">Community Signals</div>', unsafe_allow_html=True)
    community_signal_cards = snapshot.get("community_signal_cards", [])
    if not community_signal_cards:
        st.markdown(
            """
            <div class="console-card">
              <div class="console-muted">No community signals are loaded yet.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    for signal in community_signal_cards[:4]:
        st.markdown(
            f"""
            <div class="console-card">
              <div class="console-row">
                <div>
                  <h4>{signal['claim']}</h4>
                  <div class="console-muted">{signal['why_it_matters']}</div>
                  <div class="console-muted">{signal['benchmark_slice'] or 'unassigned'} · {signal['source']} · {signal['source_date']}</div>
                </div>
                {pill_html(signal['status'].replace('_', ' '), signal['status'])}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_session_nav(snapshot: dict[str, object]) -> None:
    st.markdown('<div class="console-section-title">Recent Sessions</div>', unsafe_allow_html=True)
    for session in snapshot["sessions"][:8]:  # type: ignore[index]
        st.markdown(
            f"""
            <div class="console-card session-nav-card">
              <div class="console-row">
                <div>
                  <h4>{session.title}</h4>
                  <div class="console-muted">{session.latest_message}</div>
                  <div class="console-muted">{session.workflow_category.replace('_', ' ')} · {session.lane.replace('_', ' ')} · attempt {session.attempt}</div>
                  <div class="console-muted">{session.latest_artifact_title or 'No artifact yet'} · activity {session.last_activity_at or session.updated_at}</div>
                </div>
                {pill_html(session.status.value.replace('_', ' '), session.status.value)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Session", key=f"operator_session_{session.session_id}", use_container_width=True):
            st.session_state["operator_selected_session"] = session.session_id
            st.rerun()


def _render_session_detail(runtime: LocalAgentRuntime, session_snapshot: dict[str, object]) -> None:
    session = session_snapshot["session"]
    pending_approval = session_snapshot["pending_approval"]
    summary = session_snapshot.get("summary", {})
    st.markdown('<div class="console-shell">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="console-card session-summary-card">
          <div class="console-row">
            <div>
              <div class="console-section-title">{session.workflow_category.replace('_', ' ')}</div>
              <h4>{session.title}</h4>
              <div class="console-muted">{session.latest_message}</div>
              <div class="console-muted">Session {session.session_id} · lane {session.lane.replace('_', ' ')}</div>
            </div>
            {pill_html(session.status.value.replace('_', ' '), session.status.value)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="console-card session-summary-card">
          <div class="console-row">
            <div>
              <div class="console-section-title">Session Snapshot</div>
              <h4>{summary.get('workflow_title', session.title)}</h4>
              <div class="console-muted">{summary.get('message', session.latest_message)}</div>
              <div class="console-muted">{summary.get('workflow_category', session.workflow_category).replace('_', ' ')} · {summary.get('lane', session.lane).replace('_', ' ')} · attempt {summary.get('attempt', session.attempt)}</div>
            </div>
            {pill_html(str(summary.get('status', session.status.value)).replace('_', ' '), str(summary.get('status', session.status.value)))}
          </div>
          <div class="console-inline-grid compact">
            <div class="console-chip"><span class="console-chip-label">Artifacts</span><span class="console-chip-value">{summary.get('artifacts', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Revisions</span><span class="console-chip-value">{summary.get('revisions', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Approvals</span><span class="console-chip-value">{summary.get('approvals', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Latest artifact</span><span class="console-chip-value">{summary.get('latest_artifact_title', 'n/a') or 'n/a'}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(max(1, len(session_snapshot["metric_cards"])), gap="small")
    for column, metric in zip(metric_columns, session_snapshot["metric_cards"], strict=False):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{metric['label']}</div>
                  <div class="metric-value">{metric['value'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    stat_cols = st.columns(4, gap="small")
    for column, (label, value) in zip(
        stat_cols,
        (
            ("Events", session_snapshot["stats"]["events"]),
            ("Artifacts", session_snapshot["stats"]["artifacts"]),
            ("Revisions", session_snapshot["stats"]["revisions"]),
            ("Tool calls", session_snapshot["stats"]["tool_invocations"]),
        ),
        strict=False,
    ):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if pending_approval is not None:
        st.markdown(
            f"""
            <div class="console-card approval-highlight">
              <div class="console-row">
                <div>
                  <div class="console-section-title">Awaiting Approval</div>
                  <h4>{pending_approval.title}</h4>
                  <div class="console-muted">{pending_approval.reason}</div>
                </div>
                {pill_html('needs review', 'awaiting_approval')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        note = st.text_input("Approval note", key=f"session_note_{pending_approval.approval_id}")
        approve_col, deny_col = st.columns(2, gap="small")
        with approve_col:
            if st.button("Approve and Finalize", key=f"session_approve_{pending_approval.approval_id}", use_container_width=True):
                runtime.resolve_approval(session.session_id, decision="approve", note=note)
                st.rerun()
        with deny_col:
            if st.button("Deny", key=f"session_deny_{pending_approval.approval_id}", use_container_width=True):
                runtime.resolve_approval(session.session_id, decision="deny", note=note)
                st.rerun()

    action_cols = st.columns(2, gap="small")
    with action_cols[0]:
        if session_snapshot.get("can_resume") and st.button("Resume Session", key=f"session_resume_{session.session_id}", use_container_width=True):
            runtime.resume_session(session.session_id)
            st.rerun()
    with action_cols[1]:
        if session_snapshot.get("can_retry") and st.button("Retry Session", key=f"session_retry_{session.session_id}", use_container_width=True):
            retried = runtime.retry_session(session.session_id, background=True)
            st.session_state["operator_selected_session"] = retried.session_id
            st.rerun()

    tabs = st.tabs(["Overview", "Artifacts", "Diffs", "Timeline"])
    with tabs[0]:
        _render_session_overview(session_snapshot)
    with tabs[1]:
        _render_session_artifacts(session_snapshot)
    with tabs[2]:
        _render_session_diffs(session_snapshot)
    with tabs[3]:
        _render_session_timeline(session_snapshot)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_session_overview(session_snapshot: dict[str, object]) -> None:
    session = session_snapshot["session"]
    summary = session_snapshot.get("summary", {})
    if session_snapshot.get("trace_error"):
        st.markdown(
            f"""
            <div class="console-card approval-highlight">
              <div class="console-section-title">Trace Recovery</div>
              <div class="console-muted">{session_snapshot['trace_error']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if session.preview_asset and Path(session.preview_asset).exists():
        st.image(session.preview_asset, use_container_width=True)
        st.markdown('<div class="preview-caption">Workflow preview</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="console-card">
          <div class="console-row">
            <div>
              <h4>{summary.get('workflow_title', session.title)}</h4>
              <div class="console-muted">{summary.get('message', session.latest_message)}</div>
            </div>
            {pill_html(session.status.value.replace('_', ' '), session.status.value)}
          </div>
          <div class="console-inline-grid compact">
            <div class="console-chip"><span class="console-chip-label">Events</span><span class="console-chip-value">{summary.get('events', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Artifacts</span><span class="console-chip-value">{summary.get('artifacts', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Tool calls</span><span class="console-chip-value">{summary.get('tool_invocations', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Last activity</span><span class="console-chip-value">{summary.get('last_activity_at', session.updated_at)}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_left, overview_right = st.columns([1.2, 1], gap="large")
    with overview_left:
        st.markdown('<div class="console-section-title">Review Signals</div>', unsafe_allow_html=True)
        review_cards = session_snapshot["review_cards"] or []
        if review_cards:
            for review in review_cards[:3]:
                improvements = ", ".join(review["expected_improvements"]) if review["expected_improvements"] else "No explicit checklist."
                st.markdown(
                    f"""
                    <div class="console-card">
                      <h4>{review['artifact_id']}</h4>
                      <div class="console-muted">{review['feedback']}</div>
                      <div class="console-muted">Expected: {improvements}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="console-card"><div class="console-muted">No review rounds recorded.</div></div>', unsafe_allow_html=True)
    with overview_right:
        st.markdown('<div class="console-section-title">Browser State</div>', unsafe_allow_html=True)
        browser_cards = session_snapshot["browser_cards"] or []
        if browser_cards:
            for browser in browser_cards[:4]:
                st.markdown(
                    f"""
                    <div class="console-card">
                      <div class="console-row">
                        <div>
                          <h4>{browser['action']} → {browser['target']}</h4>
                          <div class="console-muted">{browser['purpose'] or browser['surface']}</div>
                          <div class="console-muted">{browser['evidence']}</div>
                        </div>
                        {pill_html(browser['outcome'].replace('_', ' '), browser['outcome'])}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="console-card"><div class="console-muted">No browser actions recorded.</div></div>', unsafe_allow_html=True)

    if summary.get("latest_review_feedback"):
        st.markdown('<div class="console-section-title">Latest Review Feedback</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="console-card">
              <h4>{summary.get('latest_revision_artifact_id', 'Revision')}</h4>
              <div class="console-muted">{summary.get('latest_review_feedback', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_session_artifacts(session_snapshot: dict[str, object]) -> None:
    artifact_previews = session_snapshot["artifact_previews"] or []
    if not artifact_previews:
        st.markdown('<div class="console-card"><div class="console-muted">No artifact previews captured yet.</div></div>', unsafe_allow_html=True)
        return
    for artifact in artifact_previews:
        st.markdown(
            f"""
            <div class="console-card">
              <div class="console-row">
                <div>
                  <h4>{artifact['title']}</h4>
                  <div class="console-muted">{artifact.get('file_name', '')}</div>
                  <div class="console-muted">Revision {artifact['revision']} · score {(artifact['score'] or 0.0) * 100:.1f}% · stage {artifact['source_stage'] or 'n/a'}</div>
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
        if artifact.get("path"):
            st.markdown(f"[Open artifact]({artifact['path']})")


def _render_session_diffs(session_snapshot: dict[str, object]) -> None:
    revision_cards = session_snapshot["revision_cards"] or []
    if not revision_cards:
        st.markdown('<div class="console-card"><div class="console-muted">No revision diff captured yet.</div></div>', unsafe_allow_html=True)
        return
    for revision in revision_cards[:4]:
        st.markdown(
            f"""
            <div class="console-card">
              <div class="console-row">
                <div>
                  <h4>{revision['artifact_id']}</h4>
                  <div class="console-muted">Revision {revision['revision']} · score {(revision['score'] or 0.0) * 100:.1f}%</div>
                </div>
                {pill_html(revision['source_stage'] or 'revision', 'running')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        before_col, after_col = st.columns(2, gap="small")
        with before_col:
            st.caption("Before")
            st.code(revision.get("before_excerpt", "") or "No previous revision.", language="markdown")
        with after_col:
            st.caption("After")
            st.code(revision.get("after_excerpt", "") or revision.get("excerpt", ""), language="markdown")
        st.caption("Diff")
        st.code(revision["diff_excerpt"], language="diff")


def _render_session_timeline(session_snapshot: dict[str, object]) -> None:
    st.markdown('<div class="console-section-title">Session Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="event-line">', unsafe_allow_html=True)
    for event in session_snapshot["events"]:
        st.markdown(
            f"""
            <div class="console-card">
              <div class="console-row">
                <div>
                  <h4>{event.kind.replace('_', ' ').title()}</h4>
                  <div class="console-muted">{event.message}</div>
                  <div class="console-muted">{event.created_at}</div>
                </div>
                {pill_html(f"#{event.sequence}", event.kind)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    if session_snapshot["trace_paths"]:
        st.markdown('<div class="console-section-title">Trace Paths</div>', unsafe_allow_html=True)
        for item in session_snapshot["trace_paths"]:
            st.markdown(f"- [{item['label']}]({item['path']})")


def _render_approval_queue(runtime: LocalAgentRuntime, snapshot: dict[str, object]) -> None:
    st.markdown('<div class="console-section-title">Needs Review</div>', unsafe_allow_html=True)
    approvals = snapshot["approvals"]  # type: ignore[index]
    if approvals:
        for approval in approvals:
            linked_session = next((session for session in snapshot["sessions"] if session.session_id == approval.session_id), None)  # type: ignore[index]
            session_meta = f"{linked_session.workflow_category.replace('_', ' ')} · {linked_session.lane.replace('_', ' ')}" if linked_session else "Session unavailable"
            artifact_meta = linked_session.latest_artifact_title if linked_session and linked_session.latest_artifact_title else "No artifact yet"
            st.markdown(
                f"""
                <div class="console-card">
                  <div class="console-row">
                    <div>
                      <h4>{approval.title}</h4>
                      <div class="console-muted">{approval.reason}</div>
                      <div class="console-muted">{session_meta}</div>
                      <div class="console-muted">{artifact_meta}</div>
                    </div>
                    {pill_html('review', 'awaiting_approval')}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            note = st.text_input("Approval note", key=f"approval_note_{approval.approval_id}")
            approve_col, deny_col = st.columns(2, gap="small")
            with approve_col:
                if st.button("Approve", key=f"approve_{approval.approval_id}", use_container_width=True):
                    runtime.resolve_approval(approval.session_id, decision="approve", note=note)
                    st.rerun()
            with deny_col:
                if st.button("Deny", key=f"deny_{approval.approval_id}", use_container_width=True):
                    runtime.resolve_approval(approval.session_id, decision="deny", note=note)
                    st.rerun()
    else:
        st.markdown(
            """
            <div class="console-card">
              <h4>Approval queue is clear</h4>
              <div class="console-muted">Approval-sensitive workflows will surface here when they hit review gates.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_runtime_profiles(snapshot: dict[str, object]) -> None:
    st.markdown('<div class="console-section-title">Runtime Profiles</div>', unsafe_allow_html=True)
    for profile in snapshot["runtime_profiles"][:5]:  # type: ignore[index]
        st.markdown(
            f"""
            <div class="console-card">
              <div class="console-row">
                <div>
                  <h4>{profile['short_label'] or profile['display_name']}</h4>
                  <div class="console-muted">{profile['capability_family']} · {profile['executor_mode']}</div>
                  <div class="console-muted">Avg readiness {(profile['avg_readiness'] or 0.0) * 100:.1f}% · last request {(profile['last_request_elapsed_ms'] or 0.0):.0f} ms</div>
                  <div class="console-muted">Params {(profile['total_params_b'] or 0.0):.2f}B · warmup {(profile['warmup_load_ms'] or 0.0):.0f} ms · cost {(profile['total_cost_per_mtok'] or 0.0):.2f}/Mtok</div>
                </div>
                {pill_html(profile['run_intent'], 'completed')}
              </div>
              <div class="console-inline-grid compact">
                <div class="console-chip"><span class="console-chip-label">Lanes</span><span class="console-chip-value">{profile['lane_count']}</span></div>
                <div class="console-chip"><span class="console-chip-label">Requests</span><span class="console-chip-value">{profile['requests_completed']}</span></div>
                <div class="console-chip"><span class="console-chip-label">Best lane</span><span class="console-chip-value">{profile['best_lane'] or 'n/a'}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_benchmark_pulse(snapshot: dict[str, object]) -> None:
    comparison_health = snapshot.get("comparison_health", {})
    comparison_batches = snapshot.get("comparison_batches", [])
    st.markdown('<div class="console-section-title">Benchmark Pulse</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="console-card session-summary-card">
          <div class="console-row">
            <div>
              <h4>Full-lane comparison health</h4>
              <div class="console-muted">{len(comparison_batches)} comparison batches in view.</div>
            </div>
            {pill_html('publishable' if comparison_health.get('timed_out', 0) == 0 and comparison_health.get('failed', 0) == 0 else 'mixed', 'completed')}
          </div>
          <div class="console-inline-grid compact">
            <div class="console-chip"><span class="console-chip-label">Completed</span><span class="console-chip-value">{comparison_health.get('completed', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Partial</span><span class="console-chip-value">{comparison_health.get('partial', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Timed out</span><span class="console-chip-value">{comparison_health.get('timed_out', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Coverage</span><span class="console-chip-value">{(comparison_health.get('avg_coverage', 0.0) or 0.0) * 100:.1f}%</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
