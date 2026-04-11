from __future__ import annotations

from pathlib import Path

import streamlit as st

from gemma4_capability_map.app.theme import pill_html
from gemma4_capability_map.app.view_models import build_console_snapshot, default_profile_id, default_session_id
from gemma4_capability_map.runtime.core import LocalAgentRuntime


def render_operator_console(runtime: LocalAgentRuntime) -> None:
    snapshot = build_console_snapshot(runtime)
    selected_session_id = st.session_state.get("operator_selected_session") or default_session_id(snapshot)
    selected_profile_id = st.session_state.get("operator_selected_profile") or default_profile_id(snapshot)

    st.markdown(
        f"""
        <div class="console-hero">
          <p class="console-title">Moonie Operator Console</p>
          <p class="console-subtitle">Local-first agent runtime, benchmark-backed workflows, and reviewable approval flows.</p>
          <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Sessions</div><div class="metric-value">{snapshot['counts']['sessions']}</div></div>
            <div class="metric-card"><div class="metric-label">Running</div><div class="metric-value">{snapshot['counts']['running']}</div></div>
            <div class="metric-card"><div class="metric-label">Needs Review</div><div class="metric-value">{snapshot['counts']['approvals']}</div></div>
            <div class="metric-card"><div class="metric-label">Completed</div><div class="metric-value">{snapshot['counts']['completed']}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    refresh_col, summary_col = st.columns([1, 3], gap="small")
    with refresh_col:
        if st.button("Refresh Console", use_container_width=True):
            st.rerun()
    with summary_col:
        for card in snapshot["lane_cards"]:
            st.markdown(
                f"""
                <div class="console-card">
                  <div class="console-row">
                    <div>
                      <div class="console-section-title">{card['label']} leader</div>
                      <h4>{card['short_label']}</h4>
                      <div class="console-muted">Readiness {card['readiness'] * 100:.1f}% · Strict {card['strict'] * 100:.1f}% · Browser {card['browser'] * 100:.1f}%</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, center, right = st.columns([1.15, 1.9, 1.25], gap="large")

    with left:
        st.markdown('<div class="console-shell">', unsafe_allow_html=True)
        st.markdown('<div class="console-section-title">Launch Workflow</div>', unsafe_allow_html=True)
        profile_options = {profile.system_id: profile for profile in snapshot["profiles"]}
        workflow_options = {workflow["workflow_id"]: workflow for workflow in snapshot["workflows"]}
        profile_id = st.selectbox(
            "System profile",
            options=list(profile_options.keys()),
            index=max(0, list(profile_options.keys()).index(selected_profile_id)) if selected_profile_id in profile_options else 0,
            format_func=lambda system_id: profile_options[system_id].short_label,
        )
        st.session_state["operator_selected_profile"] = profile_id
        workflow_id = st.selectbox(
            "Workflow",
            options=list(workflow_options.keys()),
            format_func=lambda item: workflow_options[item]["title"],
        )
        lane = st.radio("Lane", ["replayable_core", "live_web_stress"], horizontal=True)
        request = st.text_area("Task framing", placeholder="Add operator notes or constraints for this run.")
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

        st.markdown('<div class="console-section-title">Packaged Workflows</div>', unsafe_allow_html=True)
        for workflow in snapshot["workflows"]:
            st.markdown(
                f"""
                <div class="console-card">
                  <div class="console-row">
                    <div>
                      <h4>{workflow['title']}</h4>
                      <div class="console-muted">{workflow['subtitle']}</div>
                      <div class="console-muted">{workflow['category'].replace('_', ' ')}</div>
                    </div>
                    {pill_html('review' if workflow['supports_approval'] else 'fast path', 'awaiting_approval' if workflow['supports_approval'] else 'completed')}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="console-section-title">Recent Sessions</div>', unsafe_allow_html=True)
        for session in snapshot["sessions"][:8]:
            if st.button(f"{session.title} · {session.status.value.replace('_', ' ')}", key=f"session_{session.session_id}", use_container_width=True):
                st.session_state["operator_selected_session"] = session.session_id
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with center:
        selected_session = next((session for session in snapshot["sessions"] if session.session_id == selected_session_id), None)
        if selected_session is None:
            st.markdown(
                """
                <div class="console-shell">
                  <div class="console-section-title">Active Session</div>
                  <div class="console-card">
                    <h4>No active session yet</h4>
                    <div class="console-muted">Launch a packaged workflow to see traces, artifacts, approvals, and benchmark-backed quality signals here.</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            _render_session_detail(runtime, selected_session)

    with right:
        st.markdown('<div class="console-shell">', unsafe_allow_html=True)
        st.markdown('<div class="console-section-title">Needs Review</div>', unsafe_allow_html=True)
        if snapshot["approvals"]:
            for approval in snapshot["approvals"]:
                st.markdown(
                    f"""
                    <div class="console-card">
                      <div class="console-row">
                        <div>
                          <h4>{approval.title}</h4>
                          <div class="console-muted">{approval.reason}</div>
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
                  <div class="console-muted">Approval-sensitive workflows will surface here after they produce review-ready work.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="console-section-title">Local Systems</div>', unsafe_allow_html=True)
        for row in snapshot["top_local_rows"]:
            st.markdown(
                f"""
                <div class="console-card">
                  <div class="console-row">
                    <div>
                      <h4>{row.get('short_label') or row.get('display_name')}</h4>
                      <div class="console-muted">{row.get('lane', '').replace('_', ' ')}</div>
                    </div>
                    {pill_html(f"{(row.get('real_world_readiness_avg') or 0.0) * 100:.1f}%", 'completed')}
                  </div>
                  <div class="console-muted">Artifact {(row.get('artifact_quality_avg') or 0.0) * 100:.1f}% · Strict {(row.get('strict_interface_avg') or 0.0) * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_session_detail(runtime: LocalAgentRuntime, session) -> None:  # noqa: ANN001
    events = runtime.get_events(session.session_id)
    st.markdown('<div class="console-shell">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="console-card">
          <div class="console-row">
            <div>
              <div class="console-section-title">{session.workflow_category.replace('_', ' ')}</div>
              <h4>{session.title}</h4>
              <div class="console-muted">{session.latest_message}</div>
            </div>
            {pill_html(session.status.value.replace('_', ' '), session.status.value)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if session.preview_asset and Path(session.preview_asset).exists():
        st.image(session.preview_asset, use_container_width=True)
        st.markdown('<div class="preview-caption">Workflow preview</div>', unsafe_allow_html=True)
    if session.metrics:
        cols = st.columns(min(5, len(session.metrics)), gap="small")
        labels = [
            ("role_readiness_score", "Readiness"),
            ("artifact_quality_score", "Artifact"),
            ("browser_workflow_score", "Browser"),
            ("strict_interface_score", "Strict"),
            ("recovered_execution_score", "Recovered"),
        ]
        for col, (key, label) in zip(cols, labels, strict=False):
            with col:
                value = float(session.metrics.get(key, 0.0))
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-label">{label}</div>
                      <div class="metric-value">{value * 100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="console-section-title">Session Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="event-line">', unsafe_allow_html=True)
    for event in events:
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

    if session.artifact_paths:
        st.markdown('<div class="console-section-title">Artifacts</div>', unsafe_allow_html=True)
        for artifact_path in session.artifact_paths:
            st.markdown(f"- [{Path(artifact_path).name}]({artifact_path})")
    if session.runtime_trace is not None:
        st.markdown('<div class="console-section-title">Trace Paths</div>', unsafe_allow_html=True)
        st.markdown(f"- [manifest.json]({session.runtime_trace.manifest_path})")
        st.markdown(f"- [summary.json]({session.runtime_trace.summary_path})")
        st.markdown(f"- [episode_trace.json]({session.runtime_trace.episode_trace_path})")
    st.markdown("</div>", unsafe_allow_html=True)
