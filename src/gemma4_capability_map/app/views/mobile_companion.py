from __future__ import annotations

from pathlib import Path

import streamlit as st

from gemma4_capability_map.app.theme import pill_html
from gemma4_capability_map.app.view_models import build_console_snapshot, default_profile_id
from gemma4_capability_map.runtime.core import LocalAgentRuntime


def render_mobile_companion(runtime: LocalAgentRuntime) -> None:
    snapshot = build_console_snapshot(runtime)
    selected_profile_id = st.session_state.get("mobile_selected_profile") or default_profile_id(snapshot)
    st.markdown(
        """
        <div class="mobile-shell">
          <p class="mobile-title">Moonie Mobile Companion</p>
          <p class="mobile-subtitle">Review active work, keep sessions moving, and approve high-signal tasks from your phone.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profiles = {profile.system_id: profile for profile in snapshot["profiles"]}
    workflows = {workflow["workflow_id"]: workflow for workflow in snapshot["workflows"]}

    with st.container(border=False):
        st.markdown('<div class="console-section-title" style="color:#112318;">Quick Start</div>', unsafe_allow_html=True)
        profile_id = st.selectbox(
            "System profile",
            options=list(profiles.keys()),
            index=max(0, list(profiles.keys()).index(selected_profile_id)) if selected_profile_id in profiles else 0,
            format_func=lambda system_id: profiles[system_id].short_label,
            key="mobile_profile_id",
        )
        st.session_state["mobile_selected_profile"] = profile_id
        workflow_id = st.selectbox(
            "Workflow",
            options=list(workflows.keys()),
            format_func=lambda item: workflows[item]["title"],
            key="mobile_workflow_id",
        )
        lane = st.radio("Lane", ["replayable_core", "live_web_stress"], horizontal=True, key="mobile_lane")
        if st.button("Start Agent", use_container_width=True):
            runtime.launch_session(workflow_id=workflow_id, system_id=profile_id, lane=lane, background=True)
            st.rerun()

    st.markdown('<div class="console-section-title" style="color:#112318;">Needs Review</div>', unsafe_allow_html=True)
    if snapshot["approvals"]:
        for approval in snapshot["approvals"]:
            st.markdown(
                f"""
                <div class="console-card light">
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
            approve_col, deny_col = st.columns(2, gap="small")
            with approve_col:
                if st.button("Approve", key=f"mobile_approve_{approval.approval_id}", use_container_width=True):
                    runtime.resolve_approval(approval.session_id, decision="approve")
                    st.rerun()
            with deny_col:
                if st.button("Deny", key=f"mobile_deny_{approval.approval_id}", use_container_width=True):
                    runtime.resolve_approval(approval.session_id, decision="deny")
                    st.rerun()
    else:
        st.markdown(
            """
            <div class="console-card light">
              <h4>No pending reviews</h4>
              <div class="console-muted">Approval-ready work will rise here as sessions hit review gates.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="console-section-title" style="color:#112318;">Active Sessions</div>', unsafe_allow_html=True)
    for session in snapshot["sessions"][:6]:
        st.markdown(
            f"""
            <div class="console-card light">
              <div class="console-row">
                <div>
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
        if session.artifact_paths:
            st.caption("Artifacts")
            for artifact_path in session.artifact_paths[:2]:
                st.markdown(f"- [{Path(artifact_path).name}]({artifact_path})")

    st.markdown('<div class="console-section-title" style="color:#112318;">Top Local Profiles</div>', unsafe_allow_html=True)
    for row in snapshot["top_local_rows"][:3]:
        st.markdown(
            f"""
            <div class="console-card light">
              <div class="console-row">
                <div>
                  <h4>{row.get('short_label') or row.get('display_name')}</h4>
                  <div class="console-muted">{row.get('lane', '').replace('_', ' ')}</div>
                </div>
                {pill_html(f"{(row.get('real_world_readiness_avg') or 0.0) * 100:.1f}%", 'completed')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
