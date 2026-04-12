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


def render_mobile_companion(runtime: LocalAgentRuntime) -> None:
    snapshot = build_console_snapshot(runtime)
    selected_profile_id = st.session_state.get("mobile_selected_profile") or default_profile_id(snapshot)
    selected_session_id = st.session_state.get("mobile_selected_session") or default_session_id(snapshot)
    summary = snapshot["board_summary"]
    comparison_health = snapshot["comparison_health"]
    community_signal_summary = snapshot.get("community_signal_summary", {})
    top_local = summary.get("highest_readiness_local_profile", {})
    comparison_batches = summary.get("comparison_batches", [])

    st.markdown(
        f"""
        <div class="mobile-shell">
          <p class="mobile-title">Moonie Mobile Companion</p>
          <p class="mobile-subtitle">Calm mobile layer for active work, approvals, results, and short follow-up actions.</p>
          <div class="mobile-chip-row">
            <span class="mobile-chip">Needs review {snapshot['counts']['approvals']}</span>
            <span class="mobile-chip">Running {snapshot['counts']['running']}</span>
            <span class="mobile-chip">Top local {(top_local.get('avg_readiness', 0.0) or 0.0) * 100:.1f}%</span>
            <span class="mobile-chip">Comparison batches {len(comparison_batches)}</span>
            <span class="mobile-chip">Coverage {(comparison_health.get('avg_coverage', 0.0) or 0.0) * 100:.1f}%</span>
            <span class="mobile-chip">Harnessability {summary.get('harnessability_slice_count', 0)}</span>
            <span class="mobile-chip">Direction {summary.get('direction_following_slice_count', 0)}</span>
            <span class="mobile-chip">Signals {community_signal_summary.get('row_count', 0)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profiles = {profile.system_id: profile for profile in snapshot["profiles"]}
    workflow_cards_by_lane = snapshot.get("workflow_cards_by_lane", {})
    if profiles:
        st.markdown('<div class="console-section-title mobile-heading">Quick Start</div>', unsafe_allow_html=True)
        profile_id = st.selectbox(
            "System profile",
            options=list(profiles.keys()),
            index=max(0, list(profiles.keys()).index(selected_profile_id)) if selected_profile_id in profiles else 0,
            format_func=lambda system_id: profiles[system_id].short_label,
            key="mobile_profile_id",
        )
        st.session_state["mobile_selected_profile"] = profile_id
        lane = st.radio("Lane", ["replayable_core", "live_web_stress"], horizontal=True, key="mobile_lane")
        workflow_cards = {
            workflow["workflow_id"]: workflow
            for workflow in workflow_cards_by_lane.get(lane, snapshot["workflow_cards"])
        }
        if not workflow_cards:
            st.markdown(
                """
                <div class="console-card light mobile-card">
                  <h4>No workflows for this lane</h4>
                  <div class="console-muted">Switch lanes to launch a supported workflow on this surface.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        workflow_id = st.selectbox(
            "Workflow",
            options=list(workflow_cards.keys()),
            format_func=lambda item: workflow_cards[item]["title"],
            key="mobile_workflow_id",
        )
        request = st.text_input("Follow-up note", placeholder="Latest constraint or short operator note.")
        workflow = workflow_cards[workflow_id]
        st.markdown(
            f"""
            <div class="console-card light mobile-card">
              <div class="console-row">
                <div>
                  <h4>{workflow['title']}</h4>
                  <div class="console-muted">{workflow['subtitle']}</div>
                  <div class="console-muted">{workflow['description']}</div>
                </div>
                {pill_html('review' if workflow['supports_approval'] else 'fast path', 'awaiting_approval' if workflow['supports_approval'] else 'completed')}
              </div>
              <div class="console-inline-grid compact">
                <div class="console-chip"><span class="console-chip-label">Active</span><span class="console-chip-value">{workflow['active_sessions']}</span></div>
                <div class="console-chip"><span class="console-chip-label">Review</span><span class="console-chip-value">{workflow['pending_approvals']}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Agent", use_container_width=True):
            session = runtime.launch_session(
                workflow_id=workflow_id,
                system_id=profile_id,
                lane=lane,
                human_request=request,
                background=True,
            )
            st.session_state["mobile_selected_session"] = session.session_id
            st.rerun()

    st.markdown('<div class="console-section-title mobile-heading">Community Signals</div>', unsafe_allow_html=True)
    community_signal_cards = snapshot.get("community_signal_cards", [])
    if community_signal_cards:
        for signal in community_signal_cards[:4]:
            st.markdown(
                f"""
                <div class="console-card light mobile-card">
                  <div class="console-row">
                    <div>
                      <h4>{signal['claim']}</h4>
                      <div class="console-muted">{signal['why_it_matters']}</div>
                      <div class="console-muted">{signal['benchmark_slice'] or 'unassigned'} · {signal['source']}</div>
                    </div>
                    {pill_html(signal['status'].replace('_', ' '), signal['status'])}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="console-card light mobile-card">
              <h4>No community signals loaded</h4>
              <div class="console-muted">Community signals will appear here once the registry is exported or configured.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="console-section-title mobile-heading">Needs Review</div>', unsafe_allow_html=True)
    approvals = snapshot["approvals"]
    if approvals:
        for approval in approvals[:4]:
            linked_session = next((session for session in snapshot["sessions"] if session.session_id == approval.session_id), None)
            session_meta = f"{linked_session.workflow_category.replace('_', ' ')} · {linked_session.lane.replace('_', ' ')}" if linked_session else "Session unavailable"
            st.markdown(
                f"""
                <div class="console-card light mobile-card">
                  <div class="console-row">
                    <div>
                      <h4>{approval.title}</h4>
                      <div class="console-muted">{approval.reason}</div>
                      <div class="console-muted">{session_meta}</div>
                      <div class="console-muted">{linked_session.latest_artifact_title if linked_session and linked_session.latest_artifact_title else 'No artifact yet'}</div>
                    </div>
                    {pill_html('review', 'awaiting_approval')}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            note = st.text_input("Approval note", key=f"mobile_note_{approval.approval_id}")
            approve_col, deny_col = st.columns(2, gap="small")
            with approve_col:
                if st.button("Approve + Finalize", key=f"mobile_approve_{approval.approval_id}", use_container_width=True):
                    runtime.resolve_approval(approval.session_id, decision="approve", note=note)
                    st.rerun()
            with deny_col:
                if st.button("Deny", key=f"mobile_deny_{approval.approval_id}", use_container_width=True):
                    runtime.resolve_approval(approval.session_id, decision="deny", note=note)
                    st.rerun()
    else:
        st.markdown(
            """
            <div class="console-card light mobile-card">
              <h4>No pending reviews</h4>
              <div class="console-muted">Approval-ready work rises here when a session reaches a safe review gate.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="console-section-title mobile-heading">Active Agents</div>', unsafe_allow_html=True)
    for session in snapshot["sessions"][:6]:
        status_label = session.status.value.replace("_", " ")
        st.markdown(
            f"""
            <div class="console-card light mobile-card">
              <div class="console-row">
                <div>
                  <h4>{session.title}</h4>
                  <div class="console-muted">{session.latest_message}</div>
                  <div class="console-muted">{session.workflow_category.replace('_', ' ')} · {session.lane.replace('_', ' ')} · attempt {session.attempt}</div>
                  <div class="console-muted">{session.latest_artifact_title or 'No artifact yet'} · {session.last_activity_at or session.updated_at}</div>
                </div>
                {pill_html(status_label, session.status.value)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Session", key=f"mobile_session_{session.session_id}", use_container_width=True):
            st.session_state["mobile_selected_session"] = session.session_id
            st.rerun()

    selected_snapshot = build_session_snapshot(runtime, selected_session_id) if selected_session_id else None
    st.markdown('<div class="console-section-title mobile-heading">Session Focus</div>', unsafe_allow_html=True)
    if selected_snapshot is None:
        st.markdown(
            """
            <div class="console-card light mobile-card">
              <h4>No session selected</h4>
              <div class="console-muted">Pick a recent session to review progress, artifacts, or approvals.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        _render_mobile_session_focus(runtime, selected_snapshot, comparison_health)

    st.markdown('<div class="console-section-title mobile-heading">Workflow Shelf</div>', unsafe_allow_html=True)
    shelf_cards = workflow_cards_by_lane.get("replayable_core", snapshot["workflow_cards"])
    for workflow in shelf_cards[:4]:
        st.markdown(
            f"""
            <div class="console-card light mobile-card">
              <div class="console-row">
                <div>
                  <h4>{workflow['title']}</h4>
                  <div class="console-muted">{workflow['subtitle']}</div>
                  <div class="console-muted">Recommended {(workflow['recommended_readiness'] or 0.0) * 100:.1f}% · active {workflow['active_sessions']}</div>
                </div>
                {pill_html('review' if workflow['supports_approval'] else 'fast', 'awaiting_approval' if workflow['supports_approval'] else 'completed')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_mobile_session_focus(runtime: LocalAgentRuntime, session_snapshot: dict[str, object], comparison_health: dict[str, object]) -> None:
    session = session_snapshot["session"]
    summary = session_snapshot.get("summary", {})
    st.markdown(
        f"""
        <div class="console-card light mobile-card">
          <div class="console-row">
            <div>
              <h4>{session.title}</h4>
              <div class="console-muted">{session.latest_message}</div>
              <div class="console-muted">{session.workflow_category.replace('_', ' ')} · {session.lane.replace('_', ' ')}</div>
            </div>
            {pill_html(session.status.value.replace('_', ' '), session.status.value)}
          </div>
          <div class="console-inline-grid compact">
            <div class="console-chip"><span class="console-chip-label">Events</span><span class="console-chip-value">{summary.get('events', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Artifacts</span><span class="console-chip-value">{summary.get('artifacts', 0)}</span></div>
            <div class="console-chip"><span class="console-chip-label">Approvals</span><span class="console-chip-value">{summary.get('approvals', 0)}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(max(1, min(3, len(session_snapshot["metric_cards"]))), gap="small")
    for column, metric in zip(metric_columns, session_snapshot["metric_cards"][:3], strict=False):
        with column:
            st.markdown(
                f"""
                <div class="metric-card light">
                  <div class="metric-label">{metric['label']}</div>
                  <div class="metric-value">{metric['value'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    pending_approval = session_snapshot["pending_approval"]
    if pending_approval is not None:
        st.markdown(
            f"""
            <div class="console-card light mobile-card">
              <div class="console-section-title mobile-heading">Approval Context</div>
              <h4>{pending_approval.title}</h4>
              <div class="console-muted">{pending_approval.reason}</div>
              <div class="console-muted">Latest session note: {summary.get('message', session.latest_message)}</div>
              <div class="console-muted">{summary.get('latest_artifact_title', 'No artifact yet') or 'No artifact yet'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if session_snapshot.get("trace_error"):
        st.markdown(
            f"""
            <div class="console-card light mobile-card">
              <div class="console-section-title mobile-heading">Trace Recovery</div>
              <div class="console-muted">{session_snapshot['trace_error']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    artifact_previews = session_snapshot["artifact_previews"]
    if artifact_previews:
        top_artifact = artifact_previews[0]
        if top_artifact["kind"] == "image" and top_artifact.get("image_path") and Path(top_artifact["image_path"]).exists():
            st.image(top_artifact["image_path"], use_container_width=True)
        else:
            st.code(top_artifact.get("preview_excerpt", top_artifact.get("excerpt", "")), language="markdown")
        if top_artifact.get("path"):
            st.markdown(f"[Open artifact]({top_artifact['path']})")

    revision_cards = session_snapshot["revision_cards"]
    if revision_cards:
        st.markdown('<div class="console-section-title mobile-heading">Latest Revision Diff</div>', unsafe_allow_html=True)
        revision = revision_cards[0]
        st.markdown(
            f"""
            <div class="console-card light mobile-card">
              <h4>{revision['artifact_id']}</h4>
              <div class="console-muted">Revision {revision['revision']} · score {(revision['score'] or 0.0) * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(revision.get("before_excerpt", "") or "No previous revision.", language="markdown")
        st.code(revision.get("after_excerpt", "") or revision.get("excerpt", ""), language="markdown")
        st.code(revision["diff_excerpt"], language="diff")

    resume_col, retry_col = st.columns(2, gap="small")
    with resume_col:
        if session_snapshot.get("can_resume") and st.button("Resume", key=f"mobile_resume_{session.session_id}", use_container_width=True):
            runtime.resume_session(session.session_id)
            st.rerun()
    with retry_col:
        if session_snapshot.get("can_retry") and st.button("Retry", key=f"mobile_retry_{session.session_id}", use_container_width=True):
            retried = runtime.retry_session(session.session_id, background=True)
            st.session_state["mobile_selected_session"] = retried.session_id
            st.rerun()

    review_cards = session_snapshot["review_cards"]
    if review_cards:
        review = review_cards[0]
        expected = ", ".join(review["expected_improvements"]) if review["expected_improvements"] else "No checklist."
        st.markdown(
            f"""
            <div class="console-card light mobile-card">
              <h4>{review['artifact_id']}</h4>
              <div class="console-muted">{review['feedback']}</div>
              <div class="console-muted">Expected: {expected}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div class="console-card light mobile-card">
          <div class="console-row">
            <div>
              <h4>Benchmark pulse</h4>
              <div class="console-muted">Completed {comparison_health.get('completed', 0)} · partial {comparison_health.get('partial', 0)} · timed out {comparison_health.get('timed_out', 0)}</div>
            </div>
            {pill_html('coverage', 'completed')}
          </div>
          <div class="console-muted">Average comparison coverage {(comparison_health.get('avg_coverage', 0.0) or 0.0) * 100:.1f}%.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
