from __future__ import annotations

from pathlib import Path

import json

from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.runtime.schemas import ApprovalStatus, SessionStatus


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_lists_packaged_workflows_with_absolute_preview_assets(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    workflows = runtime.list_workflows(lane="replayable_core")

    assert workflows
    dashboard = next(workflow for workflow in workflows if workflow["workflow_id"] == "executive_visual_dashboard_review")
    assert dashboard["episode_id"] == "kwa_exec_visual_dashboard_brief"
    assert Path(dashboard["preview_asset"]).is_absolute()
    live_dashboard = next(workflow for workflow in runtime.list_workflows(lane="live_web_stress") if workflow["workflow_id"] == "executive_visual_dashboard_review")
    assert live_dashboard["episode_id"] == "kwa_exec_live_visual_dashboard_brief"


def test_runtime_profiles_expose_reasoner_budgets(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    profiles = {profile.system_id: profile for profile in runtime.list_system_profiles()}

    assert profiles["hf_service_gemma4_specialists_cpu"].reasoner_max_new_tokens == 96
    assert profiles["hf_service_gemma4_specialists_cpu"].request_timeout_seconds == 600.0
    assert profiles["hf_service_gemma4_e4b_reasoner_only"].reasoner_max_new_tokens == 64
    assert profiles["hf_service_gemma4_e4b_reasoner_only"].run_timeout_seconds == 1800.0


def test_runtime_launches_non_approval_workflow_and_persists_trace(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    session = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        human_request="Keep the brief tight and highlight operator follow-up.",
        background=False,
    )

    assert session.status == SessionStatus.COMPLETED
    assert session.runtime_trace is not None
    assert session.metrics["strict_interface_score"] == 1.0
    assert Path(session.runtime_trace.manifest_path or "").exists()
    assert Path(session.runtime_trace.summary_path or "").exists()
    assert Path(session.runtime_trace.episode_trace_path or "").exists()
    assert session.latest_artifact_title
    assert session.latest_artifact_path

    events = runtime.get_events(session.session_id)
    assert [event.kind for event in events] == ["created", "warming", "running", "artifacts_ready", "completed"]


def test_runtime_approval_flow_uses_same_session_contract(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    session = runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )

    assert session.status == SessionStatus.AWAITING_APPROVAL
    assert session.approvals
    assert session.approvals[0].status == ApprovalStatus.PENDING

    approved = runtime.resolve_approval(session.session_id, decision="approve", note="Looks good.")

    assert approved.status == SessionStatus.COMPLETED
    assert approved.approvals[0].status == ApprovalStatus.APPROVED
    assert approved.approvals[0].note == "Looks good."
    assert approved.active_approval_id is None

    events = runtime.get_events(session.session_id)
    assert [event.kind for event in events][-6:] == ["artifacts_ready", "approval_required", "approved", "approval_resolved", "resumed", "completed"]


def test_runtime_recovers_interrupted_sessions_on_startup(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )

    session_path = (tmp_path / "runtime" / "sessions" / session.session_id / "session.json")
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    payload["latest_message"] = "Executing workflow."
    session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    recovered = LocalAgentRuntime(results_root=tmp_path / "runtime")
    interrupted = recovered.get_session(session.session_id)

    assert interrupted.status == SessionStatus.INTERRUPTED
    assert interrupted.resumable is True
    assert "interrupted" in interrupted.latest_message.lower()
    assert any(event.kind == "interrupted" for event in recovered.get_events(session.session_id))


def test_runtime_resume_session_reexecutes_interrupted_run(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )
    session_path = tmp_path / "runtime" / "sessions" / session.session_id / "session.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    payload["latest_message"] = "Executing workflow."
    session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    recovered = LocalAgentRuntime(results_root=tmp_path / "runtime")
    resumed = recovered.resume_session(session.session_id, note="Continue after interruption.", background=False)

    assert resumed.status == SessionStatus.COMPLETED
    assert resumed.runtime_trace is not None
    events = recovered.get_events(session.session_id)
    assert any(event.kind == "resumed" for event in events)
    assert events[-1].kind == "completed"


def test_runtime_retry_creates_new_attempt_with_lineage(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )

    retried = runtime.retry_session(session.session_id, note="Run again with same constraints.", background=False)

    assert retried.session_id != session.session_id
    assert retried.retry_of_session_id == session.session_id
    assert retried.parent_session_id == session.session_id
    assert retried.attempt == session.attempt + 1
    assert retried.status == SessionStatus.COMPLETED
    assert any(event.kind == "retry_requested" for event in runtime.get_events(session.session_id))


def test_runtime_wait_for_events_supports_cursor_tailing(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )

    tailed = runtime.wait_for_events(session.session_id, after_sequence=2, timeout_s=0.1, poll_s=0.01)

    assert [event.kind for event in tailed] == ["running", "artifacts_ready", "completed"]


def test_runtime_stream_session_returns_status_events_and_pending_approval(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )

    payload = runtime.stream_session(session.session_id, after_sequence=3, timeout_s=0.1, poll_s=0.01)

    assert payload["session"].session_id == session.session_id
    assert payload["pending_approval"] is not None
    assert [event.kind for event in payload["events"]] == ["artifacts_ready", "approval_required"]


def test_runtime_filters_sessions_and_approvals(tmp_path: Path) -> None:
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

    completed_sessions = runtime.list_sessions(status="completed")
    pending_approvals = runtime.list_approvals()
    all_approvals = runtime.list_approvals(status=None)

    assert any(session.session_id == completed.session_id for session in completed_sessions)
    assert all(session.status == SessionStatus.COMPLETED for session in completed_sessions)
    assert len(pending_approvals) == 1
    assert pending_approvals[0].session_id == awaiting.session_id
    assert len(all_approvals) == 1
