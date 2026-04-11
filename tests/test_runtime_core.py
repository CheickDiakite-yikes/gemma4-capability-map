from __future__ import annotations

from pathlib import Path

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

    events = runtime.get_events(session.session_id)
    assert [event.kind for event in events] == ["created", "warming", "running", "completed"]


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

    events = runtime.get_events(session.session_id)
    assert [event.kind for event in events][-2:] == ["approval_required", "approved"]
