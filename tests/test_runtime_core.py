from __future__ import annotations

from pathlib import Path

import json

from gemma4_capability_map.knowledge_work.schemas import BenchmarkLane, BrowserAction, EpisodeTrace, RoleFamily
from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.runtime.sandbox import SandboxViolation, assert_path_inside, sandbox_policy_blocks_for_trace
from gemma4_capability_map.runtime.schemas import ApprovalStatus, SessionStatus
from gemma4_capability_map.runtime.workflows import DEFAULT_WORKFLOWS_PATH, validate_packaged_workflows


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
    parallel_audit = next(workflow for workflow in runtime.list_workflows(lane="live_web_stress") if workflow["workflow_id"] == "ops_parallel_audit_review")
    assert parallel_audit["episode_id"] == "kwa_ops_live_parallel_audit_review_v1"
    assert "parallel_tool_calling" in parallel_audit["tags"]


def test_packaged_workflow_registry_resolves_declared_episode_lanes() -> None:
    assert validate_packaged_workflows(DEFAULT_WORKFLOWS_PATH) == []


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
        project_id="research-alpha",
        human_request="Keep the brief tight and highlight operator follow-up.",
        background=False,
    )

    assert session.status == SessionStatus.COMPLETED
    assert session.project_id == "research-alpha"
    assert session.latest_instruction == "Keep the brief tight and highlight operator follow-up."
    assert session.instruction_history
    assert session.instruction_history[0].source == "launch"
    assert session.instruction_history[0].content == "Keep the brief tight and highlight operator follow-up."
    assert session.runtime_trace is not None
    assert session.sandbox_mode == "ephemeral_copy"
    assert Path(session.sandbox_root).exists()
    assert Path(session.sandbox_manifest_path).exists()
    assert session.runtime_trace.sandbox_root == session.sandbox_root
    assert Path(session.runtime_trace.output_dir).is_relative_to(Path(session.sandbox_root))
    assert Path(session.runtime_trace.manifest_path or "").is_relative_to(Path(session.sandbox_root))
    assert all(Path(path).is_relative_to(Path(session.sandbox_root)) for path in session.runtime_trace.artifact_paths)
    assert session.metrics["strict_interface_score"] == 1.0
    assert Path(session.runtime_trace.manifest_path or "").exists()
    assert Path(session.runtime_trace.summary_path or "").exists()
    assert Path(session.runtime_trace.episode_trace_path or "").exists()
    assert session.latest_artifact_title
    assert session.latest_artifact_path

    history = runtime.get_session_history(session.session_id)
    assert history["session"].project_id == "research-alpha"
    assert history["instruction_history"]
    assert history["artifact_history"]

    events = runtime.get_events(session.session_id)
    event_kinds = [event.kind for event in events]
    assert event_kinds[0:4] == ["created", "instruction_updated", "sandbox_prepared", "warming"]
    assert "running" in event_kinds
    assert "tool_call_attempt" in event_kinds
    assert "tool_call_result" in event_kinds
    assert "artifact_revision" in event_kinds
    assert event_kinds[-2:] == ["artifacts_ready", "completed"]


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
    event_kinds = [event.kind for event in events]
    assert "approval_required" in event_kinds
    assert "approved" in event_kinds
    assert "approval_resolved" in event_kinds
    assert "resume_started" in event_kinds
    assert "resumed" in event_kinds
    assert event_kinds[-1] == "completed"


def test_parallel_audit_live_sandbox_manifest_records_replay_attribution(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    session = runtime.launch_session(
        workflow_id="ops_parallel_audit_review",
        system_id="oracle_gemma4_e2b",
        lane="live_web_stress",
        background=False,
    )

    manifest = json.loads(Path(session.sandbox_manifest_path).read_text(encoding="utf-8"))
    assert manifest["entrypoint"] == "packaged_workflow"
    assert manifest["workflow_id"] == "ops_parallel_audit_review"
    assert manifest["episode_id"] == "kwa_ops_live_parallel_audit_review_v1"
    assert manifest["source_replay_cases"] == ["parallel_audit_array_literal"]
    assert "parallel_tool_calling" in manifest["workflow_tags"]
    assert "operations_audit" in manifest["episode_tags"]
    assert manifest["artifact_targets"] == [
        {
            "artifact_id": "live_parallel_audit_note",
            "kind": "memo",
            "path_or_target": "workspaces/ops-live-parallel-audit-review/live_parallel_audit_note.docx",
        }
    ]
    assert manifest["live_web_policy"]["packaged_workflows_only"] is True
    assert manifest["allowed_write_roots"] == [str((Path(session.sandbox_root) / "output").resolve())]


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
    assert resumed.latest_instruction == "Continue after interruption."
    assert resumed.instruction_history[-1].source == "resume"
    events = recovered.get_events(session.session_id)
    assert "resume_requested" in [event.kind for event in events]
    assert "resume_started" in [event.kind for event in events]
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

    assert any(event.kind == "running" for event in tailed)
    assert any(event.kind == "completed" for event in tailed)


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
    assert any(event.kind == "artifacts_ready" for event in payload["events"])
    assert any(event.kind == "approval_required" for event in payload["events"])


def test_runtime_filters_sessions_and_approvals(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    completed = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        project_id="alpha",
        background=False,
    )
    awaiting = runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        project_id="alpha",
        background=False,
    )
    beta = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        project_id="beta",
        background=False,
    )

    completed_sessions = runtime.list_sessions(status="completed")
    alpha_sessions = runtime.list_sessions(project_id="alpha")
    pending_approvals = runtime.list_approvals()
    all_approvals = runtime.list_approvals(status=None)

    assert any(session.session_id == completed.session_id for session in completed_sessions)
    assert all(session.status == SessionStatus.COMPLETED for session in completed_sessions)
    assert {session.project_id for session in alpha_sessions} == {"alpha"}
    assert any(session.session_id == beta.session_id for session in runtime.list_sessions(project_id="beta"))
    assert len(pending_approvals) == 1
    assert pending_approvals[0].session_id == awaiting.session_id
    assert len(all_approvals) == 1


def test_runtime_sandbox_blocks_path_escapes(tmp_path: Path) -> None:
    sandbox_root = tmp_path / "sandbox"
    allowed = sandbox_root / "output" / "artifact.txt"
    escaped = tmp_path / "outside.txt"

    assert assert_path_inside(allowed, sandbox_root) == allowed.resolve()
    try:
        assert_path_inside(escaped, sandbox_root)
    except SandboxViolation as exc:
        assert "escapes sandbox root" in str(exc)
    else:  # pragma: no cover - explicit safety assertion
        raise AssertionError("Expected sandbox path escape to be blocked.")


def test_sandbox_policy_records_live_web_side_effect_blocks() -> None:
    trace = EpisodeTrace(
        run_id="policy-live-web",
        episode_id="kwa_policy_live",
        role_family=RoleFamily.EXECUTIVE_ASSISTANT,
        lane=BenchmarkLane.LIVE_WEB_STRESS,
        workspace_id="policy-live",
        browser_actions=[
            BrowserAction(
                stage_id="stage_1",
                action="submit_form",
                target="https://example.test/apply",
                purpose="Submit only to the sandbox mirror.",
                expected_signal="Sandbox endpoint receives the dry-run payload.",
                submission_gate="sandbox_only",
                sandbox_endpoint="https://sandbox.local/policy-live/submit",
                status="dry_run",
            )
        ],
    )

    blocks = sandbox_policy_blocks_for_trace(trace=trace, lane=trace.lane)

    assert len(blocks) == 1
    assert blocks[0].severity == "info"
    assert blocks[0].reason == "Live-web side effect held inside sandbox."
    assert blocks[0].sandbox_endpoint == "https://sandbox.local/policy-live/submit"


def test_runtime_records_live_sandbox_policy_blocks(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")

    session = runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="live_web_stress",
        background=False,
    )

    assert session.sandbox_policy_blocks
    assert session.runtime_trace is not None
    assert session.runtime_trace.sandbox_policy_blocks == session.sandbox_policy_blocks
    assert any(event.kind == "sandbox_policy_block" for event in runtime.get_events(session.session_id))


def test_runtime_resolves_approval_by_stable_id_and_tracks_history(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        project_id="project-x",
        background=False,
    )

    approval_id = session.approvals[0].approval_id
    resolved = runtime.resolve_approval_by_id(approval_id, decision="approve", note="Ship it.", resume=True)
    history = runtime.get_session_history(session.session_id)

    assert resolved.approvals[0].approval_id == approval_id
    assert resolved.approvals[0].status == ApprovalStatus.APPROVED
    assert history["session"].project_id == "project-x"
    assert history["session"].latest_instruction == "Ship it."
    assert history["instruction_history"][-1].content == "Ship it."
    assert history["artifact_history"]
