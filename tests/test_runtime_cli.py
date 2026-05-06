from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma4_capability_map.runtime import cli as runtime_cli
from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.runtime.sandbox import DEFAULT_SANDBOX_POLICY_ID


def test_runtime_cli_lists_workflows(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: LocalAgentRuntime(results_root=tmp_path / "runtime"))
    monkeypatch.setattr(runtime_cli, "parse_args", lambda: runtime_cli.argparse.Namespace(command="workflows", lane="replayable_core"))

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert any(workflow["workflow_id"] == "executive_visual_dashboard_review" for workflow in output)


def test_runtime_cli_lists_pending_approvals(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(runtime_cli, "parse_args", lambda: runtime_cli.argparse.Namespace(command="approvals", all=False))

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert len(output) == 1
    assert output[0]["status"] == "pending"


def test_runtime_cli_retry_round_trip(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="executive_visual_dashboard_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="retry",
            session_id=session.session_id,
            note="CLI retry",
            background=False,
            timeout_s=30.0,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["retry_of_session_id"] == session.session_id
    assert output["status"] == "completed"


def test_runtime_cli_watch_returns_session_and_events(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="watch",
            session_id=session.session_id,
            after=3,
            timeout_s=0.1,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["session"]["session_id"] == session.session_id
    assert output["pending_approval"]["session_id"] == session.session_id
    event_kinds = [event["kind"] for event in output["events"]]
    assert "artifacts_ready" in event_kinds
    assert "approval_required" in event_kinds
    assert event_kinds[-2:] == ["artifacts_ready", "approval_required"]


def test_runtime_cli_live_launches_packaged_workflow_and_attaches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    attached: dict[str, str] = {}

    def fake_attach(runtime_arg: LocalAgentRuntime, session_id: str, **_: object):
        attached["session_id"] = session_id
        return runtime_arg.wait_for_session(session_id, timeout_s=30.0)

    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(runtime_cli, "attach_to_session", fake_attach)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="live",
            workflow_id="executive_visual_dashboard_review",
            system_id="oracle_gemma4_e2b",
            lane="replayable_core",
            title=None,
            human_request="CLI live smoke.",
            project_id=None,
            refresh_s=0.1,
            timeout_s=0.1,
            once=True,
            sandbox_mode="ephemeral_copy",
            sandbox_policy_id=DEFAULT_SANDBOX_POLICY_ID,
        ),
    )

    runtime_cli.main()

    assert attached["session_id"]
    session = runtime.get_session(attached["session_id"])
    assert session.status.value == "completed"
    assert session.system_id == "oracle_gemma4_e2b"
    assert session.sandbox_root
    assert session.sandbox_policy_id == DEFAULT_SANDBOX_POLICY_ID
