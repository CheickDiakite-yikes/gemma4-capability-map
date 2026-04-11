from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma4_capability_map.runtime import cli as runtime_cli
from gemma4_capability_map.runtime.core import LocalAgentRuntime


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
    assert [event["kind"] for event in output["events"]] == ["artifacts_ready", "approval_required"]
