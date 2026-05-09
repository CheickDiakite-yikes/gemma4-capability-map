from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma4_capability_map.runtime import cli as runtime_cli
from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.runtime.sandbox import DEFAULT_SANDBOX_POLICY_ID


def test_runtime_cli_lists_workflows(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: LocalAgentRuntime(results_root=tmp_path / "runtime"))
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(command="workflows", lane="replayable_core", workflow_id=None, validate=False),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert any(workflow["workflow_id"] == "executive_visual_dashboard_review" for workflow in output)


def test_runtime_cli_validates_and_filters_parallel_workflow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: LocalAgentRuntime(results_root=tmp_path / "runtime"))
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="workflows",
            lane="live_web_stress",
            workflow_id="ops_parallel_audit_review",
            validate=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["validation_errors"] == []
    assert output["workflow_count"] == 1
    assert output["workflows"][0]["episode_id"] == "kwa_ops_live_parallel_audit_review_v1"


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


def test_runtime_cli_attach_can_apply_approval_action(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="finance_visual_invoice_review",
        system_id="oracle_gemma4_e2b",
        lane="replayable_core",
        background=False,
    )
    attached: dict[str, str] = {}

    def fake_attach(runtime_arg: LocalAgentRuntime, session_id: str, **_: object):
        attached["session_id"] = session_id
        return runtime_arg.get_session(session_id)

    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(runtime_cli, "attach_to_session", fake_attach)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="attach",
            session_id=session.session_id,
            refresh_s=0.1,
            timeout_s=0.1,
            once=True,
            action="approve",
            note="Approved from operator.",
            no_resume=False,
            foreground=True,
        ),
    )

    runtime_cli.main()

    approved = runtime.get_session(session.session_id)
    assert attached["session_id"] == session.session_id
    assert approved.status.value == "completed"
    assert approved.approvals[0].status.value == "approved"
    assert approved.approvals[0].note == "Approved from operator."


def test_runtime_cli_inspect_outputs_sandbox_and_artifacts(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
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
            command="inspect",
            session_id=session.session_id,
            target="all",
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["session_id"] == session.session_id
    assert output["sandbox"]["root"] == session.sandbox_root
    assert output["sandbox"]["manifest_exists"] is True
    assert output["artifacts"]
    assert all(item["exists"] for item in output["artifacts"])
    assert output["summary"]["summary_path"]
    assert output["scorecard"]["metrics"]["role_readiness_score"] >= 0.0
    assert isinstance(output["scorecard"]["controller_findings"], list)


def test_runtime_cli_inspect_scorecard_target(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
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
            command="inspect",
            session_id=session.session_id,
            target="scorecard",
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"session_id", "status", "workflow_id", "system_id", "lane", "scorecard"}
    assert output["scorecard"]["metrics"]["strict_interface_score"] == 1.0


def test_runtime_cli_inspect_policy_target_lists_live_web_blocks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    session = runtime.launch_session(
        workflow_id="jobs_visual_form_hold",
        system_id="oracle_gemma4_e2b",
        lane="live_web_stress",
        background=False,
    )
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="inspect",
            session_id=session.session_id,
            target="policy",
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["policy_blocks"]
    assert any(block["submission_gate"] == "approval_required" for block in output["policy_blocks"])
    assert all(block["sandbox_endpoint"] for block in output["policy_blocks"])


def test_runtime_cli_gemini_baseline_writes_dry_run_packet(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="gemini-baseline",
            workflow_id="executive_visual_dashboard_review",
            lane="replayable_core",
            binary="definitely-missing-gemini-cli",
            execute=False,
            timeout_s=5.0,
            output_dir=str(tmp_path / "gemini-baseline"),
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["workflow_id"] == "executive_visual_dashboard_review"
    assert output["dry_run"] is True
    assert output["availability"]["available"] is False
    assert Path(output["output_path"]).exists()


def test_runtime_cli_report_json_inspects_generated_report_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    report_dir = _write_fake_research_report(tmp_path / "report")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="report",
            report_id="custom",
            report_dir=str(report_dir),
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["report_dir"] == str(report_dir.resolve())
    assert output["packet_count"] == 2
    assert output["prompt_contract_candidate_count"] == 2
    assert output["prompt_contract_candidate_ids"] == ["schema_anchor_v1", "literal_argument_guard_v1"]
    assert len(output["tables"]) == 1
    assert len(output["figures"]) == 1


def test_runtime_cli_report_renders_rich_overview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    report_dir = _write_fake_research_report(tmp_path / "report")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="report",
            report_id="custom",
            report_dir=str(report_dir),
            json=False,
        ),
    )

    runtime_cli.main()

    output = capsys.readouterr().out
    assert "Moonie Research Report" in output
    assert "Prompt-Contract Candidates" in output
    assert "schema_anchor_v1" in output


def test_runtime_cli_packet_json_inspects_prompt_contract_probe_packet(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_prompt_contract_packet(tmp_path / "packet")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="prompt-contract-probe",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["packet_dir"] == str(packet_dir.resolve())
    assert output["candidate_count"] == 2
    assert output["executed_count"] == 1
    assert output["dry_run_count"] == 1
    assert output["command_count"] == 2
    assert output["candidate_rows"][0]["tool_prompt_contract_id"] == "schema_anchor_v1"


def test_runtime_cli_packet_json_inspects_tool_catalog_profile_probe_packet(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_catalog_profile_packet(tmp_path / "catalog_packet")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-catalog-profile-probe",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["packet_kind"] == "tool-catalog-profile-probe"
    assert output["candidate_count"] == 1
    assert output["candidate_rows"][0]["tool_catalog_profile_id"] == "visual_role_catalog_v1"


def test_runtime_cli_packet_json_inspects_visual_hard_slice_probe_packet(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_visual_hard_slice_packet(tmp_path / "visual_packet")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="visual-hard-slice-probe",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["packet_kind"] == "visual-hard-slice-probe"
    assert output["candidate_count"] == 2
    assert output["candidate_rows"][1]["hard_slice_gate"] == "no_directive_reference"


def test_runtime_cli_packet_renders_rich_overview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_prompt_contract_packet(tmp_path / "packet")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="prompt-contract-probe",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=False,
        ),
    )

    runtime_cli.main()

    output = capsys.readouterr().out
    assert "Moonie Research Packet" in output
    assert "schema_anchor_v1" in output
    assert "literal_argument_guard_v1" in output


def test_runtime_cli_packet_json_inspects_tool_probe_replay_packet(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_packet(tmp_path / "replay")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-probe-replay",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["packet_kind"] == "tool-probe-replay"
    assert output["case_count"] == 2
    assert output["failure_mode_counts"] == {"argument_mismatch": 1, "no_tool_call": 1}
    assert output["replay_case_rows"][0]["case_id"] == "cli_invoice_lock_hyphen_query"
    assert output["next_action_rows"][1]["next_action"] == "build_parallel_array_replay_or_workflow"


def test_runtime_cli_packet_renders_tool_probe_replay_overview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_packet(tmp_path / "replay")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-probe-replay",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=False,
        ),
    )

    runtime_cli.main()

    output = capsys.readouterr().out
    assert "Moonie Research Packet" in output
    assert "Replay Cases" in output
    assert "Next Actions" in output
    assert "no_tool_call" in output


def test_runtime_cli_replay_live_json_writes_dry_run_packet(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_packet(tmp_path / "replay")
    output_dir = tmp_path / "live-replay"
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="replay-live",
            packet_id="latest",
            packet_dir=str(packet_dir),
            output_dir=str(output_dir),
            system_id="mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
            registry=str(tmp_path / "registry.yaml"),
            case_ids=["parallel_audit_array_literal"],
            execute=False,
            refresh_s=0.1,
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["case_count"] == 1
    assert output["summary"]["execute"] is False
    assert output["case_states"][0]["case_id"] == "parallel_audit_array_literal"
    assert output["case_states"][0]["status"] == "dry_run"
    assert output["manifest"]["entrypoint"] == "moonie-agent replay-live"
    assert (output_dir / "live_case_states.csv").exists()


def test_runtime_cli_packet_json_inspects_tool_probe_replay_live_packet(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_live_packet(tmp_path / "live-replay")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-probe-replay-live",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["packet_kind"] == "tool-probe-replay-live"
    assert output["case_count"] == 1
    assert output["execute"] is True
    assert output["executed_count"] == 1
    assert output["exact_rate"] == 0.0
    assert output["case_state_rows"][0]["status"] == "non_exact"


def test_runtime_cli_packet_renders_tool_probe_replay_live_overview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_live_packet(tmp_path / "live-replay")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-probe-replay-live",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=False,
        ),
    )

    runtime_cli.main()

    output = capsys.readouterr().out
    assert "Moonie Research Packet" in output
    assert "Live Replay Cases" in output
    assert "no_tool_call" in output


def test_runtime_cli_packet_json_inspects_tool_probe_replay_live_comparison(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_live_comparison(tmp_path / "live-comparison")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-probe-replay-live-comparison",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["packet_kind"] == "tool-probe-replay-live-comparison"
    assert output["shared_case_count"] == 1
    assert output["delta_exact_rate"] == -1.0
    assert output["case_delta_rows"][0]["candidate_replay_failure_mode"] == "no_tool_call"


def test_runtime_cli_packet_renders_tool_probe_replay_live_comparison(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_live_comparison(tmp_path / "live-comparison")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-probe-replay-live-comparison",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=False,
        ),
    )

    runtime_cli.main()

    output = capsys.readouterr().out
    assert "Moonie Research Packet" in output
    assert "Live Replay Comparison" in output
    assert "Delta exact" in output


def test_runtime_cli_packet_json_inspects_tool_probe_replay_live_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_live_diagnostic(tmp_path / "live-diagnostic")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-probe-replay-live-diagnostic",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=True,
        ),
    )

    runtime_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["packet_kind"] == "tool-probe-replay-live-diagnostic"
    assert output["packet_count"] == 3
    assert output["case_count"] == 1
    assert output["diagnosis_counts"] == {"visual_literal_argument_mismatch": 1}
    assert output["diagnostic_rows"][0]["packet_label"] == "visual_role_catalog_v1"


def test_runtime_cli_packet_renders_tool_probe_replay_live_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    packet_dir = _write_fake_tool_probe_replay_live_diagnostic(tmp_path / "live-diagnostic")
    monkeypatch.setattr(runtime_cli, "LocalAgentRuntime", lambda: runtime)
    monkeypatch.setattr(
        runtime_cli,
        "parse_args",
        lambda: runtime_cli.argparse.Namespace(
            command="packet",
            kind="tool-probe-replay-live-diagnostic",
            packet_id="latest",
            packet_dir=str(packet_dir),
            json=False,
        ),
    )

    runtime_cli.main()

    output = capsys.readouterr().out
    assert "Moonie Research Packet" in output
    assert "Live Replay Diagnostics" in output
    assert "visual_literal_argument_mismatch" in output


def _write_fake_research_report(report_dir: Path) -> Path:
    (report_dir / "tables").mkdir(parents=True)
    (report_dir / "figures").mkdir()
    (report_dir / "report.md").write_text("# Fake report\n", encoding="utf-8")
    (report_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-07T00:00:00+00:00",
                "table_count": 1,
                "figure_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "packet_summary": [{"packet": "H1f"}, {"packet": "H1i"}],
                "prompt_contract_candidates": [
                    {"tool_prompt_contract_id": "schema_anchor_v1"},
                    {"tool_prompt_contract_id": "literal_argument_guard_v1"},
                ],
                "gemini": {"packet_run_id": "gemini_dry_run", "dry_run": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "tables" / "packet_summary.csv").write_text("packet\nH1f\n", encoding="utf-8")
    (report_dir / "figures" / "summary.svg").write_text("<svg></svg>\n", encoding="utf-8")
    return report_dir


def _write_fake_prompt_contract_packet(packet_dir: Path) -> Path:
    packet_dir.mkdir(parents=True)
    (packet_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packet_run_id": "fake_packet",
                "created_at": "2026-05-07T00:00:00+00:00",
                "execute": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "commands.json").write_text(
        json.dumps(
            [
                {"system_id": "schema", "command": ["run", "schema"]},
                {"system_id": "literal", "command": ["run", "literal"]},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "results.json").write_text(
        json.dumps({"candidate_count": 2, "executed_count": 1, "dry_run_count": 1})
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "candidate_summary.csv").write_text(
        "\n".join(
            [
                "system_id,tool_prompt_contract_id,execute,output_dir,comparison_path,exact_match_rate,executable_match_rate",
                "schema,schema_anchor_v1,True,/tmp/schema,/tmp/schema/probe_comparison.json,0.5,0.5",
                "literal,literal_argument_guard_v1,False,/tmp/literal,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return packet_dir


def _write_fake_tool_catalog_profile_packet(packet_dir: Path) -> Path:
    packet_dir.mkdir(parents=True)
    (packet_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packet_run_id": "fake_catalog_packet",
                "created_at": "2026-05-08T00:00:00+00:00",
                "execute": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "commands.json").write_text(
        json.dumps([{"system_id": "catalog", "command": ["run", "catalog"]}]) + "\n",
        encoding="utf-8",
    )
    (packet_dir / "results.json").write_text(
        json.dumps({"candidate_count": 1, "executed_count": 1, "dry_run_count": 0}) + "\n",
        encoding="utf-8",
    )
    (packet_dir / "candidate_summary.csv").write_text(
        "\n".join(
            [
                "system_id,tool_catalog_profile_id,execute,output_dir,comparison_path,exact_match_rate,executable_match_rate",
                "catalog,visual_role_catalog_v1,True,/tmp/catalog,/tmp/catalog/probe_comparison.json,0.125,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return packet_dir


def _write_fake_visual_hard_slice_packet(packet_dir: Path) -> Path:
    packet_dir.mkdir(parents=True)
    (packet_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packet_run_id": "fake_visual_hard_slice",
                "created_at": "2026-05-09T00:00:00+00:00",
                "execute": True,
                "case_count": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "commands.json").write_text(
        json.dumps(
            [
                {"system_id": "contracted", "command": ["run", "contracted"]},
                {"system_id": "no_directive", "command": ["run", "no_directive"]},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "results.json").write_text(
        json.dumps({"candidate_count": 2, "executed_count": 2, "dry_run_count": 0, "case_count": 8}) + "\n",
        encoding="utf-8",
    )
    (packet_dir / "candidate_summary.csv").write_text(
        "\n".join(
            [
                "system_id,execute,output_dir,exact_match_rate,executable_match_rate,hard_slice_gate",
                "contracted,True,/tmp/contracted,1.0,1.0,contracted_reference",
                "no_directive,True,/tmp/no_directive,0.0,0.0,no_directive_reference",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return packet_dir


def _write_fake_tool_probe_replay_packet(packet_dir: Path) -> Path:
    packet_dir.mkdir(parents=True)
    (packet_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packet_run_id": "fake_replay",
                "created_at": "2026-05-07T00:00:00+00:00",
                "case_ids": ["cli_invoice_lock_hyphen_query", "parallel_audit_array_literal"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "summary.json").write_text(
        json.dumps(
            {
                "case_count": 2,
                "dry_run": True,
                "failure_mode_counts": {"argument_mismatch": 1, "no_tool_call": 1},
                "family_counts": {"cli_canonicalization": 1, "parallel_tool_calling": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "commands.json").write_text(
        json.dumps(
            [
                {"case_id": "cli_invoice_lock_hyphen_query", "command": ["run", "cli"]},
                {"case_id": "parallel_audit_array_literal", "command": ["run", "parallel"]},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "replay_cases.csv").write_text(
        "\n".join(
            [
                "case_id,family,source_failure_mode,baseline_exact_match",
                "cli_invoice_lock_hyphen_query,cli_canonicalization,argument_mismatch,True",
                "parallel_audit_array_literal,parallel_tool_calling,no_tool_call,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "replay_next_actions.csv").write_text(
        "\n".join(
            [
                "case_id,family,source_failure_mode,priority,next_action,why",
                "cli_invoice_lock_hyphen_query,cli_canonicalization,argument_mismatch,medium,build_canonical_argument_replay,right tool wrong args",
                "parallel_audit_array_literal,parallel_tool_calling,no_tool_call,high,build_parallel_array_replay_or_workflow,missing array workflow",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return packet_dir


def _write_fake_tool_probe_replay_live_packet(packet_dir: Path) -> Path:
    packet_dir.mkdir(parents=True)
    (packet_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packet_run_id": "fake_live_replay",
                "created_at": "2026-05-07T00:00:00+00:00",
                "operator_surface": "rich_cli_exact_probe_replay_v1",
                "entrypoint": "moonie-agent replay-live",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "summary.json").write_text(
        json.dumps(
            {
                "case_count": 1,
                "execute": True,
                "executed_count": 1,
                "exact_count": 0,
                "exact_rate": 0.0,
                "failure_mode_counts": {"no_tool_call": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "commands.json").write_text(
        json.dumps([{"case_id": "parallel_audit_array_literal", "command": ["moonie-agent", "replay-live"]}])
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "live_case_states.csv").write_text(
        "\n".join(
            [
                "case_id,family,source_failure_mode,status,replay_failure_mode,replay_exact_match",
                "parallel_audit_array_literal,parallel_tool_calling,no_tool_call,non_exact,no_tool_call,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "live_replay_results.csv").write_text(
        "\n".join(
            [
                "case_id,family,source_failure_mode,replay_failure_mode,replay_exact_match",
                "parallel_audit_array_literal,parallel_tool_calling,no_tool_call,no_tool_call,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return packet_dir


def _write_fake_tool_probe_replay_live_comparison(packet_dir: Path) -> Path:
    packet_dir.mkdir(parents=True)
    (packet_dir / "live_replay_comparison.json").write_text(
        json.dumps(
            {
                "summary": {
                    "baseline_system_id": "mlx_gemma4_e2b_reasoner_only",
                    "candidate_system_id": "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
                    "shared_case_count": 1,
                    "baseline_exact_rate": 1.0,
                    "candidate_exact_rate": 0.0,
                    "delta_exact_rate": -1.0,
                },
                "case_deltas": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "live_replay_case_deltas.csv").write_text(
        "\n".join(
            [
                "case_id,family,baseline_replay_exact_match,candidate_replay_exact_match,delta_actual_call_count,candidate_replay_failure_mode",
                "parallel_audit_array_literal,parallel_tool_calling,True,False,-2,no_tool_call",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "live_replay_summary.md").write_text("# Fake comparison\n", encoding="utf-8")
    return packet_dir


def _write_fake_tool_probe_replay_live_diagnostic(packet_dir: Path) -> Path:
    packet_dir.mkdir(parents=True)
    (packet_dir / "visual_tool_choice_diagnostics.json").write_text(
        json.dumps(
            {
                "summary": {
                    "created_at": "2026-05-08T00:00:00+00:00",
                    "packet_count": 3,
                    "case_count": 1,
                    "diagnosis_counts": {"visual_literal_argument_mismatch": 1},
                    "case_diagnosis_transitions": {
                        "visual_latest_filter_literal": [
                            "visual_role_catalog_v1:visual_literal_argument_mismatch"
                        ]
                    },
                },
                "rows": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "visual_tool_choice_diagnostics.csv").write_text(
        "\n".join(
            [
                "packet_run_id,packet_label,system_id,case_id,family,expected_tools,actual_tools,replay_failure_mode,diagnosis,next_diagnostic",
                "catalog,visual_role_catalog_v1,candidate,visual_latest_filter_literal,visual_referent_carryover,refine_selection,refine_selection,argument_mismatch,visual_literal_argument_mismatch,preserve literal visual selector arguments after correct routing",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "visual_tool_choice_diagnostics.md").write_text("# Fake diagnostic\n", encoding="utf-8")
    return packet_dir
