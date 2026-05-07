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
    assert "no_tool_call" in output


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
    return packet_dir
