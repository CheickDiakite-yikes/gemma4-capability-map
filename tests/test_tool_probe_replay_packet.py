from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gemma4_capability_map.runtime.tool_directive_probe import build_tool_directive_probe_cases


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_tool_probe_replay_packet.py"
SPEC = importlib.util.spec_from_file_location("build_tool_probe_replay_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_tool_probe_replay_packet_writes_failed_case_artifacts(tmp_path: Path) -> None:
    source_probe = tmp_path / "source"
    baseline_probe = tmp_path / "baseline"
    source_probe.mkdir()
    baseline_probe.mkdir()
    cases = build_tool_directive_probe_cases()
    source_rows = [
        {
            "case_id": cases[0].case_id,
            "family": cases[0].family,
            "expected_call_count": 1,
            "actual_call_count": 1,
            "exact_match": False,
            "executable_match": None,
            "expected_calls": [{"name": "cli_search_logs", "arguments": {"path": "logs/billing.log"}}],
            "actual_calls": [{"name": "cli_search_logs", "arguments": {"path": "billing.log"}}],
            "raw_model_output": "{}",
        },
        {
            "case_id": cases[1].case_id,
            "family": cases[1].family,
            "expected_call_count": 1,
            "actual_call_count": 1,
            "exact_match": True,
            "executable_match": None,
            "expected_calls": [{"name": "cli_apply_patch", "arguments": {}}],
            "actual_calls": [{"name": "cli_apply_patch", "arguments": {}}],
            "raw_model_output": "{}",
        },
    ]
    baseline_rows = [
        {**source_rows[0], "exact_match": True},
        source_rows[1],
    ]
    _write_probe(source_probe, "source_system", source_rows)
    _write_probe(baseline_probe, "baseline_system", baseline_rows)

    packet = SCRIPT.build_tool_probe_replay_packet(
        source_probe_dir=source_probe,
        baseline_probe_dir=baseline_probe,
        output_root=tmp_path / "packets",
        run_group_id="probe_replay_dry_run",
    )

    packet_dir = Path(packet["packet_dir"])
    assert packet["summary"]["case_count"] == 1
    assert packet["summary"]["failure_mode_counts"] == {"argument_mismatch": 1}
    assert packet["summary"]["next_action_counts"] == {"build_canonical_argument_replay": 1}
    assert packet["rows"][0]["case_id"] == cases[0].case_id
    assert packet["next_actions"][0]["priority"] == "medium"
    assert packet["rows"][0]["baseline_exact_match"] is True
    assert packet["replay_cases"][0]["live_entrypoint_status"] == "probe_replay_packet_only_v1"
    assert "messages" in packet["replay_cases"][0]
    assert "tool_specs" in packet["replay_cases"][0]
    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "summary.json").exists()
    assert (packet_dir / "commands.json").exists()
    assert (packet_dir / "replay_cases.csv").exists()
    assert (packet_dir / "replay_next_actions.csv").exists()
    assert (packet_dir / "cases" / f"{cases[0].case_id}.json").exists()
    command = packet["commands"][0]["command"]
    assert "--case-id" in command
    assert cases[0].case_id in command


def test_tool_probe_replay_packet_can_include_exact_cases(tmp_path: Path) -> None:
    source_probe = tmp_path / "source"
    baseline_probe = tmp_path / "baseline"
    source_probe.mkdir()
    baseline_probe.mkdir()
    case = build_tool_directive_probe_cases()[0]
    rows = [
        {
            "case_id": case.case_id,
            "family": case.family,
            "expected_call_count": 1,
            "actual_call_count": 1,
            "exact_match": True,
            "executable_match": None,
            "expected_calls": [{"name": "cli_search_logs", "arguments": {}}],
            "actual_calls": [{"name": "cli_search_logs", "arguments": {}}],
            "raw_model_output": "{}",
        }
    ]
    _write_probe(source_probe, "source_system", rows)
    _write_probe(baseline_probe, "baseline_system", rows)

    packet = SCRIPT.build_tool_probe_replay_packet(
        source_probe_dir=source_probe,
        baseline_probe_dir=baseline_probe,
        output_root=tmp_path / "packets",
        run_group_id="probe_replay_all",
        include_exact=True,
    )

    assert packet["summary"]["case_count"] == 1
    assert packet["summary"]["failure_mode_counts"] == {"exact": 1}


def test_tool_probe_replay_packet_can_execute_selected_cases(tmp_path: Path) -> None:
    source_probe = tmp_path / "source"
    baseline_probe = tmp_path / "baseline"
    source_probe.mkdir()
    baseline_probe.mkdir()
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_probe:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    case = build_tool_directive_probe_cases()[0]
    rows = [
        {
            "case_id": case.case_id,
            "family": case.family,
            "expected_call_count": 1,
            "actual_call_count": 1,
            "exact_match": False,
            "executable_match": None,
            "expected_calls": [{"name": "cli_search_logs", "arguments": {"path": "logs/billing.log"}}],
            "actual_calls": [{"name": "cli_search_logs", "arguments": {"path": "billing.log"}}],
            "raw_model_output": "{}",
        }
    ]
    _write_probe(source_probe, "source_system", rows)
    _write_probe(baseline_probe, "baseline_system", [{**rows[0], "exact_match": True}])

    packet = SCRIPT.build_tool_probe_replay_packet(
        source_probe_dir=source_probe,
        baseline_probe_dir=baseline_probe,
        output_root=tmp_path / "packets",
        run_group_id="probe_replay_execute",
        registry_path=registry_path,
        system_id="heuristic_probe",
        execute=True,
    )

    packet_dir = Path(packet["packet_dir"])
    assert packet["summary"]["dry_run"] is False
    assert packet["summary"]["executed_count"] == 1
    assert packet["summary"]["replay_exact_match_rate"] == 1.0
    assert packet["replay_results"][0]["replay_exact_match"] is True
    assert (packet_dir / "replay_results.json").exists()
    assert (packet_dir / "replay_results.csv").exists()
    assert (packet_dir / "runs" / case.case_id / "probe_results.json").exists()


def _write_probe(path: Path, system_id: str, rows: list[dict[str, object]]) -> None:
    manifest = {
        "system_id": system_id,
        "summary": {
            "case_count": len(rows),
            "exact_match_count": sum(1 for row in rows if row["exact_match"]),
            "exact_match_rate": 0.0,
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (path / "probe_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
