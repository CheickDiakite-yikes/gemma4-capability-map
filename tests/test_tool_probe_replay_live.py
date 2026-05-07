from __future__ import annotations

import json
from pathlib import Path

from gemma4_capability_map.runtime.tool_directive_probe import build_tool_directive_probe_cases
from gemma4_capability_map.runtime.tool_probe_replay_live import run_tool_probe_replay_live


def test_tool_probe_replay_live_dry_run_writes_operator_packet(tmp_path: Path) -> None:
    packet_dir = _write_replay_packet(tmp_path / "source_replay", ["parallel_audit_array_literal"])

    payload = run_tool_probe_replay_live(
        packet_dir=packet_dir,
        output_dir=tmp_path / "live_replay",
        case_ids=["parallel_audit_array_literal"],
        execute=False,
        render=False,
    )

    output_dir = Path(payload["packet_dir"])
    assert payload["summary"]["case_count"] == 1
    assert payload["summary"]["execute"] is False
    assert payload["case_states"][0]["status"] == "dry_run"
    assert payload["manifest"]["operator_surface"] == "rich_cli_exact_probe_replay_v1"
    assert payload["commands"][0]["command"][2] == "gemma4_capability_map.runtime.cli"
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "live_case_states.csv").exists()
    assert (output_dir / "commands.json").exists()


def test_tool_probe_replay_live_can_execute_heuristic_case(tmp_path: Path) -> None:
    packet_dir = _write_replay_packet(tmp_path / "source_replay", ["cli_invoice_lock_hyphen_query"])
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

    payload = run_tool_probe_replay_live(
        packet_dir=packet_dir,
        output_dir=tmp_path / "live_replay",
        system_id="heuristic_probe",
        registry_path=registry_path,
        execute=True,
        render=False,
    )

    output_dir = Path(payload["packet_dir"])
    assert payload["summary"]["execute"] is True
    assert payload["summary"]["executed_count"] == 1
    assert payload["results"][0]["replay_exact_match"] is True
    assert payload["case_states"][0]["status"] == "exact"
    assert (output_dir / "live_replay_results.json").exists()
    assert (output_dir / "runs" / "cli_invoice_lock_hyphen_query" / "probe_results.json").exists()


def _write_replay_packet(packet_dir: Path, case_ids: list[str]) -> Path:
    packet_dir.mkdir(parents=True)
    cases = {case.case_id: case for case in build_tool_directive_probe_cases()}
    rows = []
    for case_id in case_ids:
        case = cases[case_id]
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "source_failure_mode": "no_tool_call" if case.family == "parallel_tool_calling" else "argument_mismatch",
                "source_exact_match": "False",
                "source_executable_match": "",
                "baseline_exact_match": "True",
                "expected_call_count": "1",
                "source_actual_call_count": "0",
                "case_path": str(packet_dir / "cases" / f"{case_id}.json"),
            }
        )
    (packet_dir / "manifest.json").write_text(
        json.dumps({"packet_run_id": packet_dir.name, "case_ids": case_ids}) + "\n",
        encoding="utf-8",
    )
    (packet_dir / "replay_cases.csv").write_text(
        "\n".join(
            [
                "case_id,family,source_failure_mode,source_exact_match,source_executable_match,baseline_exact_match,expected_call_count,source_actual_call_count,case_path",
                *[
                    ",".join(
                        [
                            row["case_id"],
                            row["family"],
                            row["source_failure_mode"],
                            row["source_exact_match"],
                            row["source_executable_match"],
                            row["baseline_exact_match"],
                            row["expected_call_count"],
                            row["source_actual_call_count"],
                            row["case_path"],
                        ]
                    )
                    for row in rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return packet_dir
