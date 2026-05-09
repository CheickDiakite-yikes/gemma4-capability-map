from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gemma4_capability_map.runtime.visual_hard_slice import build_visual_hard_slice_cases


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_visual_hard_slice_replay_packet.py"
SPEC = importlib.util.spec_from_file_location("build_visual_hard_slice_replay_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_visual_hard_slice_replay_packet_writes_live_replay_artifacts(tmp_path: Path) -> None:
    visual_packet = _write_visual_packet(tmp_path / "visual_packet")

    packet = SCRIPT.build_visual_hard_slice_replay_packet(
        visual_packet_dir=visual_packet,
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_replay_dry_run",
    )

    packet_dir = Path(packet["packet_dir"])
    assert packet["summary"]["case_count"] == 1
    assert packet["summary"]["failure_mode_counts"] == {"argument_mismatch": 1}
    assert packet["manifest"]["entrypoint"] == "moonie-agent replay-live"
    assert packet["manifest"]["operator_surface"] == "rich_cli_visual_hard_slice_replay_v1"
    assert packet["rows"][0]["source_exact_match"] is False
    assert packet["replay_cases"][0]["live_entrypoint_status"] == "visual_hard_slice_replay_packet_v1"
    assert packet["replay_cases"][0]["initial_state"]
    assert "messages" in packet["replay_cases"][0]
    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "summary.json").exists()
    assert (packet_dir / "commands.json").exists()
    assert (packet_dir / "replay_cases.json").exists()
    assert (packet_dir / "replay_cases.csv").exists()
    assert (packet_dir / "cases" / f"{packet['rows'][0]['case_id']}.json").exists()
    command = packet["commands"][0]["command"]
    assert command[1] == "-m"
    assert "replay-live" in command
    assert "--case-id" in command


def test_visual_hard_slice_replay_packet_can_include_exact_cases(tmp_path: Path) -> None:
    visual_packet = _write_visual_packet(tmp_path / "visual_packet")

    packet = SCRIPT.build_visual_hard_slice_replay_packet(
        visual_packet_dir=visual_packet,
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_replay_all",
        include_exact=True,
    )

    assert packet["summary"]["case_count"] == 2
    assert packet["summary"]["failure_mode_counts"] == {"argument_mismatch": 1, "exact": 1}


def test_visual_hard_slice_replay_packet_filters_by_failure_mode(tmp_path: Path) -> None:
    visual_packet = _write_visual_packet(tmp_path / "visual_packet")

    packet = SCRIPT.build_visual_hard_slice_replay_packet(
        visual_packet_dir=visual_packet,
        output_root=tmp_path / "replay_packets",
        run_group_id="visual_replay_filtered",
        failure_modes=["no_tool_call"],
    )

    assert packet["summary"]["case_count"] == 0
    assert packet["manifest"]["filters"]["failure_modes"] == ["no_tool_call"]


def _write_visual_packet(packet_dir: Path) -> Path:
    cases = build_visual_hard_slice_cases()
    exact_case = cases[0]
    mismatch_case = cases[1]
    source_system = SCRIPT.DEFAULT_SOURCE_SYSTEM_ID
    baseline_system = SCRIPT.DEFAULT_BASELINE_SYSTEM_ID
    (packet_dir / source_system).mkdir(parents=True)
    (packet_dir / baseline_system).mkdir(parents=True)
    manifest = {
        "packet_run_id": packet_dir.name,
        "case_ids": [exact_case.case_id, mismatch_case.case_id],
        "case_count": 2,
        "source": "unit_test",
    }
    source_rows = [
        _probe_row(exact_case.case_id, exact_case.family, exact=True, target_query="validation error"),
        _probe_row(mismatch_case.case_id, mismatch_case.family, exact=False, target_query="visible validation error"),
    ]
    baseline_rows = [
        _probe_row(exact_case.case_id, exact_case.family, exact=True, target_query="validation error"),
        _probe_row(mismatch_case.case_id, mismatch_case.family, exact=True, target_query="validation error"),
    ]
    (packet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (packet_dir / source_system / "probe_results.json").write_text(json.dumps(source_rows, indent=2) + "\n", encoding="utf-8")
    (packet_dir / baseline_system / "probe_results.json").write_text(json.dumps(baseline_rows, indent=2) + "\n", encoding="utf-8")
    return packet_dir


def _probe_row(case_id: str, family: str, *, exact: bool, target_query: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "family": family,
        "expected_call_count": 1,
        "actual_call_count": 1,
        "exact_match": exact,
        "executable_match": exact,
        "expected_calls": [{"name": "extract_layout", "arguments": {"image_id": "img-hard-form-decoy", "target_query": "validation error"}}],
        "actual_calls": [{"name": "extract_layout", "arguments": {"image_id": "img-hard-form-decoy", "target_query": target_query}}],
        "raw_model_output": "{}",
    }
