from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_tool_prompt_contract_probe_packet.py"
SPEC = importlib.util.spec_from_file_location("run_tool_prompt_contract_probe_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_tool_prompt_contract_probe_packet_dry_run_writes_candidate_commands(tmp_path: Path) -> None:
    packet = SCRIPT.build_tool_prompt_contract_probe_packet(
        output_root=tmp_path,
        run_group_id="candidate_probe_dry_run",
        execute=False,
    )

    packet_dir = Path(packet["packet_dir"])
    assert packet["candidate_count"] == 3
    assert packet["executed_count"] == 0
    assert packet["dry_run_count"] == 3
    assert packet["manifest"]["no_directive_probe_dir"]
    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "commands.json").exists()
    assert (packet_dir / "candidate_summary.csv").exists()

    contract_ids = {row["tool_prompt_contract_id"] for row in packet["rows"]}
    assert contract_ids == {"schema_anchor_v1", "literal_argument_guard_v1", "tool_required_parallel_v1"}
    assert all("no_directive_comparison_path" in row for row in packet["rows"])
    assert all("probe_gate" in row for row in packet["rows"])
    first_command = packet["commands"][0]["command"]
    assert "run_tool_directive_probe.py" in first_command[1]
    assert "--system-id" in first_command
    assert "--output-dir" in first_command


def test_tool_prompt_contract_probe_packet_can_select_second_wave(tmp_path: Path) -> None:
    packet = SCRIPT.build_tool_prompt_contract_probe_packet(
        output_root=tmp_path,
        run_group_id="candidate_probe_wave2_dry_run",
        candidate_wave="v2",
        execute=False,
    )

    assert packet["candidate_count"] == 3
    assert packet["manifest"]["candidate_wave"] == "v2"
    contract_ids = {row["tool_prompt_contract_id"] for row in packet["rows"]}
    assert contract_ids == {
        "schema_literal_tool_required_v2",
        "visual_next_call_state_v2",
        "parallel_array_required_v2",
    }


def test_tool_prompt_contract_probe_packet_can_select_third_wave(tmp_path: Path) -> None:
    packet = SCRIPT.build_tool_prompt_contract_probe_packet(
        output_root=tmp_path,
        run_group_id="candidate_probe_wave3_dry_run",
        candidate_wave="v3",
        execute=False,
    )

    assert packet["candidate_count"] == 3
    assert packet["manifest"]["candidate_wave"] == "v3"
    contract_ids = {row["tool_prompt_contract_id"] for row in packet["rows"]}
    assert contract_ids == {
        "canonical_json_copy_v3",
        "visual_tool_initiation_v3",
        "parallel_two_call_array_v3",
    }


def test_tool_prompt_contract_probe_packet_can_select_fourth_wave(tmp_path: Path) -> None:
    packet = SCRIPT.build_tool_prompt_contract_probe_packet(
        output_root=tmp_path,
        run_group_id="candidate_probe_wave4_dry_run",
        candidate_wave="v4",
        execute=False,
    )

    assert packet["candidate_count"] == 1
    assert packet["manifest"]["candidate_wave"] == "v4"
    contract_ids = {row["tool_prompt_contract_id"] for row in packet["rows"]}
    assert contract_ids == {"visual_state_tool_selection_v4"}


def test_tool_prompt_contract_probe_packet_validates_candidate_controls(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  bad_candidate:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    research_controls:
      tool_prompt_contract_id: schema_anchor_v1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        SCRIPT.build_tool_prompt_contract_probe_packet(
            output_root=tmp_path / "out",
            run_group_id="bad",
            registry_path=registry_path,
            system_ids=["bad_candidate"],
            execute=False,
        )
    except ValueError as exc:
        assert "must disable the exact tool-turn directive" in str(exc)
    else:
        raise AssertionError("Expected bad candidate validation to fail.")


def test_tool_prompt_contract_probe_gate_marks_improvement_over_no_directive() -> None:
    assert (
        SCRIPT._probe_gate(
            {
                "delta_exact_match_rate": 0.125,
                "baseline_executable_match_rate": 0.0,
                "candidate_executable_match_rate": 0.0,
            }
        )
        == "probe_improved_vs_no_directive"
    )
    assert (
        SCRIPT._probe_gate(
            {
                "delta_exact_match_rate": 0.0,
                "baseline_executable_match_rate": 0.0,
                "candidate_executable_match_rate": 1.0,
            }
        )
        == "probe_improved_vs_no_directive"
    )
    assert (
        SCRIPT._probe_gate(
            {
                "delta_exact_match_rate": 0.0,
                "baseline_executable_match_rate": 1.0,
                "candidate_executable_match_rate": 0.0,
            }
        )
        == "no_probe_improvement_vs_no_directive"
    )
