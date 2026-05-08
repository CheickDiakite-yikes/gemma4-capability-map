from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_tool_catalog_profile_probe_packet.py"
SPEC = importlib.util.spec_from_file_location("run_tool_catalog_profile_probe_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_tool_catalog_profile_probe_packet_dry_run_writes_candidate_commands(tmp_path: Path) -> None:
    packet = SCRIPT.build_tool_catalog_profile_probe_packet(
        output_root=tmp_path,
        run_group_id="catalog_profile_probe_dry_run",
        execute=False,
    )

    packet_dir = Path(packet["packet_dir"])
    assert packet["candidate_count"] == 1
    assert packet["executed_count"] == 0
    assert packet["dry_run_count"] == 1
    assert packet["manifest"]["candidate_wave"] == "v1"
    assert packet["manifest"]["no_directive_probe_dir"]
    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "commands.json").exists()
    assert (packet_dir / "candidate_summary.csv").exists()

    row = packet["rows"][0]
    assert row["tool_catalog_profile_id"] == "visual_role_catalog_v1"
    assert row["probe_gate"] == ""
    first_command = packet["commands"][0]["command"]
    assert "run_tool_directive_probe.py" in first_command[1]
    assert "--system-id" in first_command
    assert "--output-dir" in first_command


def test_tool_catalog_profile_probe_packet_wave_two_targets_argument_hints(tmp_path: Path) -> None:
    packet = SCRIPT.build_tool_catalog_profile_probe_packet(
        output_root=tmp_path,
        run_group_id="catalog_profile_probe_v2_dry_run",
        candidate_wave="v2",
        execute=False,
    )

    row = packet["rows"][0]
    assert packet["candidate_count"] == 1
    assert packet["manifest"]["candidate_wave"] == "v2"
    assert row["system_id"] == "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints"
    assert row["tool_catalog_profile_id"] == "visual_role_catalog_argument_hints_v2"


def test_tool_catalog_profile_probe_packet_validates_candidate_controls(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  bad_candidate:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    research_controls:
      disable_tool_turn_directive: true
      tool_prompt_contract_id: visual_tool_initiation_v3
      tool_catalog_profile_id: visual_role_catalog_v1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        SCRIPT.build_tool_catalog_profile_probe_packet(
            output_root=tmp_path / "out",
            run_group_id="bad",
            registry_path=registry_path,
            system_ids=["bad_candidate"],
            execute=False,
        )
    except ValueError as exc:
        assert "must not set tool_prompt_contract_id" in str(exc)
    else:
        raise AssertionError("Expected mixed prompt-contract/catalog candidate validation to fail.")


def test_tool_catalog_profile_probe_gate_marks_improvement_over_no_directive() -> None:
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
