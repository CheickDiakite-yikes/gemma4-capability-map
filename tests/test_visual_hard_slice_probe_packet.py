from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_visual_hard_slice_probe_packet.py"
SPEC = importlib.util.spec_from_file_location("run_visual_hard_slice_probe_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_visual_hard_slice_probe_packet_dry_run_writes_system_commands(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_probe_packet(
        output_root=tmp_path,
        run_group_id="visual_hard_slice_dry_run",
        execute=False,
    )

    packet_dir = Path(packet["packet_dir"])
    assert packet["candidate_count"] == len(SCRIPT.DEFAULT_SYSTEM_IDS)
    assert packet["case_count"] == 8
    assert packet["executed_count"] == 0
    assert packet["dry_run_count"] == len(SCRIPT.DEFAULT_SYSTEM_IDS)
    assert packet["manifest"]["contracted_system_id"] == SCRIPT.CONTRACTED_SYSTEM_ID
    assert packet["manifest"]["no_directive_system_id"] == SCRIPT.NO_DIRECTIVE_SYSTEM_ID
    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "commands.json").exists()
    assert (packet_dir / "candidate_summary.csv").exists()
    assert (packet_dir / "system_summary.csv").exists()
    assert (packet_dir / "candidate_gate_summary.csv").exists()
    assert (packet_dir / "candidate_gate_summary.md").exists()
    assert (packet_dir / "candidate_failure_mode_counts.csv").exists()
    assert (packet_dir / "family_summary.csv").exists()

    first_command = packet["commands"][0]["command"]
    assert "run_visual_hard_slice_probe.py" in first_command[1]
    assert "--system-id" in first_command
    assert "--case-id" in first_command


def test_visual_hard_slice_probe_packet_can_select_cases_and_systems(tmp_path: Path) -> None:
    packet = SCRIPT.build_visual_hard_slice_probe_packet(
        output_root=tmp_path,
        run_group_id="visual_hard_slice_selected_dry_run",
        system_ids=[SCRIPT.NO_DIRECTIVE_SYSTEM_ID],
        case_ids=["visual_metric_panel_vs_table_selector"],
        execute=False,
    )

    assert packet["candidate_count"] == 1
    assert packet["case_count"] == 1
    assert packet["manifest"]["case_ids"] == ["visual_metric_panel_vs_table_selector"]
    assert packet["commands"][0]["case_ids"] == ["visual_metric_panel_vs_table_selector"]


def test_visual_hard_slice_probe_packet_validates_unknown_case(tmp_path: Path) -> None:
    try:
        SCRIPT.build_visual_hard_slice_probe_packet(
            output_root=tmp_path,
            run_group_id="bad_case",
            case_ids=["missing_case"],
            execute=False,
        )
    except ValueError as exc:
        assert "Unknown visual hard-slice case id" in str(exc)
    else:
        raise AssertionError("Expected unknown hard-slice case validation to fail.")


def test_visual_hard_slice_gate_marks_reference_and_gain() -> None:
    assert (
        SCRIPT._hard_slice_gate(system_id=SCRIPT.CONTRACTED_SYSTEM_ID, comparison_vs_no_directive={})
        == "contracted_reference"
    )
    assert (
        SCRIPT._hard_slice_gate(system_id=SCRIPT.NO_DIRECTIVE_SYSTEM_ID, comparison_vs_no_directive={})
        == "no_directive_reference"
    )
    assert (
        SCRIPT._hard_slice_gate(
            system_id="candidate",
            comparison_vs_no_directive={
                "delta_exact_match_rate": 0.0,
                "baseline_executable_match_rate": 0.0,
                "candidate_executable_match_rate": 0.25,
            },
        )
        == "hard_slice_improved_vs_no_directive"
    )
