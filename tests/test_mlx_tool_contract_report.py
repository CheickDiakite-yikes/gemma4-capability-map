from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_mlx_tool_contract_report.py"
SPEC = importlib.util.spec_from_file_location("build_mlx_tool_contract_report_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_build_mlx_tool_contract_report_writes_tables_figures_and_payload(tmp_path: Path) -> None:
    payload = SCRIPT.build_report(output_dir=tmp_path)

    packet_rows = {row["packet"]: row for row in payload["packet_summary"]}
    assert packet_rows["H1h full"]["contracted_readiness"] == 0.96891
    assert packet_rows["H1h full"]["no_directive_controller_repair"] == 0.7
    assert packet_rows["H1i worst-family"]["no_repair_readiness"] == 0.64697
    assert packet_rows["H1i worst-family"]["failure_candidates"] == 12

    h1i_rows = {row["label"]: row for row in payload["h1i_system_metrics"]}
    assert h1i_rows["contracted"]["strict_interface_avg"] == 1.0
    assert h1i_rows["no directive"]["raw_planning_clean_rate_avg"] == 0.0
    assert h1i_rows["no directive + no repair"]["recovered_execution_avg"] == 0.0

    failure_modes = {
        (row["side"], row["failure_mode"]): row["case_count"] for row in payload["probe_failure_modes"]
    }
    assert failure_modes[("candidate", "argument_mismatch")] == 4
    assert failure_modes[("candidate", "no_tool_call")] == 4
    assert failure_modes[("baseline_non_exact", "executable_paraphrase")] == 1

    assert payload["gemini"]["dry_run"] is True
    assert payload["gemini"]["workflow_count"] == 10
    assert payload["manifest"]["table_count"] == 9
    assert payload["manifest"]["figure_count"] == 6

    candidates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_candidates"]}
    assert set(candidates) == {"schema_anchor_v1", "literal_argument_guard_v1", "tool_required_parallel_v1"}
    assert candidates["schema_anchor_v1"]["disable_tool_turn_directive"] is True
    gates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_probe_gates"]}
    assert gates["schema_anchor_v1"]["recommendation"] == "weak_exact_gain"
    assert gates["literal_argument_guard_v1"]["recommendation"] == "visual_executable_gain_only"

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "packet_summary.csv").exists()
    assert (tmp_path / "tables" / "probe_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_candidates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_probe_failure_modes.csv").exists()
    assert (tmp_path / "figures" / "h1i_readiness_strict_recovered.svg").exists()
    assert (tmp_path / "figures" / "h1h_h1i_controller_burden.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_candidate_targets.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_probe_gate.svg").exists()
