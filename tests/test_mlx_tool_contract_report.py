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
    assert payload["manifest"]["table_count"] == 16
    assert payload["manifest"]["figure_count"] == 10

    candidates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_candidates"]}
    assert set(candidates) == {
        "schema_anchor_v1",
        "literal_argument_guard_v1",
        "tool_required_parallel_v1",
        "schema_literal_tool_required_v2",
        "visual_next_call_state_v2",
        "parallel_array_required_v2",
    }
    assert candidates["schema_anchor_v1"]["disable_tool_turn_directive"] is True
    gates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_probe_gates"]}
    assert gates["schema_anchor_v1"]["recommendation"] == "weak_exact_gain"
    assert gates["literal_argument_guard_v1"]["recommendation"] == "visual_executable_gain_only"
    wave2_gates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_wave2_probe_gates"]}
    assert wave2_gates["schema_literal_tool_required_v2"]["recommendation"] == "weak_exact_gain"
    assert wave2_gates["visual_next_call_state_v2"]["executable_match_rate"] == "1.0"
    assert wave2_gates["parallel_array_required_v2"]["probe_gate"] == "no_probe_improvement_vs_no_directive"
    promotion = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_promotion_decisions"]}
    assert promotion["schema_anchor_v1"]["promotion_decision"] == "hold_for_exact_probe_replay"
    assert promotion["visual_next_call_state_v2"]["promotion_reason"].startswith("executable recovery exists")
    assert promotion["parallel_array_required_v2"]["promotion_decision"] == "reject_for_h1_promotion"
    h1i_candidates = {row["system_id"]: row for row in payload["h1i_prompt_contract_candidate_metrics"]}
    assert h1i_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["tool_turn_directive_enabled"] == "False"
    assert h1i_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor"]["raw_planning_clean_rate_avg"] == "1.0"
    h1i_repeats = {row["system_id"]: row for row in payload["h1i_prompt_contract_repeat3_metrics"]}
    assert h1i_repeats["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["raw_planning_clean_rate_avg"] == "1.0"
    assert h1i_repeats["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required"]["controller_repair_avg"] == "0.0"
    h1j_candidates = {row["system_id"]: row for row in payload["h1j_probe_derived_candidate_metrics"]}
    assert h1j_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["real_world_readiness_avg"] == "0.9657666666666667"
    assert h1j_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor"]["raw_planning_clean_rate_avg"] == "1.0"
    h1j_helpers = {row["system_id"]: row for row in payload["h1j_probe_derived_helper_metrics"]}
    assert h1j_helpers["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair"]["strict_interface_avg"] == "1.0"
    assert h1j_helpers["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback"]["raw_planning_clean_rate_avg"] == "1.0"

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "packet_summary.csv").exists()
    assert (tmp_path / "tables" / "probe_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_candidates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_probe_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_wave2_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_wave2_probe_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_promotion_decisions.csv").exists()
    assert (tmp_path / "tables" / "h1i_prompt_contract_candidate_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1i_prompt_contract_repeat3_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1j_probe_derived_candidate_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1j_probe_derived_helper_metrics.csv").exists()
    assert (tmp_path / "figures" / "h1i_readiness_strict_recovered.svg").exists()
    assert (tmp_path / "figures" / "h1h_h1i_controller_burden.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_candidate_targets.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_probe_gate.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_wave2_probe_gate.svg").exists()
    assert (tmp_path / "figures" / "h1i_prompt_contract_repeat3_burden.svg").exists()
    assert (tmp_path / "figures" / "h1j_probe_derived_burden.svg").exists()
    assert (tmp_path / "figures" / "h1j_probe_derived_helper_burden.svg").exists()
