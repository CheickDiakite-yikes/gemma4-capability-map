from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tool_prompt_contract_probe_packet.py"
SPEC = importlib.util.spec_from_file_location("summarize_tool_prompt_contract_probe_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_prompt_contract_probe_packet_summary_writes_candidate_gate_outputs(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    schema_dir = packet_dir / "schema" / "comparison_vs_no_directive"
    literal_dir = packet_dir / "literal" / "comparison_vs_no_directive"
    schema_dir.mkdir(parents=True)
    literal_dir.mkdir(parents=True)
    (packet_dir / "candidate_summary.csv").write_text(
        "\n".join(
            [
                "system_id,tool_prompt_contract_id,execute,output_dir,comparison_path,no_directive_comparison_path,exact_match_rate,executable_match_rate,delta_exact_vs_contracted,delta_exact_vs_no_directive,probe_gate",
                f"schema,schema_anchor_v1,True,{packet_dir / 'schema'},,{schema_dir / 'probe_comparison.json'},0.125,0.0,-0.75,0.125,probe_improved_vs_no_directive",
                f"literal,literal_argument_guard_v1,True,{packet_dir / 'literal'},,{literal_dir / 'probe_comparison.json'},0.0,1.0,-0.875,0.0,probe_improved_vs_no_directive",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (schema_dir / "probe_case_deltas.csv").write_text(
        "\n".join(
            [
                "case_id,family,baseline_exact_match,candidate_exact_match,delta_exact_match,baseline_failure_mode,candidate_failure_mode,baseline_executable_match,candidate_executable_match,delta_executable_match,baseline_actual_call_count,candidate_actual_call_count,delta_actual_call_count",
                "visual_readback,visual_referent_carryover,False,True,1,no_tool_call,exact,,,,0,1,1",
                "api_fetch,api_canonicalization,False,False,0,argument_mismatch,argument_mismatch,,,,1,1,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (literal_dir / "probe_case_deltas.csv").write_text(
        "\n".join(
            [
                "case_id,family,baseline_exact_match,candidate_exact_match,delta_exact_match,baseline_failure_mode,candidate_failure_mode,baseline_executable_match,candidate_executable_match,delta_executable_match,baseline_actual_call_count,candidate_actual_call_count,delta_actual_call_count",
                "visual_target,visual_argument_copying,False,False,0,no_tool_call,executable_paraphrase,False,True,1,0,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = SCRIPT.summarize_prompt_contract_probe_packet(packet_dir)

    rows = {row["tool_prompt_contract_id"]: row for row in payload["candidate_gate_summary"]}
    assert rows["schema_anchor_v1"]["recommendation"] == "weak_exact_gain"
    assert rows["schema_anchor_v1"]["improved_case_count"] == 1
    assert rows["literal_argument_guard_v1"]["recommendation"] == "visual_executable_gain_only"
    assert (packet_dir / "candidate_gate_summary.csv").exists()
    assert (packet_dir / "candidate_failure_mode_counts.csv").exists()
    assert "weak_exact_gain" in (packet_dir / "candidate_gate_summary.md").read_text(encoding="utf-8")
