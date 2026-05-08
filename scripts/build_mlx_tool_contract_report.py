from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry
from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.tools.prompt_contracts import TOOL_PROMPT_CONTRACTS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "mlx_tool_contract_harnessing"
DEFAULT_H1F_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1"
)
DEFAULT_H1H_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1"
)
DEFAULT_H1I_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1"
)
DEFAULT_PROBE_COMPARISON = (
    ROOT / "results" / "tool_directive_probe" / "20260507T_mlx_no_directive_probe_v1" / "probe_comparison.json"
)
DEFAULT_GEMINI_PACKET = ROOT / "results" / "gemini_cli" / "20260507T_h1h_gemini_cli_dry_run_baseline_v1"
DEFAULT_PROMPT_CONTRACT_PACKET = (
    ROOT / "results" / "tool_prompt_contract_probe_packets" / "20260507T_prompt_contract_candidates_execute_v1"
)
DEFAULT_PROMPT_CONTRACT_WAVE2_PACKET = (
    ROOT / "results" / "tool_prompt_contract_probe_packets" / "20260507T_prompt_contract_wave2_execute_v1"
)
DEFAULT_PROMPT_CONTRACT_WAVE3_PACKET = (
    ROOT / "results" / "tool_prompt_contract_probe_packets" / "20260507T_prompt_contract_wave3_execute_v1"
)
DEFAULT_PROMPT_CONTRACT_WAVE4_PACKET = (
    ROOT / "results" / "tool_prompt_contract_probe_packets" / "20260508T_prompt_contract_wave4_execute_v1"
)
DEFAULT_H1I_PROMPT_CONTRACT_PACKET = (
    ROOT / "results" / "knowledge_work_h1_slice" / "20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet"
)
DEFAULT_H1I_PROMPT_CONTRACT_REPEAT_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet"
)
DEFAULT_H1J_PROMPT_CONTRACT_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet"
)
DEFAULT_H1J_HELPER_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet"
)
DEFAULT_H1K_PARALLEL_AUDIT_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet"
)
DEFAULT_H1K_PARALLEL_AUDIT_HELPER_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet"
)
DEFAULT_EXACT_REPLAY_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_comparisons"
    / "20260507T_contracted_vs_no_directive_exact_replay_v1"
)
DEFAULT_VISUAL_REPLAY_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_comparisons"
    / "20260507T_visual_state_contracted_vs_no_directive_v1"
)
DEFAULT_PARALLEL_REPLAY_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_comparisons"
    / "20260507T_parallel_array_contracted_vs_no_directive_v1"
)
DEFAULT_CANONICAL_ARGUMENT_REPLAY_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_comparisons"
    / "20260507T_canonical_argument_contracted_vs_no_directive_v1"
)
DEFAULT_LIVE_PARALLEL_REPLAY_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260507T_parallel_array_contracted_vs_no_directive_live_v1"
)
DEFAULT_LIVE_VISUAL_REPLAY_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260507T_visual_state_contracted_vs_no_directive_live_v1"
)
DEFAULT_LIVE_CANONICAL_REPLAY_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260507T_canonical_argument_contracted_vs_no_directive_live_v1"
)
DEFAULT_WAVE3_LIVE_CANONICAL_VS_NO_DIRECTIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260507T_canonical_argument_canonical_json_copy_vs_no_directive_live_v1"
)
DEFAULT_WAVE3_LIVE_CANONICAL_VS_CONTRACTED_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260507T_canonical_argument_contracted_vs_canonical_json_copy_live_v1"
)
DEFAULT_WAVE3_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260507T_visual_state_visual_tool_initiation_vs_no_directive_live_v1"
)
DEFAULT_WAVE3_LIVE_VISUAL_VS_CONTRACTED_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260507T_visual_state_contracted_vs_visual_tool_initiation_live_v1"
)
DEFAULT_WAVE4_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260508T_visual_state_tool_selection_vs_no_directive_live_v1"
)
DEFAULT_WAVE4_LIVE_VISUAL_VS_CONTRACTED_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260508T_visual_state_contracted_vs_tool_selection_live_v1"
)

SYSTEM_LABELS = {
    "mlx_gemma4_e2b_reasoner_only": "contracted",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive": "no directive",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair": "no directive + no repair",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback": "no directive + no fallback",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair": "no directive + no arg repair",
}


def build_report(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    h1f_packet: str | Path = DEFAULT_H1F_PACKET,
    h1h_packet: str | Path = DEFAULT_H1H_PACKET,
    h1i_packet: str | Path = DEFAULT_H1I_PACKET,
    probe_comparison_path: str | Path = DEFAULT_PROBE_COMPARISON,
    gemini_packet: str | Path = DEFAULT_GEMINI_PACKET,
    prompt_contract_packet: str | Path = DEFAULT_PROMPT_CONTRACT_PACKET,
    prompt_contract_wave2_packet: str | Path = DEFAULT_PROMPT_CONTRACT_WAVE2_PACKET,
    prompt_contract_wave3_packet: str | Path = DEFAULT_PROMPT_CONTRACT_WAVE3_PACKET,
    prompt_contract_wave4_packet: str | Path = DEFAULT_PROMPT_CONTRACT_WAVE4_PACKET,
    h1i_prompt_contract_packet: str | Path = DEFAULT_H1I_PROMPT_CONTRACT_PACKET,
    h1i_prompt_contract_repeat_packet: str | Path = DEFAULT_H1I_PROMPT_CONTRACT_REPEAT_PACKET,
    h1j_prompt_contract_packet: str | Path = DEFAULT_H1J_PROMPT_CONTRACT_PACKET,
    h1j_helper_packet: str | Path = DEFAULT_H1J_HELPER_PACKET,
    h1k_parallel_audit_packet: str | Path = DEFAULT_H1K_PARALLEL_AUDIT_PACKET,
    h1k_parallel_audit_helper_packet: str | Path = DEFAULT_H1K_PARALLEL_AUDIT_HELPER_PACKET,
    exact_replay_comparison: str | Path = DEFAULT_EXACT_REPLAY_COMPARISON,
    visual_replay_comparison: str | Path = DEFAULT_VISUAL_REPLAY_COMPARISON,
    parallel_replay_comparison: str | Path = DEFAULT_PARALLEL_REPLAY_COMPARISON,
    canonical_argument_replay_comparison: str | Path = DEFAULT_CANONICAL_ARGUMENT_REPLAY_COMPARISON,
    live_parallel_replay_comparison: str | Path = DEFAULT_LIVE_PARALLEL_REPLAY_COMPARISON,
    live_visual_replay_comparison: str | Path = DEFAULT_LIVE_VISUAL_REPLAY_COMPARISON,
    live_canonical_replay_comparison: str | Path = DEFAULT_LIVE_CANONICAL_REPLAY_COMPARISON,
    wave3_live_canonical_vs_no_directive_comparison: str | Path = DEFAULT_WAVE3_LIVE_CANONICAL_VS_NO_DIRECTIVE_COMPARISON,
    wave3_live_canonical_vs_contracted_comparison: str | Path = DEFAULT_WAVE3_LIVE_CANONICAL_VS_CONTRACTED_COMPARISON,
    wave3_live_visual_vs_no_directive_comparison: str | Path = DEFAULT_WAVE3_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON,
    wave3_live_visual_vs_contracted_comparison: str | Path = DEFAULT_WAVE3_LIVE_VISUAL_VS_CONTRACTED_COMPARISON,
    wave4_live_visual_vs_no_directive_comparison: str | Path = DEFAULT_WAVE4_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON,
    wave4_live_visual_vs_contracted_comparison: str | Path = DEFAULT_WAVE4_LIVE_VISUAL_VS_CONTRACTED_COMPARISON,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    target = Path(output_dir)
    tables_dir = target / "tables"
    figures_dir = target / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packets = [
        _packet_payload("H1f compact", Path(h1f_packet)),
        _packet_payload("H1h full", Path(h1h_packet)),
        _packet_payload("H1i worst-family", Path(h1i_packet)),
    ]
    probe = json.loads(Path(probe_comparison_path).read_text(encoding="utf-8"))
    gemini_manifest = json.loads((Path(gemini_packet) / "manifest.json").read_text(encoding="utf-8"))
    registry = load_model_registry(registry_path)

    packet_rows = [_packet_summary_row(packet) for packet in packets]
    h1i_system_rows = _system_metric_rows(packets[-1]["tool_contract"]["system_rows"])
    probe_case_rows = probe["case_deltas"]
    probe_failure_rows = _probe_failure_rows(probe_case_rows)
    h1i_failure_rows = _csv_rows(Path(h1i_packet) / "trace_failure_mode_counts.csv")
    h1i_workflow_failures = _csv_rows(Path(h1i_packet) / "workflow_family_failures.csv")
    candidate_rows = _prompt_contract_candidate_rows(registry)
    prompt_contract_gate_rows = _csv_rows(Path(prompt_contract_packet) / "candidate_gate_summary.csv")
    prompt_contract_failure_rows = _csv_rows(Path(prompt_contract_packet) / "candidate_failure_mode_counts.csv")
    prompt_contract_wave2_gate_rows = _csv_rows(Path(prompt_contract_wave2_packet) / "candidate_gate_summary.csv")
    prompt_contract_wave2_failure_rows = _csv_rows(Path(prompt_contract_wave2_packet) / "candidate_failure_mode_counts.csv")
    prompt_contract_wave3_gate_rows = _csv_rows(Path(prompt_contract_wave3_packet) / "candidate_gate_summary.csv")
    prompt_contract_wave3_failure_rows = _csv_rows(Path(prompt_contract_wave3_packet) / "candidate_failure_mode_counts.csv")
    prompt_contract_wave4_gate_rows = _csv_rows(Path(prompt_contract_wave4_packet) / "candidate_gate_summary.csv")
    prompt_contract_wave4_failure_rows = _csv_rows(Path(prompt_contract_wave4_packet) / "candidate_failure_mode_counts.csv")
    prompt_contract_promotion_rows = _prompt_contract_promotion_rows(
        wave1_rows=prompt_contract_gate_rows,
        wave2_rows=prompt_contract_wave2_gate_rows,
        wave3_rows=prompt_contract_wave3_gate_rows,
        wave4_rows=prompt_contract_wave4_gate_rows,
    )
    h1i_prompt_contract_rows = _csv_rows(Path(h1i_prompt_contract_packet) / "tool_contract_system_deltas.csv")
    h1i_prompt_contract_repeat_rows = _csv_rows(
        Path(h1i_prompt_contract_repeat_packet) / "tool_contract_system_deltas.csv"
    )
    h1j_prompt_contract_rows = _csv_rows(Path(h1j_prompt_contract_packet) / "tool_contract_system_deltas.csv")
    h1j_helper_rows = _csv_rows(Path(h1j_helper_packet) / "tool_contract_system_deltas.csv")
    h1k_parallel_audit_rows = _csv_rows(Path(h1k_parallel_audit_packet) / "tool_contract_system_deltas.csv")
    h1k_parallel_audit_helper_rows = _csv_rows(
        Path(h1k_parallel_audit_helper_packet) / "tool_contract_system_deltas.csv"
    )
    exact_replay_comparison_payload = json.loads(
        (Path(exact_replay_comparison) / "replay_comparison.json").read_text(encoding="utf-8")
    )
    exact_replay_case_rows = _csv_rows(Path(exact_replay_comparison) / "replay_case_deltas.csv")
    exact_replay_family_rows = _csv_rows(Path(exact_replay_comparison) / "replay_family_deltas.csv")
    visual_replay_comparison_payload = json.loads(
        (Path(visual_replay_comparison) / "replay_comparison.json").read_text(encoding="utf-8")
    )
    parallel_replay_comparison_payload = json.loads(
        (Path(parallel_replay_comparison) / "replay_comparison.json").read_text(encoding="utf-8")
    )
    canonical_argument_replay_comparison_payload = json.loads(
        (Path(canonical_argument_replay_comparison) / "replay_comparison.json").read_text(encoding="utf-8")
    )
    exact_replay_focus_rows = _replay_focus_summary_rows(
        [
            ("all failures", exact_replay_comparison_payload),
            ("canonical arguments", canonical_argument_replay_comparison_payload),
            ("visual no-call", visual_replay_comparison_payload),
            ("parallel array", parallel_replay_comparison_payload),
        ]
    )
    live_parallel_replay_comparison_payload = json.loads(
        (Path(live_parallel_replay_comparison) / "live_replay_comparison.json").read_text(encoding="utf-8")
    )
    live_parallel_replay_case_rows = _csv_rows(Path(live_parallel_replay_comparison) / "live_replay_case_deltas.csv")
    live_visual_replay_comparison_payload = json.loads(
        (Path(live_visual_replay_comparison) / "live_replay_comparison.json").read_text(encoding="utf-8")
    )
    live_visual_replay_case_rows = _csv_rows(Path(live_visual_replay_comparison) / "live_replay_case_deltas.csv")
    live_canonical_replay_comparison_payload = json.loads(
        (Path(live_canonical_replay_comparison) / "live_replay_comparison.json").read_text(encoding="utf-8")
    )
    live_canonical_replay_case_rows = _csv_rows(Path(live_canonical_replay_comparison) / "live_replay_case_deltas.csv")
    live_replay_focus_rows = _live_replay_focus_rows(
        [
            ("canonical arguments", live_canonical_replay_comparison_payload),
            ("parallel array", live_parallel_replay_comparison_payload),
            ("visual no-call", live_visual_replay_comparison_payload),
        ]
    )
    wave3_live_comparisons = [
        (
            "canonical JSON vs no directive",
            json.loads(
                (Path(wave3_live_canonical_vs_no_directive_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "canonical JSON vs contracted",
            json.loads(
                (Path(wave3_live_canonical_vs_contracted_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "visual initiation vs no directive",
            json.loads(
                (Path(wave3_live_visual_vs_no_directive_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "visual initiation vs contracted",
            json.loads(
                (Path(wave3_live_visual_vs_contracted_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
    ]
    wave3_live_summary_rows = _live_candidate_summary_rows(wave3_live_comparisons)
    wave3_live_case_rows = _live_candidate_case_rows(wave3_live_comparisons)
    wave4_live_comparisons = [
        (
            "visual state tool selection vs no directive",
            json.loads(
                (Path(wave4_live_visual_vs_no_directive_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "visual state tool selection vs contracted",
            json.loads(
                (Path(wave4_live_visual_vs_contracted_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
    ]
    wave4_live_summary_rows = _live_candidate_summary_rows(wave4_live_comparisons)
    wave4_live_case_rows = _live_candidate_case_rows(wave4_live_comparisons)

    _write_csv(tables_dir / "packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1i_system_metrics.csv", h1i_system_rows)
    _write_csv(tables_dir / "probe_case_deltas.csv", probe_case_rows)
    _write_csv(tables_dir / "probe_failure_modes.csv", probe_failure_rows)
    _write_csv(tables_dir / "h1i_failure_modes.csv", h1i_failure_rows)
    _write_csv(tables_dir / "h1i_workflow_failures.csv", h1i_workflow_failures)
    _write_csv(tables_dir / "prompt_contract_candidates.csv", candidate_rows)
    _write_csv(tables_dir / "prompt_contract_probe_gates.csv", prompt_contract_gate_rows)
    _write_csv(tables_dir / "prompt_contract_probe_failure_modes.csv", prompt_contract_failure_rows)
    _write_csv(tables_dir / "prompt_contract_wave2_probe_gates.csv", prompt_contract_wave2_gate_rows)
    _write_csv(tables_dir / "prompt_contract_wave2_probe_failure_modes.csv", prompt_contract_wave2_failure_rows)
    _write_csv(tables_dir / "prompt_contract_wave3_probe_gates.csv", prompt_contract_wave3_gate_rows)
    _write_csv(tables_dir / "prompt_contract_wave3_probe_failure_modes.csv", prompt_contract_wave3_failure_rows)
    _write_csv(tables_dir / "prompt_contract_wave4_probe_gates.csv", prompt_contract_wave4_gate_rows)
    _write_csv(tables_dir / "prompt_contract_wave4_probe_failure_modes.csv", prompt_contract_wave4_failure_rows)
    _write_csv(tables_dir / "prompt_contract_promotion_decisions.csv", prompt_contract_promotion_rows)
    _write_csv(tables_dir / "h1i_prompt_contract_candidate_metrics.csv", h1i_prompt_contract_rows)
    _write_csv(tables_dir / "h1i_prompt_contract_repeat3_metrics.csv", h1i_prompt_contract_repeat_rows)
    _write_csv(tables_dir / "h1j_probe_derived_candidate_metrics.csv", h1j_prompt_contract_rows)
    _write_csv(tables_dir / "h1j_probe_derived_helper_metrics.csv", h1j_helper_rows)
    _write_csv(tables_dir / "h1k_parallel_audit_candidate_metrics.csv", h1k_parallel_audit_rows)
    _write_csv(tables_dir / "h1k_parallel_audit_helper_metrics.csv", h1k_parallel_audit_helper_rows)
    _write_csv(tables_dir / "exact_probe_replay_case_deltas.csv", exact_replay_case_rows)
    _write_csv(tables_dir / "exact_probe_replay_family_deltas.csv", exact_replay_family_rows)
    _write_csv(tables_dir / "exact_probe_replay_focus_summary.csv", exact_replay_focus_rows)
    _write_csv(tables_dir / "live_parallel_replay_case_deltas.csv", live_parallel_replay_case_rows)
    _write_csv(tables_dir / "live_visual_replay_case_deltas.csv", live_visual_replay_case_rows)
    _write_csv(tables_dir / "live_canonical_replay_case_deltas.csv", live_canonical_replay_case_rows)
    _write_csv(tables_dir / "wave3_live_candidate_replay_summary.csv", wave3_live_summary_rows)
    _write_csv(tables_dir / "wave3_live_candidate_case_deltas.csv", wave3_live_case_rows)
    _write_csv(tables_dir / "wave4_live_candidate_replay_summary.csv", wave4_live_summary_rows)
    _write_csv(tables_dir / "wave4_live_candidate_case_deltas.csv", wave4_live_case_rows)

    _write_grouped_metric_svg(
        figures_dir / "h1i_readiness_strict_recovered.svg",
        title="H1i readiness vs interface recovery",
        rows=h1i_system_rows,
        label_field="label",
        metrics=[
            ("real_world_readiness_avg", "readiness", "#2563EB"),
            ("strict_interface_avg", "strict", "#059669"),
            ("recovered_execution_avg", "recovered", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "h1h_h1i_controller_burden.svg",
        title="No-directive controller burden: H1h vs H1i",
        rows=[
            _burden_row("H1h full", packets[1]),
            _burden_row("H1i worst-family", packets[2]),
        ],
        label_field="packet",
        metrics=[
            ("controller_repair_avg", "repair", "#7C3AED"),
            ("controller_fallback_avg", "fallback", "#DC2626"),
            ("argument_repair_avg", "arg repair", "#0891B2"),
            ("raw_planning_clean_rate_avg", "raw clean", "#16A34A"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "tool_probe_contract_gap.svg",
        title="Tool probe contract gap",
        rows=[
            {
                "label": "contracted",
                "exact_match_rate": probe["baseline_exact_match_rate"],
                "executable_match_rate": probe["baseline_executable_match_rate"],
            },
            {
                "label": "no directive",
                "exact_match_rate": probe["candidate_exact_match_rate"],
                "executable_match_rate": probe["candidate_executable_match_rate"],
            },
        ],
        label_field="label",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
        ],
    )
    _write_bar_svg(
        figures_dir / "h1i_failure_modes.svg",
        title="H1i failure candidate modes",
        rows=[{"label": row["failure_mode"], "value": int(row["count"])} for row in h1i_failure_rows],
        color="#B91C1C",
    )
    _write_bar_svg(
        figures_dir / "prompt_contract_candidate_targets.svg",
        title="Prompt contract candidate target tags",
        rows=_candidate_tag_rows(candidate_rows),
        color="#0F766E",
    )
    _write_grouped_metric_svg(
        figures_dir / "prompt_contract_probe_gate.svg",
        title="Executed prompt contract probe gate",
        rows=prompt_contract_gate_rows,
        label_field="tool_prompt_contract_id",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
            ("delta_exact_vs_no_directive", "delta exact", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "prompt_contract_wave2_probe_gate.svg",
        title="Prompt contract wave two probe gate",
        rows=prompt_contract_wave2_gate_rows,
        label_field="tool_prompt_contract_id",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
            ("delta_exact_vs_no_directive", "delta exact", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "prompt_contract_wave3_probe_gate.svg",
        title="Prompt contract wave three probe gate",
        rows=prompt_contract_wave3_gate_rows,
        label_field="tool_prompt_contract_id",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
            ("delta_exact_vs_no_directive", "delta exact", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "prompt_contract_wave4_probe_gate.svg",
        title="Prompt contract wave four probe gate",
        rows=prompt_contract_wave4_gate_rows,
        label_field="tool_prompt_contract_id",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
            ("delta_exact_vs_no_directive", "delta exact", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "h1i_prompt_contract_repeat3_burden.svg",
        title="H1i prompt-contract repeat3 burden",
        rows=_label_system_rows(h1i_prompt_contract_repeat_rows),
        label_field="label",
        metrics=[
            ("controller_repair_avg", "repair", "#7C3AED"),
            ("controller_fallback_avg", "fallback", "#DC2626"),
            ("argument_repair_avg", "arg repair", "#0891B2"),
            ("raw_planning_clean_rate_avg", "raw clean", "#16A34A"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "h1j_probe_derived_burden.svg",
        title="H1j probe-derived candidate burden",
        rows=_label_system_rows(h1j_prompt_contract_rows),
        label_field="label",
        metrics=[
            ("controller_repair_avg", "repair", "#7C3AED"),
            ("controller_fallback_avg", "fallback", "#DC2626"),
            ("argument_repair_avg", "arg repair", "#0891B2"),
            ("raw_planning_clean_rate_avg", "raw clean", "#16A34A"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "h1j_probe_derived_helper_burden.svg",
        title="H1j probe-derived helper burden",
        rows=_label_system_rows(h1j_helper_rows),
        label_field="label",
        metrics=[
            ("controller_repair_avg", "repair", "#7C3AED"),
            ("controller_fallback_avg", "fallback", "#DC2626"),
            ("argument_repair_avg", "arg repair", "#0891B2"),
            ("raw_planning_clean_rate_avg", "raw clean", "#16A34A"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "h1k_parallel_audit_burden.svg",
        title="H1k parallel-audit candidate burden",
        rows=_label_system_rows(h1k_parallel_audit_rows),
        label_field="label",
        metrics=[
            ("controller_repair_avg", "repair", "#7C3AED"),
            ("controller_fallback_avg", "fallback", "#DC2626"),
            ("argument_repair_avg", "arg repair", "#0891B2"),
            ("raw_planning_clean_rate_avg", "raw clean", "#16A34A"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "h1k_parallel_audit_helper_burden.svg",
        title="H1k parallel-audit helper burden",
        rows=_label_system_rows(h1k_parallel_audit_helper_rows),
        label_field="label",
        metrics=[
            ("controller_repair_avg", "repair", "#7C3AED"),
            ("controller_fallback_avg", "fallback", "#DC2626"),
            ("argument_repair_avg", "arg repair", "#0891B2"),
            ("raw_planning_clean_rate_avg", "raw clean", "#16A34A"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "exact_probe_replay_gap.svg",
        title="Exact probe replay gap",
        rows=_exact_replay_gap_rows(exact_replay_comparison_payload["summary"]),
        label_field="label",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "exact_probe_replay_focus_gap.svg",
        title="Focused exact replay gaps",
        rows=exact_replay_focus_rows,
        label_field="slice",
        metrics=[
            ("baseline_exact_match_rate", "contracted", "#2563EB"),
            ("candidate_exact_match_rate", "no directive", "#DC2626"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "live_parallel_replay_gap.svg",
        title="CLI-live parallel replay gap",
        rows=_live_replay_gap_rows(live_parallel_replay_comparison_payload["summary"]),
        label_field="label",
        metrics=[
            ("exact_rate", "exact", "#2563EB"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "live_replay_focus_gap.svg",
        title="CLI-live focused replay gaps",
        rows=live_replay_focus_rows,
        label_field="slice",
        metrics=[
            ("baseline_exact_rate", "contracted", "#2563EB"),
            ("candidate_exact_rate", "no directive", "#DC2626"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "wave3_live_candidate_replay_gate.svg",
        title="Wave three live replay gate",
        rows=wave3_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executable_rate", "candidate executable", "#059669"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "wave4_live_candidate_replay_gate.svg",
        title="Wave four live replay gate",
        rows=wave4_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executable_rate", "candidate executable", "#059669"),
        ],
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(target.resolve()),
        "source_packets": {
            packet["name"]: str(packet["packet_dir"]) for packet in packets
        },
        "probe_comparison": str(Path(probe_comparison_path).resolve()),
        "gemini_packet": str(Path(gemini_packet).resolve()),
        "prompt_contract_packet": str(Path(prompt_contract_packet).resolve()),
        "prompt_contract_wave2_packet": str(Path(prompt_contract_wave2_packet).resolve()),
        "prompt_contract_wave3_packet": str(Path(prompt_contract_wave3_packet).resolve()),
        "prompt_contract_wave4_packet": str(Path(prompt_contract_wave4_packet).resolve()),
        "h1i_prompt_contract_packet": str(Path(h1i_prompt_contract_packet).resolve()),
        "h1i_prompt_contract_repeat_packet": str(Path(h1i_prompt_contract_repeat_packet).resolve()),
        "h1j_prompt_contract_packet": str(Path(h1j_prompt_contract_packet).resolve()),
        "h1j_helper_packet": str(Path(h1j_helper_packet).resolve()),
        "h1k_parallel_audit_packet": str(Path(h1k_parallel_audit_packet).resolve()),
        "h1k_parallel_audit_helper_packet": str(Path(h1k_parallel_audit_helper_packet).resolve()),
        "exact_replay_comparison": str(Path(exact_replay_comparison).resolve()),
        "visual_replay_comparison": str(Path(visual_replay_comparison).resolve()),
        "parallel_replay_comparison": str(Path(parallel_replay_comparison).resolve()),
        "canonical_argument_replay_comparison": str(Path(canonical_argument_replay_comparison).resolve()),
        "live_parallel_replay_comparison": str(Path(live_parallel_replay_comparison).resolve()),
        "live_visual_replay_comparison": str(Path(live_visual_replay_comparison).resolve()),
        "live_canonical_replay_comparison": str(Path(live_canonical_replay_comparison).resolve()),
        "wave3_live_canonical_vs_no_directive_comparison": str(
            Path(wave3_live_canonical_vs_no_directive_comparison).resolve()
        ),
        "wave3_live_canonical_vs_contracted_comparison": str(
            Path(wave3_live_canonical_vs_contracted_comparison).resolve()
        ),
        "wave3_live_visual_vs_no_directive_comparison": str(
            Path(wave3_live_visual_vs_no_directive_comparison).resolve()
        ),
        "wave3_live_visual_vs_contracted_comparison": str(
            Path(wave3_live_visual_vs_contracted_comparison).resolve()
        ),
        "wave4_live_visual_vs_no_directive_comparison": str(
            Path(wave4_live_visual_vs_no_directive_comparison).resolve()
        ),
        "wave4_live_visual_vs_contracted_comparison": str(
            Path(wave4_live_visual_vs_contracted_comparison).resolve()
        ),
        "registry_path": str(Path(registry_path).resolve()),
        "table_count": 32,
        "figure_count": 20,
    }
    report_payload = {
        "manifest": manifest,
        "packet_summary": packet_rows,
        "h1i_system_metrics": h1i_system_rows,
        "probe_failure_modes": probe_failure_rows,
        "prompt_contract_candidates": candidate_rows,
        "prompt_contract_probe_gates": prompt_contract_gate_rows,
        "prompt_contract_probe_failure_modes": prompt_contract_failure_rows,
        "prompt_contract_wave2_probe_gates": prompt_contract_wave2_gate_rows,
        "prompt_contract_wave2_probe_failure_modes": prompt_contract_wave2_failure_rows,
        "prompt_contract_wave3_probe_gates": prompt_contract_wave3_gate_rows,
        "prompt_contract_wave3_probe_failure_modes": prompt_contract_wave3_failure_rows,
        "prompt_contract_wave4_probe_gates": prompt_contract_wave4_gate_rows,
        "prompt_contract_wave4_probe_failure_modes": prompt_contract_wave4_failure_rows,
        "prompt_contract_promotion_decisions": prompt_contract_promotion_rows,
        "h1i_prompt_contract_candidate_metrics": h1i_prompt_contract_rows,
        "h1i_prompt_contract_repeat3_metrics": h1i_prompt_contract_repeat_rows,
        "h1j_probe_derived_candidate_metrics": h1j_prompt_contract_rows,
        "h1j_probe_derived_helper_metrics": h1j_helper_rows,
        "h1k_parallel_audit_candidate_metrics": h1k_parallel_audit_rows,
        "h1k_parallel_audit_helper_metrics": h1k_parallel_audit_helper_rows,
        "exact_probe_replay_comparison": exact_replay_comparison_payload,
        "exact_probe_replay_case_deltas": exact_replay_case_rows,
        "exact_probe_replay_family_deltas": exact_replay_family_rows,
        "exact_probe_replay_focus_summary": exact_replay_focus_rows,
        "live_parallel_replay_comparison": live_parallel_replay_comparison_payload,
        "live_parallel_replay_case_deltas": live_parallel_replay_case_rows,
        "live_visual_replay_comparison": live_visual_replay_comparison_payload,
        "live_visual_replay_case_deltas": live_visual_replay_case_rows,
        "live_canonical_replay_comparison": live_canonical_replay_comparison_payload,
        "live_canonical_replay_case_deltas": live_canonical_replay_case_rows,
        "live_replay_focus_summary": live_replay_focus_rows,
        "wave3_live_candidate_replay_summary": wave3_live_summary_rows,
        "wave3_live_candidate_case_deltas": wave3_live_case_rows,
        "wave4_live_candidate_replay_summary": wave4_live_summary_rows,
        "wave4_live_candidate_case_deltas": wave4_live_case_rows,
        "gemini": gemini_manifest,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "report.json").write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the current MLX tool-contract research report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--h1f-packet", default=str(DEFAULT_H1F_PACKET))
    parser.add_argument("--h1h-packet", default=str(DEFAULT_H1H_PACKET))
    parser.add_argument("--h1i-packet", default=str(DEFAULT_H1I_PACKET))
    parser.add_argument("--probe-comparison", default=str(DEFAULT_PROBE_COMPARISON))
    parser.add_argument("--gemini-packet", default=str(DEFAULT_GEMINI_PACKET))
    parser.add_argument("--prompt-contract-packet", default=str(DEFAULT_PROMPT_CONTRACT_PACKET))
    parser.add_argument("--prompt-contract-wave2-packet", default=str(DEFAULT_PROMPT_CONTRACT_WAVE2_PACKET))
    parser.add_argument("--prompt-contract-wave3-packet", default=str(DEFAULT_PROMPT_CONTRACT_WAVE3_PACKET))
    parser.add_argument("--prompt-contract-wave4-packet", default=str(DEFAULT_PROMPT_CONTRACT_WAVE4_PACKET))
    parser.add_argument("--h1i-prompt-contract-packet", default=str(DEFAULT_H1I_PROMPT_CONTRACT_PACKET))
    parser.add_argument("--h1i-prompt-contract-repeat-packet", default=str(DEFAULT_H1I_PROMPT_CONTRACT_REPEAT_PACKET))
    parser.add_argument("--h1j-prompt-contract-packet", default=str(DEFAULT_H1J_PROMPT_CONTRACT_PACKET))
    parser.add_argument("--h1j-helper-packet", default=str(DEFAULT_H1J_HELPER_PACKET))
    parser.add_argument("--h1k-parallel-audit-packet", default=str(DEFAULT_H1K_PARALLEL_AUDIT_PACKET))
    parser.add_argument(
        "--h1k-parallel-audit-helper-packet",
        default=str(DEFAULT_H1K_PARALLEL_AUDIT_HELPER_PACKET),
    )
    parser.add_argument("--exact-replay-comparison", default=str(DEFAULT_EXACT_REPLAY_COMPARISON))
    parser.add_argument("--visual-replay-comparison", default=str(DEFAULT_VISUAL_REPLAY_COMPARISON))
    parser.add_argument("--parallel-replay-comparison", default=str(DEFAULT_PARALLEL_REPLAY_COMPARISON))
    parser.add_argument("--canonical-argument-replay-comparison", default=str(DEFAULT_CANONICAL_ARGUMENT_REPLAY_COMPARISON))
    parser.add_argument("--live-parallel-replay-comparison", default=str(DEFAULT_LIVE_PARALLEL_REPLAY_COMPARISON))
    parser.add_argument("--live-visual-replay-comparison", default=str(DEFAULT_LIVE_VISUAL_REPLAY_COMPARISON))
    parser.add_argument("--live-canonical-replay-comparison", default=str(DEFAULT_LIVE_CANONICAL_REPLAY_COMPARISON))
    parser.add_argument(
        "--wave3-live-canonical-vs-no-directive-comparison",
        default=str(DEFAULT_WAVE3_LIVE_CANONICAL_VS_NO_DIRECTIVE_COMPARISON),
    )
    parser.add_argument(
        "--wave3-live-canonical-vs-contracted-comparison",
        default=str(DEFAULT_WAVE3_LIVE_CANONICAL_VS_CONTRACTED_COMPARISON),
    )
    parser.add_argument(
        "--wave3-live-visual-vs-no-directive-comparison",
        default=str(DEFAULT_WAVE3_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON),
    )
    parser.add_argument(
        "--wave3-live-visual-vs-contracted-comparison",
        default=str(DEFAULT_WAVE3_LIVE_VISUAL_VS_CONTRACTED_COMPARISON),
    )
    parser.add_argument(
        "--wave4-live-visual-vs-no-directive-comparison",
        default=str(DEFAULT_WAVE4_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON),
    )
    parser.add_argument(
        "--wave4-live-visual-vs-contracted-comparison",
        default=str(DEFAULT_WAVE4_LIVE_VISUAL_VS_CONTRACTED_COMPARISON),
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_report(
        output_dir=args.output_dir,
        h1f_packet=args.h1f_packet,
        h1h_packet=args.h1h_packet,
        h1i_packet=args.h1i_packet,
        probe_comparison_path=args.probe_comparison,
        gemini_packet=args.gemini_packet,
        prompt_contract_packet=args.prompt_contract_packet,
        prompt_contract_wave2_packet=args.prompt_contract_wave2_packet,
        prompt_contract_wave3_packet=args.prompt_contract_wave3_packet,
        prompt_contract_wave4_packet=args.prompt_contract_wave4_packet,
        h1i_prompt_contract_packet=args.h1i_prompt_contract_packet,
        h1i_prompt_contract_repeat_packet=args.h1i_prompt_contract_repeat_packet,
        h1j_prompt_contract_packet=args.h1j_prompt_contract_packet,
        h1j_helper_packet=args.h1j_helper_packet,
        h1k_parallel_audit_packet=args.h1k_parallel_audit_packet,
        h1k_parallel_audit_helper_packet=args.h1k_parallel_audit_helper_packet,
        exact_replay_comparison=args.exact_replay_comparison,
        visual_replay_comparison=args.visual_replay_comparison,
        parallel_replay_comparison=args.parallel_replay_comparison,
        canonical_argument_replay_comparison=args.canonical_argument_replay_comparison,
        live_parallel_replay_comparison=args.live_parallel_replay_comparison,
        live_visual_replay_comparison=args.live_visual_replay_comparison,
        live_canonical_replay_comparison=args.live_canonical_replay_comparison,
        wave3_live_canonical_vs_no_directive_comparison=args.wave3_live_canonical_vs_no_directive_comparison,
        wave3_live_canonical_vs_contracted_comparison=args.wave3_live_canonical_vs_contracted_comparison,
        wave3_live_visual_vs_no_directive_comparison=args.wave3_live_visual_vs_no_directive_comparison,
        wave3_live_visual_vs_contracted_comparison=args.wave3_live_visual_vs_contracted_comparison,
        wave4_live_visual_vs_no_directive_comparison=args.wave4_live_visual_vs_no_directive_comparison,
        wave4_live_visual_vs_contracted_comparison=args.wave4_live_visual_vs_contracted_comparison,
        registry_path=args.registry,
    )
    print(
        json.dumps(
            {
                "output_dir": payload["manifest"]["output_dir"],
                "table_count": payload["manifest"]["table_count"],
                "figure_count": payload["manifest"]["figure_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _packet_payload(name: str, packet_dir: Path) -> dict[str, Any]:
    return {
        "name": name,
        "packet_dir": packet_dir.resolve(),
        "tool_contract": json.loads((packet_dir / "tool_contract_summary.json").read_text(encoding="utf-8")),
        "trace_summary": json.loads((packet_dir / "trace_note_summary.json").read_text(encoding="utf-8")),
    }


def _packet_summary_row(packet: dict[str, Any]) -> dict[str, Any]:
    findings = packet["tool_contract"]["findings"]
    rows = {row["system_id"]: row for row in packet["tool_contract"]["system_rows"]}
    no_repair = rows["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair"]
    no_fallback = rows["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback"]
    no_args = rows["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair"]
    return {
        "packet": packet["name"],
        "episode_count": int(rows["mlx_gemma4_e2b_reasoner_only"]["runs"]),
        "contracted_readiness": _round(findings["contracted_readiness"]),
        "no_directive_readiness": _round(findings["no_directive_readiness"]),
        "readiness_delta_no_directive_vs_contracted": _round(findings["readiness_delta_no_directive_vs_contracted"]),
        "no_directive_controller_repair": _round(findings["no_directive_controller_repair"]),
        "no_directive_controller_fallback": _round(findings["no_directive_controller_fallback"]),
        "no_directive_argument_repair": _round(findings["no_directive_argument_repair"]),
        "no_directive_raw_clean": _round(findings["no_directive_raw_planning_clean_rate"]),
        "no_repair_readiness": _round(no_repair["real_world_readiness_avg"]),
        "no_fallback_readiness": _round(no_fallback["real_world_readiness_avg"]),
        "no_argument_repair_readiness": _round(no_args["real_world_readiness_avg"]),
        "failure_candidates": int(packet["trace_summary"]["failure_candidate_count"]),
    }


def _system_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _system_order(str(row["system_id"])))
    return [
        {
            "label": SYSTEM_LABELS.get(str(row["system_id"]), str(row["system_id"])),
            "system_id": row["system_id"],
            "runs": int(float(row["runs"])),
            "real_world_readiness_avg": _round(row["real_world_readiness_avg"]),
            "strict_interface_avg": _round(row["strict_interface_avg"]),
            "recovered_execution_avg": _round(row["recovered_execution_avg"]),
            "controller_repair_avg": _round(row["controller_repair_avg"]),
            "controller_fallback_avg": _round(row["controller_fallback_avg"]),
            "argument_repair_avg": _round(row["argument_repair_avg"]),
            "raw_planning_clean_rate_avg": _round(row["raw_planning_clean_rate_avg"]),
            "disabled_controls": row.get("disabled_controls", ""),
        }
        for row in ordered
    ]


def _burden_row(packet_name: str, packet: dict[str, Any]) -> dict[str, Any]:
    rows = {row["system_id"]: row for row in packet["tool_contract"]["system_rows"]}
    no_directive = rows["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]
    return {
        "packet": packet_name,
        "controller_repair_avg": _round(no_directive["controller_repair_avg"]),
        "controller_fallback_avg": _round(no_directive["controller_fallback_avg"]),
        "argument_repair_avg": _round(no_directive["argument_repair_avg"]),
        "raw_planning_clean_rate_avg": _round(no_directive["raw_planning_clean_rate_avg"]),
    }


def _probe_failure_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side, field in [
        ("candidate", "candidate_failure_mode"),
        ("baseline_non_exact", "baseline_failure_mode"),
    ]:
        counter: Counter[str] = Counter(
            str(row.get(field, "")) for row in case_rows if str(row.get(field, "")) not in {"", "exact"}
        )
        rows.extend(
            {"side": side, "failure_mode": mode, "case_count": count}
            for mode, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        )
    return rows


def _prompt_contract_candidate_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_id, meta in sorted((registry.get("systems") or {}).items()):
        controls = ResearchControls.from_mapping(meta.get("research_controls"))
        if not controls.tool_prompt_contract_id:
            continue
        contract = TOOL_PROMPT_CONTRACTS.get(controls.tool_prompt_contract_id)
        rows.append(
            {
                "system_id": system_id,
                "short_label": str(meta.get("short_label", system_id)),
                "tool_prompt_contract_id": controls.tool_prompt_contract_id,
                "disable_tool_turn_directive": controls.disable_tool_turn_directive,
                "label": contract.label if contract else "",
                "hypothesis": contract.hypothesis if contract else "",
                "tags": ";".join(contract.tags) if contract else "",
            }
        )
    return rows


def _candidate_tag_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in candidate_rows:
        for tag in str(row.get("tags", "")).split(";"):
            if tag:
                counter[tag] += 1
    return [
        {"label": tag, "value": count}
        for tag, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _exact_replay_gap_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"label": "contracted", "exact_match_rate": float(summary.get("baseline_exact_match_rate") or 0.0)},
        {"label": "no directive", "exact_match_rate": float(summary.get("candidate_exact_match_rate") or 0.0)},
    ]


def _live_replay_gap_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"label": "contracted", "exact_rate": float(summary.get("baseline_exact_rate") or 0.0)},
        {"label": "no directive", "exact_rate": float(summary.get("candidate_exact_rate") or 0.0)},
    ]


def _live_replay_focus_rows(comparisons: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, payload in comparisons:
        summary = payload["summary"]
        rows.append(
            {
                "slice": label,
                "shared_case_count": summary["shared_case_count"],
                "baseline_exact_rate": summary["baseline_exact_rate"],
                "candidate_exact_rate": summary["candidate_exact_rate"],
                "delta_exact_rate": summary["delta_exact_rate"],
                "case_delta_count": summary["case_delta_count"],
            }
        )
    return rows


def _live_candidate_summary_rows(comparisons: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, payload in comparisons:
        summary = payload["summary"]
        rows.append(
            {
                "comparison": label,
                "baseline_system_id": summary["baseline_system_id"],
                "candidate_system_id": summary["candidate_system_id"],
                "shared_case_count": summary["shared_case_count"],
                "baseline_exact_rate": summary["baseline_exact_rate"],
                "candidate_exact_rate": summary["candidate_exact_rate"],
                "delta_exact_rate": summary["delta_exact_rate"],
                "baseline_executable_rate": _none_to_blank(summary.get("baseline_executable_rate")),
                "candidate_executable_rate": _none_to_blank(summary.get("candidate_executable_rate")),
                "delta_executable_rate": _none_to_blank(summary.get("delta_executable_rate")),
            }
        )
    return rows


def _live_candidate_case_rows(comparisons: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, payload in comparisons:
        for row in payload["case_deltas"]:
            rows.append({"comparison": label, **row})
    return rows


def _replay_focus_summary_rows(comparisons: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, payload in comparisons:
        summary = payload["summary"]
        rows.append(
            {
                "slice": label,
                "shared_case_count": summary["shared_case_count"],
                "baseline_exact_match_rate": summary["baseline_exact_match_rate"],
                "candidate_exact_match_rate": summary["candidate_exact_match_rate"],
                "delta_exact_match_rate": summary["delta_exact_match_rate"],
                "case_delta_count": summary["case_delta_count"],
            }
        )
    return rows


def _prompt_contract_promotion_rows(
    *,
    wave1_rows: list[dict[str, Any]],
    wave2_rows: list[dict[str, Any]],
    wave3_rows: list[dict[str, Any]],
    wave4_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wave, gate_rows in [("v1", wave1_rows), ("v2", wave2_rows), ("v3", wave3_rows), ("v4", wave4_rows)]:
        for row in gate_rows:
            decision = _promotion_decision(row)
            rows.append(
                {
                    "wave": wave,
                    "tool_prompt_contract_id": row["tool_prompt_contract_id"],
                    "exact_match_rate": row["exact_match_rate"],
                    "executable_match_rate": row["executable_match_rate"],
                    "delta_exact_vs_no_directive": row["delta_exact_vs_no_directive"],
                    "probe_gate": row["probe_gate"],
                    "recommendation": row["recommendation"],
                    "promotion_decision": decision["promotion_decision"],
                    "promotion_reason": decision["promotion_reason"],
                    "next_use": decision["next_use"],
                }
            )
    return rows


def _promotion_decision(row: dict[str, Any]) -> dict[str, str]:
    exact = float(row.get("exact_match_rate") or 0.0)
    executable = float(row.get("executable_match_rate") or 0.0)
    delta_exact = float(row.get("delta_exact_vs_no_directive") or 0.0)
    recommendation = str(row.get("recommendation", ""))

    if exact >= 0.5 and delta_exact > 0.0:
        return {
            "promotion_decision": "promote_to_h1_candidate",
            "promotion_reason": "raw exact-call rate cleared the exploratory H1 promotion threshold",
            "next_use": "run H1i, then H1h only if controller burden moves",
        }
    if recommendation == "no_probe_gain" or (exact == 0.0 and executable == 0.0):
        return {
            "promotion_decision": "reject_for_h1_promotion",
            "promotion_reason": "no exact or executable probe gain over the no-directive baseline",
            "next_use": "replace with a sharper contract or a faithful live parallel workflow",
        }
    if executable > 0.0 and exact == 0.0:
        return {
            "promotion_decision": "hold_for_exact_probe_replay",
            "promotion_reason": "executable recovery exists, but exact JSON/tool-call fidelity did not improve",
            "next_use": "use in visual replay only, not as a general H1 candidate",
        }
    return {
        "promotion_decision": "hold_for_exact_probe_replay",
        "promotion_reason": "probe gain is too weak for H1 promotion without a stricter replay discriminator",
        "next_use": "test through exact-probe live replay before any H1 spend",
    }


def _label_system_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labelled: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        output["label"] = SYSTEM_LABELS.get(str(row.get("system_id", "")), _candidate_label(str(row.get("system_id", ""))))
        labelled.append(output)
    return sorted(labelled, key=lambda row: _system_order(str(row.get("system_id", ""))))


def _candidate_label(system_id: str) -> str:
    suffixes = {
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor": "schema anchor",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard": "literal guard",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required": "tool required",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required": "schema literal required",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_next_call_state": "visual next call",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required": "parallel array",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_canonical_json_copy": "canonical JSON",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation": "visual initiation",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_two_call_array": "parallel two-call",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection": "visual state tool",
    }
    return suffixes.get(system_id, system_id)


def _markdown_report(payload: dict[str, Any]) -> str:
    packet_rows = payload["packet_summary"]
    h1i_rows = payload["h1i_system_metrics"]
    probe_rows = payload["probe_failure_modes"]
    candidate_rows = payload["prompt_contract_candidates"]
    gate_rows = payload["prompt_contract_probe_gates"]
    wave2_gate_rows = payload["prompt_contract_wave2_probe_gates"]
    wave3_gate_rows = payload["prompt_contract_wave3_probe_gates"]
    wave4_gate_rows = payload["prompt_contract_wave4_probe_gates"]
    h1i_prompt_contract_rows = payload["h1i_prompt_contract_candidate_metrics"]
    h1i_prompt_contract_repeat_rows = payload["h1i_prompt_contract_repeat3_metrics"]
    h1j_prompt_contract_rows = payload["h1j_probe_derived_candidate_metrics"]
    h1j_helper_rows = payload["h1j_probe_derived_helper_metrics"]
    h1k_parallel_audit_rows = payload["h1k_parallel_audit_candidate_metrics"]
    h1k_parallel_audit_helper_rows = payload["h1k_parallel_audit_helper_metrics"]
    promotion_rows = payload["prompt_contract_promotion_decisions"]
    exact_replay_summary = payload["exact_probe_replay_comparison"]["summary"]
    exact_replay_case_rows = payload["exact_probe_replay_case_deltas"]
    exact_replay_focus_rows = payload["exact_probe_replay_focus_summary"]
    live_parallel_replay_summary = payload["live_parallel_replay_comparison"]["summary"]
    live_parallel_replay_case_rows = payload["live_parallel_replay_case_deltas"]
    live_visual_replay_summary = payload["live_visual_replay_comparison"]["summary"]
    live_visual_replay_case_rows = payload["live_visual_replay_case_deltas"]
    live_canonical_replay_summary = payload["live_canonical_replay_comparison"]["summary"]
    live_canonical_replay_case_rows = payload["live_canonical_replay_case_deltas"]
    live_replay_focus_rows = payload["live_replay_focus_summary"]
    wave3_live_summary_rows = payload["wave3_live_candidate_replay_summary"]
    wave3_live_case_rows = payload["wave3_live_candidate_case_deltas"]
    wave4_live_summary_rows = payload["wave4_live_candidate_replay_summary"]
    wave4_live_case_rows = payload["wave4_live_candidate_case_deltas"]
    gemini = payload["gemini"]
    lines = [
        "# MLX Tool-Contract Harnessing Report",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Executive Read",
        "",
        "The current local-Gemma research frontier is no longer top-line readiness on the aligned `32 / 26` surface. "
        "The strongest remaining signal is whether MLX Gemma can stay inside Moonie's tool interface without controller repair, fallback, and argument normalization.",
        "",
        "H1h confirmed that the compact H1f no-directive finding survives the full ten-workflow live surface. "
        "H1i then compressed the worst H1h workflow families into a faster packet and amplified the same causal ordering.",
        "",
        "The main finding is blunt: the tool-turn directive is a real model-side harness intervention, not presentation polish. "
        "When it is removed, no-directive MLX can still match readiness only because the controller repairs or substitutes calls. "
        "Raw no-directive tool compliance collapses on the probe suite.",
        "",
        "## Figures",
        "",
        "![H1i readiness, strict interface, and recovered execution](figures/h1i_readiness_strict_recovered.svg)",
        "",
        "![H1h vs H1i no-directive controller burden](figures/h1h_h1i_controller_burden.svg)",
        "",
        "![Tool probe contract gap](figures/tool_probe_contract_gap.svg)",
        "",
        "![H1i failure modes](figures/h1i_failure_modes.svg)",
        "",
        "![Prompt contract candidate targets](figures/prompt_contract_candidate_targets.svg)",
        "",
        "![Executed prompt contract probe gate](figures/prompt_contract_probe_gate.svg)",
        "",
        "![Prompt contract wave two probe gate](figures/prompt_contract_wave2_probe_gate.svg)",
        "",
        "![Prompt contract wave three probe gate](figures/prompt_contract_wave3_probe_gate.svg)",
        "",
        "![Prompt contract wave four probe gate](figures/prompt_contract_wave4_probe_gate.svg)",
        "",
        "![H1i prompt-contract repeat3 burden](figures/h1i_prompt_contract_repeat3_burden.svg)",
        "",
        "![H1j probe-derived candidate burden](figures/h1j_probe_derived_burden.svg)",
        "",
        "![H1j probe-derived helper burden](figures/h1j_probe_derived_helper_burden.svg)",
        "",
        "![H1k parallel-audit candidate burden](figures/h1k_parallel_audit_burden.svg)",
        "",
        "![H1k parallel-audit helper burden](figures/h1k_parallel_audit_helper_burden.svg)",
        "",
        "![Exact probe replay gap](figures/exact_probe_replay_gap.svg)",
        "",
        "![Focused exact replay gaps](figures/exact_probe_replay_focus_gap.svg)",
        "",
        "![CLI-live parallel replay gap](figures/live_parallel_replay_gap.svg)",
        "",
        "![CLI-live focused replay gaps](figures/live_replay_focus_gap.svg)",
        "",
        "![Wave three live replay gate](figures/wave3_live_candidate_replay_gate.svg)",
        "",
        "![Wave four live replay gate](figures/wave4_live_candidate_replay_gate.svg)",
        "",
        "## Packet Summary",
        "",
        _markdown_table(packet_rows),
        "",
        "## H1i System Metrics",
        "",
        _markdown_table(h1i_rows),
        "",
        "## Probe Failure Modes",
        "",
        _markdown_table(probe_rows),
        "",
        "## Prompt-Contract Candidate Queue",
        "",
        _markdown_table(candidate_rows),
        "",
        "These candidates are generic prompt contracts for the no-directive row. They deliberately avoid embedding the expected planned call, so they can be tested on the probe before spending H1i or H1h runs.",
        "",
        "## Executed Prompt-Contract Probe Gate",
        "",
        _markdown_table(gate_rows),
        "",
        "The first executed probe gate shows only partial gains. `schema_anchor_v1` recovers one exact visual readback case over no-directive, while `literal_argument_guard_v1` and `tool_required_parallel_v1` recover the executable visual target without improving exact JSON copy rate. All three remain far below the contracted MLX probe row.",
        "",
        "## Prompt-Contract Wave Two Probe Gate",
        "",
        _markdown_table(wave2_gate_rows),
        "",
        "The second wave confirms the same shape rather than changing the direction. `schema_literal_tool_required_v2` gives a weak one-case exact gain, `visual_next_call_state_v2` restores executable visual behavior without exact JSON fidelity, and `parallel_array_required_v2` does not improve the parallel/no-call family. None of the wave-two candidates is strong enough to replace the final tool-turn directive.",
        "",
        "## Prompt-Contract Wave Three Probe Gate",
        "",
        _markdown_table(wave3_gate_rows),
        "",
        "The third wave targets the mechanisms exposed by CLI-live replay: canonical argument copying, visual tool initiation, and two-call parallel array shape. It produces the same hard boundary in sharper form: canonical and visual-initiation wording recover one exact case, the visual-initiation contract also recovers the executable visual target, and the parallel two-call contract still does not recover the parallel no-call family.",
        "",
        "## Prompt-Contract Wave Four Probe Gate",
        "",
        _markdown_table(wave4_gate_rows),
        "",
        "`visual_state_tool_selection_v4` was the narrow follow-up to wave three's best partial result. Raw probe exact rate again reaches only `0.125`: enough to improve over the no-directive row by one case, but still far below the contracted row. The dominant failure remains `no_tool_call`, so the contract should be treated as a targeted visual replay candidate, not a general harness fix.",
        "",
        "## Prompt-Contract Promotion Decisions",
        "",
        _markdown_table(promotion_rows),
        "",
        "The promotion gate is intentionally conservative: weak one-case exact gains and visual executable-only gains are held for exact-probe replay, while candidates with no probe gain are rejected for H1 promotion.",
        "",
        "## Exact-Probe Replay Comparison",
        "",
        f"- Baseline exact rate: `{exact_replay_summary['baseline_exact_match_rate']}`",
        f"- Candidate exact rate: `{exact_replay_summary['candidate_exact_match_rate']}`",
        f"- Delta exact rate: `{exact_replay_summary['delta_exact_match_rate']}`",
        "",
        _markdown_table(exact_replay_case_rows),
        "",
        "## Focused Exact-Replay Slices",
        "",
        _markdown_table(exact_replay_focus_rows),
        "",
        "## CLI-Live Parallel Replay Comparison",
        "",
        f"- Contracted exact rate: `{live_parallel_replay_summary['baseline_exact_rate']}`",
        f"- No-directive exact rate: `{live_parallel_replay_summary['candidate_exact_rate']}`",
        f"- Delta exact rate: `{live_parallel_replay_summary['delta_exact_rate']}`",
        "",
        _markdown_table(live_parallel_replay_case_rows),
        "",
        "This is the live-operator counterpart to the focused parallel-array replay. The contracted row emits both expected tool calls, while the no-directive row emits no tool calls and asks the operator to provide inputs that were already present in the replay context.",
        "",
        f"- Visual contracted exact rate: `{live_visual_replay_summary['baseline_exact_rate']}`",
        f"- Visual no-directive exact rate: `{live_visual_replay_summary['candidate_exact_rate']}`",
        f"- Visual delta exact rate: `{live_visual_replay_summary['delta_exact_rate']}`",
        "",
        _markdown_table(live_visual_replay_case_rows),
        "",
        "The visual CLI-live comparison mirrors the focused visual replay: no-directive emits no tool calls in all three cases, while contracted MLX recovers two exact calls and one executable visual paraphrase.",
        "",
        f"- Canonical contracted exact rate: `{live_canonical_replay_summary['baseline_exact_rate']}`",
        f"- Canonical no-directive exact rate: `{live_canonical_replay_summary['candidate_exact_rate']}`",
        f"- Canonical delta exact rate: `{live_canonical_replay_summary['delta_exact_rate']}`",
        "",
        _markdown_table(live_canonical_replay_case_rows),
        "",
        "The canonical CLI/API comparison isolates argument fidelity: both rows emit one tool call per case, but no-directive misses canonical paths, ids, or query strings in all four cases.",
        "",
        "## CLI-Live Focused Replay Summary",
        "",
        _markdown_table(live_replay_focus_rows),
        "",
        "## Wave Three CLI-Live Candidate Replay",
        "",
        _markdown_table(wave3_live_summary_rows),
        "",
        "The live replay gate rejects `canonical_json_copy_v3` for canonical argument promotion: exact rate stays `0.0` against no-directive and two cases regress from argument mismatch to no tool call. `visual_tool_initiation_v3` is the first candidate with live family movement: it improves visual exact rate from `0.0` to `0.3333333333333333`, restores the executable visual-form target, and emits one tool call in all three visual cases. It remains below contracted MLX because one visual referent case still uses the wrong visual tool.",
        "",
        _markdown_table(wave3_live_case_rows),
        "",
        "## Wave Four CLI-Live Candidate Replay",
        "",
        _markdown_table(wave4_live_summary_rows),
        "",
        "`visual_state_tool_selection_v4` keeps the same exact live ceiling as wave three, not a promotion path. It improves over no-directive from `0 / 3` to `1 / 3`, but trails contracted MLX at `2 / 3`, loses executable visual-form recovery, and still fails `visual_latest_filter_literal` with the wrong visual tool. This is useful negative evidence: adding state/tool-selection wording did not fix the remaining visual referent failure.",
        "",
        _markdown_table(wave4_live_case_rows),
        "",
        "## H1i Prompt-Contract Candidate Packet",
        "",
        _markdown_table(h1i_prompt_contract_rows),
        "",
        "The H1i candidate packet is saturated: contracted, no-directive, and all three prompt-contract candidates match on readiness, strict interface, recovered execution, controller burden, and raw clean rate. That means this H1i packet did not discriminate after the probe gate; the next second-stage slice needs harder or repeated no-directive cases.",
        "",
        "## H1i Prompt-Contract Repeat3 Packet",
        "",
        _markdown_table(h1i_prompt_contract_repeat_rows),
        "",
        "The repeated H1i second-stage packet is also saturated. It expands the candidate packet to three attempts per workflow family per row, but all rows still remain controller-clean with raw clean rate `1.0`. The useful conclusion is negative: these packaged H1i workflows are now too deterministic to validate the prompt-contract candidates. The next harder slice should be probe-derived live cases, especially visual/parallel no-call cases.",
        "",
        "## H1j Probe-Derived Candidate Packet",
        "",
        _markdown_table(h1j_prompt_contract_rows),
        "",
        "H1j maps the no-directive probe failures back into six packaged live workflow families. This first candidate packet is also saturated: contracted, no-directive, and all three candidate rows remain controller-clean with raw clean rate `1.0`. That widens the evidence that benchmark-style packaged workflows are easier than the raw tool-contract probe, even when selected from the same failure families.",
        "",
        "## H1j Probe-Derived Helper Packet",
        "",
        _markdown_table(h1j_helper_rows),
        "",
        "The H1j helper-ablation packet is saturated too. Removing controller repair, controller fallback, or argument repair does not change readiness, strict interface, recovered execution, or raw clean rate on this probe-derived packaged workflow set. The trace miner records disabled-helper markers, but no failure candidates.",
        "",
        "## H1k Parallel-Audit Candidate Packet",
        "",
        _markdown_table(h1k_parallel_audit_rows),
        "",
        "H1k promotes the deferred `parallel_audit_array_literal` probe pressure into one packaged live workflow, `ops_parallel_audit_review`. The candidate packet is still saturated: the contracted row, no-directive row, and prompt-contract candidates all match readiness `0.91780`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, and zero controller burden.",
        "",
        "## H1k Parallel-Audit Helper Packet",
        "",
        _markdown_table(h1k_parallel_audit_helper_rows),
        "",
        "The H1k helper packet confirms the negative result. Removing controller repair, controller fallback, or argument repair does not move the staged parallel-audit workflow. The result is useful because it narrows the next experiment: the discriminator must preserve exact one-turn replay shape instead of further decomposing the parallel task into staged packaged steps.",
        "",
        "## Gemini CLI Baseline Status",
        "",
        f"- Packet: `{gemini['packet_run_id']}`",
        f"- H1 slice: `{gemini['h1_slice']}`",
        f"- Workflow count: `{gemini['workflow_count']}`",
        f"- Dry run: `{gemini['dry_run']}`",
        f"- Binary: `{gemini['binary']}`",
        "",
        "This packet is deliberately a dry-run prompt and command manifest. It is an external-reference baseline, not a replacement for Moonie's local MLX harness.",
        "",
        "## Interpretation",
        "",
        "- H1f established the compact causal ordering: no directive plus no controller repair was the largest drop.",
        "- H1h verified that the ordering survives all ten H1e live workflow families.",
        "- H1i is now the best fast loop because it targets the worst H1h no-repair families and makes the repair/fallback gaps larger.",
        "- The no-directive probe explains why: CLI/API calls often keep the right tool but drift on canonical arguments, while visual referent and parallel-tool cases collapse to no tool call.",
        "- The next prompt-contract experiment should be evaluated first on the probe suite and then on H1i before spending another full H1h run.",
        "",
        "## Source Artifacts",
        "",
    ]
    for name, path in payload["manifest"]["source_packets"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            f"- Probe comparison: `{payload['manifest']['probe_comparison']}`",
            f"- Prompt-contract probe packet: `{payload['manifest']['prompt_contract_packet']}`",
            f"- Prompt-contract wave two packet: `{payload['manifest']['prompt_contract_wave2_packet']}`",
            f"- Prompt-contract wave three packet: `{payload['manifest']['prompt_contract_wave3_packet']}`",
            f"- Prompt-contract wave four packet: `{payload['manifest']['prompt_contract_wave4_packet']}`",
            f"- H1i prompt-contract packet: `{payload['manifest']['h1i_prompt_contract_packet']}`",
            f"- H1i prompt-contract repeat packet: `{payload['manifest']['h1i_prompt_contract_repeat_packet']}`",
            f"- H1j probe-derived prompt-contract packet: `{payload['manifest']['h1j_prompt_contract_packet']}`",
            f"- H1j probe-derived helper packet: `{payload['manifest']['h1j_helper_packet']}`",
            f"- H1k parallel-audit prompt-contract packet: `{payload['manifest']['h1k_parallel_audit_packet']}`",
            f"- H1k parallel-audit helper packet: `{payload['manifest']['h1k_parallel_audit_helper_packet']}`",
            f"- Exact replay comparison: `{payload['manifest']['exact_replay_comparison']}`",
            f"- Canonical argument replay comparison: `{payload['manifest']['canonical_argument_replay_comparison']}`",
            f"- Visual replay comparison: `{payload['manifest']['visual_replay_comparison']}`",
            f"- Parallel replay comparison: `{payload['manifest']['parallel_replay_comparison']}`",
            f"- CLI-live parallel replay comparison: `{payload['manifest']['live_parallel_replay_comparison']}`",
            f"- CLI-live visual replay comparison: `{payload['manifest']['live_visual_replay_comparison']}`",
            f"- CLI-live canonical replay comparison: `{payload['manifest']['live_canonical_replay_comparison']}`",
            f"- Wave four live visual vs no-directive comparison: `{payload['manifest']['wave4_live_visual_vs_no_directive_comparison']}`",
            f"- Wave four live visual vs contracted comparison: `{payload['manifest']['wave4_live_visual_vs_contracted_comparison']}`",
            f"- Gemini dry-run baseline: `{payload['manifest']['gemini_packet']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_grouped_metric_svg(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    label_field: str,
    metrics: list[tuple[str, str, str]],
) -> None:
    width = 1120
    left = 250
    top = 80
    group_height = 72
    bar_height = 12
    gap = 6
    chart_width = 760
    axis_max = _axis_max(rows, metrics)
    height = top + len(rows) * group_height + 90
    parts = _svg_header(width, height, title)
    for tick in range(0, 6):
        x = left + chart_width * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="55" x2="{x:.1f}" y2="{height - 45}" stroke="#E5E7EB" stroke-width="1"/>')
        tick_value = axis_max * tick / 5
        parts.append(f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" font-size="12" fill="#475569">{tick_value:.2g}</text>')
    for index, row in enumerate(rows):
        y0 = top + index * group_height
        parts.append(f'<text x="20" y="{y0 + 23}" font-size="13" fill="#111827">{_escape(str(row[label_field]))}</text>')
        for metric_index, (field, label, color) in enumerate(metrics):
            value = float(row.get(field) or 0.0)
            y = y0 + metric_index * (bar_height + gap) + 10
            bar_width = max(0.0, min(value, axis_max)) / axis_max * chart_width
            parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{left + chart_width + 12}" y="{y + 10}" font-size="12" fill="#334155">{label}: {value:.3f}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _axis_max(rows: list[dict[str, Any]], metrics: list[tuple[str, str, str]]) -> float:
    highest = max((float(row.get(field) or 0.0) for row in rows for field, _, _ in metrics), default=1.0)
    if highest <= 1.0:
        return 1.0
    if highest <= 1.25:
        return 1.25
    if highest <= 1.5:
        return 1.5
    return highest


def _write_bar_svg(path: Path, *, title: str, rows: list[dict[str, Any]], color: str) -> None:
    width = 980
    left = 260
    top = 80
    bar_height = 24
    row_gap = 18
    chart_width = 560
    max_value = max((float(row["value"]) for row in rows), default=1.0)
    height = top + len(rows) * (bar_height + row_gap) + 70
    parts = _svg_header(width, height, title)
    for index, row in enumerate(rows):
        y = top + index * (bar_height + row_gap)
        value = float(row["value"])
        bar_width = value / max_value * chart_width if max_value else 0
        parts.append(f'<text x="20" y="{y + 17}" font-size="13" fill="#111827">{_escape(str(row["label"]))}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_width + 10}" y="{y + 17}" font-size="12" fill="#334155">{int(value)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="20" y="35" font-size="20" font-family="Inter, Arial, sans-serif" font-weight="700" fill="#0F172A">{_escape(title)}</text>',
        '<style>text{font-family:Inter,Arial,sans-serif}</style>',
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    fields = list(rows[0].keys())
    output = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(output)


def _system_order(system_id: str) -> int:
    ordered = list(SYSTEM_LABELS)
    return ordered.index(system_id) if system_id in ordered else len(ordered)


def _none_to_blank(value: Any) -> Any:
    return "" if value is None else value


def _round(value: Any) -> float:
    return round(float(value), 5)


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
