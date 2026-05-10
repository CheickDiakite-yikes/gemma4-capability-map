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
DEFAULT_PROMPT_CONTRACT_WAVE5_PACKET = (
    ROOT / "results" / "tool_prompt_contract_probe_packets" / "20260508T_prompt_contract_wave5_execute_v1"
)
DEFAULT_TOOL_CATALOG_PROFILE_PACKET = (
    ROOT / "results" / "tool_catalog_profile_probe_packets" / "20260508T_visual_role_catalog_v1_probe"
)
DEFAULT_TOOL_CATALOG_ARGUMENT_HINTS_PACKET = (
    ROOT / "results" / "tool_catalog_profile_probe_packets" / "20260508T_visual_role_catalog_argument_hints_v2_probe"
)
DEFAULT_TOOL_CATALOG_ARGUMENT_HINTS_VS_ROLE_CATALOG_COMPARISON = (
    ROOT
    / "results"
    / "tool_catalog_profile_probe_comparisons"
    / "20260508T_visual_argument_hints_vs_role_catalog_v1"
)
DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_PACKET = (
    ROOT / "results" / "tool_catalog_profile_probe_packets" / "20260508T_visual_role_catalog_split_selector_hints_v3_probe"
)
DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_VS_ARGUMENT_HINTS_COMPARISON = (
    ROOT
    / "results"
    / "tool_catalog_profile_probe_comparisons"
    / "20260508T_visual_split_selector_hints_vs_argument_hints_v2"
)
DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_VS_ROLE_CATALOG_COMPARISON = (
    ROOT
    / "results"
    / "tool_catalog_profile_probe_comparisons"
    / "20260508T_visual_split_selector_hints_vs_role_catalog_v1"
)
DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_LIVE_DECISION = (
    ROOT / "results" / "tool_probe_replay_live" / "20260508T_visual_split_selector_hints_live_replay_skipped_v1"
)
DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_PACKET = (
    ROOT / "results" / "tool_catalog_profile_probe_packets" / "20260509T_visual_role_catalog_schema_field_hints_v4_probe"
)
DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_ARGUMENT_HINTS_COMPARISON = (
    ROOT
    / "results"
    / "tool_catalog_profile_probe_comparisons"
    / "20260509T_visual_schema_field_hints_vs_argument_hints_v2"
)
DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_SPLIT_SELECTOR_COMPARISON = (
    ROOT
    / "results"
    / "tool_catalog_profile_probe_comparisons"
    / "20260509T_visual_schema_field_hints_vs_split_selector_v3"
)
DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_ROLE_CATALOG_COMPARISON = (
    ROOT
    / "results"
    / "tool_catalog_profile_probe_comparisons"
    / "20260509T_visual_schema_field_hints_vs_role_catalog_v1"
)
DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_LIVE_DECISION = (
    ROOT / "results" / "tool_probe_replay_live" / "20260509T_visual_schema_field_hints_live_replay_skipped_v1"
)
DEFAULT_PROMPT_CONTRACT_WAVE6_PACKET = (
    ROOT / "results" / "tool_prompt_contract_probe_packets" / "20260508T_visual_catalog_literal_guard_v6_probe"
)
DEFAULT_VISUAL_HARD_SLICE_PACKET = (
    ROOT / "results" / "visual_hard_slice_probe_packets" / "20260509T_visual_hard_slice_executor_equivalence_v1"
)
DEFAULT_VISUAL_HARD_SLICE_EXACTNESS_DIAGNOSTIC = (
    ROOT / "results" / "reports" / "visual_hard_slice_exactness_diagnostic"
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
DEFAULT_H1L_VISUAL_EXECUTOR_EQUIVALENCE_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet"
)
DEFAULT_H1M_VISUAL_ALIAS_REPEAT_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet"
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
DEFAULT_CATALOG_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260508T_visual_role_catalog_vs_no_directive_v1"
)
DEFAULT_CATALOG_LIVE_VISUAL_VS_VISUAL_INITIATION_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260508T_visual_role_catalog_vs_visual_tool_initiation_v1"
)
DEFAULT_CATALOG_LIVE_VISUAL_VS_VISUAL_STATE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1"
)
DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260508T_visual_catalog_argument_hints_vs_no_directive_v1"
)
DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_CONTRACTED_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260508T_visual_catalog_argument_hints_vs_contracted_v1"
)
DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_ROLE_CATALOG_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260508T_visual_catalog_argument_hints_vs_role_catalog_v1"
)
DEFAULT_VISUAL_HARD_SLICE_LIVE_REPLAY_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_schema_field_hints_vs_no_directive_live_v2"
)
DEFAULT_VISUAL_HARD_SLICE_CONTRACTED_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_contracted_vs_no_directive_live_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ROLE_CATALOG_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_role_catalog_vs_no_directive_live_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ARGUMENT_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_argument_hints_vs_no_directive_live_v1"
)
DEFAULT_VISUAL_HARD_SLICE_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_schema_literal_targets_vs_no_directive_live_v1"
)
DEFAULT_VISUAL_HARD_SLICE_STRESS_CONTRACTED_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_contracted_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_STRESS_ROLE_CATALOG_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_role_catalog_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_STRESS_ARGUMENT_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_argument_hints_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_STRESS_SCHEMA_FIELD_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_schema_field_hints_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_STRESS_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_schema_literal_targets_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_CONTRACTED_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_repeat_contracted_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_ROLE_CATALOG_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_repeat_role_catalog_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_ARGUMENT_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_repeat_argument_hints_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_SCHEMA_FIELD_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_repeat_schema_literal_targets_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_CONTRACTED_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_contracted_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ROLE_CATALOG_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_role_catalog_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ARGUMENT_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_argument_hints_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_SCHEMA_FIELD_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_schema_field_hints_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_schema_literal_targets_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_CONTRACTED_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_contracted_vs_no_directive_v2"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_ROLE_CATALOG_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_role_catalog_vs_no_directive_v2"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_ARGUMENT_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_argument_hints_vs_no_directive_v2"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_SCHEMA_FIELD_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_schema_field_hints_vs_no_directive_v2"
)
DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_schema_literal_targets_vs_no_directive_v2"
)
DEFAULT_VISUAL_HARD_SLICE_POST_REPAIR_CONTRACTED_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1n_post_repair_contracted_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_POST_REPAIR_ARGUMENT_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1n_post_repair_argument_hints_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_POST_REPAIR_CODE_HINTS_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1n_post_repair_code_hints_vs_no_directive_v1"
)
DEFAULT_VISUAL_HARD_SLICE_POST_REPAIR_CODE_GUARD_LIVE_COMPARISON = (
    ROOT
    / "results"
    / "tool_probe_replay_live_comparisons"
    / "20260510T_h1n_post_repair_code_guard_vs_no_directive_v1"
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
    prompt_contract_wave5_packet: str | Path = DEFAULT_PROMPT_CONTRACT_WAVE5_PACKET,
    tool_catalog_profile_packet: str | Path = DEFAULT_TOOL_CATALOG_PROFILE_PACKET,
    tool_catalog_argument_hints_packet: str | Path = DEFAULT_TOOL_CATALOG_ARGUMENT_HINTS_PACKET,
    tool_catalog_argument_hints_vs_role_catalog_comparison: str
    | Path = DEFAULT_TOOL_CATALOG_ARGUMENT_HINTS_VS_ROLE_CATALOG_COMPARISON,
    tool_catalog_split_selector_packet: str | Path = DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_PACKET,
    tool_catalog_split_selector_vs_argument_hints_comparison: str
    | Path = DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_VS_ARGUMENT_HINTS_COMPARISON,
    tool_catalog_split_selector_vs_role_catalog_comparison: str
    | Path = DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_VS_ROLE_CATALOG_COMPARISON,
    tool_catalog_split_selector_live_decision: str | Path = DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_LIVE_DECISION,
    tool_catalog_schema_field_hints_packet: str | Path = DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_PACKET,
    tool_catalog_schema_field_hints_vs_argument_hints_comparison: str
    | Path = DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_ARGUMENT_HINTS_COMPARISON,
    tool_catalog_schema_field_hints_vs_split_selector_comparison: str
    | Path = DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_SPLIT_SELECTOR_COMPARISON,
    tool_catalog_schema_field_hints_vs_role_catalog_comparison: str
    | Path = DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_ROLE_CATALOG_COMPARISON,
    tool_catalog_schema_field_hints_live_decision: str | Path = DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_LIVE_DECISION,
    prompt_contract_wave6_packet: str | Path = DEFAULT_PROMPT_CONTRACT_WAVE6_PACKET,
    visual_hard_slice_packet: str | Path = DEFAULT_VISUAL_HARD_SLICE_PACKET,
    visual_hard_slice_exactness_diagnostic: str | Path = DEFAULT_VISUAL_HARD_SLICE_EXACTNESS_DIAGNOSTIC,
    h1i_prompt_contract_packet: str | Path = DEFAULT_H1I_PROMPT_CONTRACT_PACKET,
    h1i_prompt_contract_repeat_packet: str | Path = DEFAULT_H1I_PROMPT_CONTRACT_REPEAT_PACKET,
    h1j_prompt_contract_packet: str | Path = DEFAULT_H1J_PROMPT_CONTRACT_PACKET,
    h1j_helper_packet: str | Path = DEFAULT_H1J_HELPER_PACKET,
    h1k_parallel_audit_packet: str | Path = DEFAULT_H1K_PARALLEL_AUDIT_PACKET,
    h1k_parallel_audit_helper_packet: str | Path = DEFAULT_H1K_PARALLEL_AUDIT_HELPER_PACKET,
    h1l_visual_executor_equivalence_packet: str | Path = DEFAULT_H1L_VISUAL_EXECUTOR_EQUIVALENCE_PACKET,
    h1m_visual_alias_repeat_packet: str | Path = DEFAULT_H1M_VISUAL_ALIAS_REPEAT_PACKET,
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
    catalog_live_visual_vs_no_directive_comparison: str | Path = DEFAULT_CATALOG_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON,
    catalog_live_visual_vs_visual_initiation_comparison: str | Path = DEFAULT_CATALOG_LIVE_VISUAL_VS_VISUAL_INITIATION_COMPARISON,
    catalog_live_visual_vs_visual_state_comparison: str | Path = DEFAULT_CATALOG_LIVE_VISUAL_VS_VISUAL_STATE_COMPARISON,
    argument_hints_live_visual_vs_no_directive_comparison: str
    | Path = DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON,
    argument_hints_live_visual_vs_contracted_comparison: str
    | Path = DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_CONTRACTED_COMPARISON,
    argument_hints_live_visual_vs_role_catalog_comparison: str
    | Path = DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_ROLE_CATALOG_COMPARISON,
    visual_hard_slice_contracted_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_CONTRACTED_LIVE_COMPARISON,
    visual_hard_slice_role_catalog_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ROLE_CATALOG_LIVE_COMPARISON,
    visual_hard_slice_argument_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ARGUMENT_HINTS_LIVE_COMPARISON,
    visual_hard_slice_live_replay_comparison: str | Path = DEFAULT_VISUAL_HARD_SLICE_LIVE_REPLAY_COMPARISON,
    visual_hard_slice_schema_literal_targets_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON,
    visual_hard_slice_stress_contracted_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_STRESS_CONTRACTED_LIVE_COMPARISON,
    visual_hard_slice_stress_role_catalog_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_STRESS_ROLE_CATALOG_LIVE_COMPARISON,
    visual_hard_slice_stress_argument_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_STRESS_ARGUMENT_HINTS_LIVE_COMPARISON,
    visual_hard_slice_stress_schema_field_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_STRESS_SCHEMA_FIELD_HINTS_LIVE_COMPARISON,
    visual_hard_slice_stress_schema_literal_targets_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_STRESS_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON,
    visual_hard_slice_alias_repeat_contracted_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_CONTRACTED_LIVE_COMPARISON,
    visual_hard_slice_alias_repeat_role_catalog_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_ROLE_CATALOG_LIVE_COMPARISON,
    visual_hard_slice_alias_repeat_argument_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_ARGUMENT_HINTS_LIVE_COMPARISON,
    visual_hard_slice_alias_repeat_schema_field_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_SCHEMA_FIELD_HINTS_LIVE_COMPARISON,
    visual_hard_slice_alias_repeat_schema_literal_targets_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_contracted_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_CONTRACTED_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_role_catalog_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ROLE_CATALOG_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_argument_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ARGUMENT_HINTS_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_schema_field_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_SCHEMA_FIELD_HINTS_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_schema_literal_targets_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_oracle_contracted_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_CONTRACTED_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_oracle_role_catalog_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_ROLE_CATALOG_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_oracle_argument_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_ARGUMENT_HINTS_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_oracle_schema_field_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_SCHEMA_FIELD_HINTS_LIVE_COMPARISON,
    visual_hard_slice_alias_transfer_oracle_schema_literal_targets_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON,
    visual_hard_slice_post_repair_contracted_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_POST_REPAIR_CONTRACTED_LIVE_COMPARISON,
    visual_hard_slice_post_repair_argument_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_POST_REPAIR_ARGUMENT_HINTS_LIVE_COMPARISON,
    visual_hard_slice_post_repair_code_hints_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_POST_REPAIR_CODE_HINTS_LIVE_COMPARISON,
    visual_hard_slice_post_repair_code_guard_live_comparison: str
    | Path = DEFAULT_VISUAL_HARD_SLICE_POST_REPAIR_CODE_GUARD_LIVE_COMPARISON,
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
    prompt_contract_wave5_gate_rows = _csv_rows(Path(prompt_contract_wave5_packet) / "candidate_gate_summary.csv")
    prompt_contract_wave5_failure_rows = _csv_rows(Path(prompt_contract_wave5_packet) / "candidate_failure_mode_counts.csv")
    tool_catalog_profile_gate_rows = (
        _csv_rows(Path(tool_catalog_profile_packet) / "candidate_summary.csv")
        + _csv_rows(Path(tool_catalog_argument_hints_packet) / "candidate_summary.csv")
        + _csv_rows(Path(tool_catalog_split_selector_packet) / "candidate_summary.csv")
        + _csv_rows(Path(tool_catalog_schema_field_hints_packet) / "candidate_summary.csv")
    )
    tool_catalog_argument_hints_vs_role_catalog_payload = json.loads(
        (
            Path(tool_catalog_argument_hints_vs_role_catalog_comparison)
            / "probe_comparison.json"
        ).read_text(encoding="utf-8")
    )
    tool_catalog_argument_hints_vs_role_catalog_case_rows = _csv_rows(
        Path(tool_catalog_argument_hints_vs_role_catalog_comparison) / "probe_case_deltas.csv"
    )
    tool_catalog_split_selector_vs_argument_hints_payload = json.loads(
        (
            Path(tool_catalog_split_selector_vs_argument_hints_comparison)
            / "probe_comparison.json"
        ).read_text(encoding="utf-8")
    )
    tool_catalog_split_selector_vs_argument_hints_case_rows = _csv_rows(
        Path(tool_catalog_split_selector_vs_argument_hints_comparison) / "probe_case_deltas.csv"
    )
    tool_catalog_split_selector_vs_role_catalog_payload = json.loads(
        (
            Path(tool_catalog_split_selector_vs_role_catalog_comparison)
            / "probe_comparison.json"
        ).read_text(encoding="utf-8")
    )
    tool_catalog_split_selector_vs_role_catalog_case_rows = _csv_rows(
        Path(tool_catalog_split_selector_vs_role_catalog_comparison) / "probe_case_deltas.csv"
    )
    tool_catalog_split_selector_live_decision_rows = [
        _live_decision_row(Path(tool_catalog_split_selector_live_decision) / "manifest.json")
    ]
    tool_catalog_schema_field_hints_vs_argument_hints_payload = json.loads(
        (
            Path(tool_catalog_schema_field_hints_vs_argument_hints_comparison)
            / "probe_comparison.json"
        ).read_text(encoding="utf-8")
    )
    tool_catalog_schema_field_hints_vs_argument_hints_case_rows = _csv_rows(
        Path(tool_catalog_schema_field_hints_vs_argument_hints_comparison) / "probe_case_deltas.csv"
    )
    tool_catalog_schema_field_hints_vs_split_selector_payload = json.loads(
        (
            Path(tool_catalog_schema_field_hints_vs_split_selector_comparison)
            / "probe_comparison.json"
        ).read_text(encoding="utf-8")
    )
    tool_catalog_schema_field_hints_vs_split_selector_case_rows = _csv_rows(
        Path(tool_catalog_schema_field_hints_vs_split_selector_comparison) / "probe_case_deltas.csv"
    )
    tool_catalog_schema_field_hints_vs_role_catalog_payload = json.loads(
        (
            Path(tool_catalog_schema_field_hints_vs_role_catalog_comparison)
            / "probe_comparison.json"
        ).read_text(encoding="utf-8")
    )
    tool_catalog_schema_field_hints_vs_role_catalog_case_rows = _csv_rows(
        Path(tool_catalog_schema_field_hints_vs_role_catalog_comparison) / "probe_case_deltas.csv"
    )
    tool_catalog_schema_field_hints_live_decision_rows = [
        _live_decision_row(Path(tool_catalog_schema_field_hints_live_decision) / "manifest.json")
    ]
    prompt_contract_wave6_gate_rows = _csv_rows(Path(prompt_contract_wave6_packet) / "candidate_gate_summary.csv")
    prompt_contract_wave6_failure_rows = _csv_rows(Path(prompt_contract_wave6_packet) / "candidate_failure_mode_counts.csv")
    visual_hard_slice_gate_rows = _label_system_rows(_csv_rows(Path(visual_hard_slice_packet) / "candidate_gate_summary.csv"))
    visual_hard_slice_failure_rows = _csv_rows(Path(visual_hard_slice_packet) / "candidate_failure_mode_counts.csv")
    visual_hard_slice_family_rows = _label_system_rows(_csv_rows(Path(visual_hard_slice_packet) / "family_summary.csv"))
    visual_hard_slice_case_deltas_vs_no_directive_rows = _label_system_rows(
        _csv_rows(Path(visual_hard_slice_packet) / "case_deltas_vs_no_directive.csv")
    )
    visual_hard_slice_case_deltas_vs_contracted_rows = _label_system_rows(
        _csv_rows(Path(visual_hard_slice_packet) / "case_deltas_vs_contracted.csv")
    )
    visual_hard_slice_exactness_summary_rows = _csv_rows(
        Path(visual_hard_slice_exactness_diagnostic) / "tables" / "visual_hard_slice_exactness_summary.csv"
    )
    visual_hard_slice_exactness_gap_rows = _csv_rows(
        Path(visual_hard_slice_exactness_diagnostic) / "tables" / "visual_hard_slice_exactness_gaps.csv"
    )
    prompt_contract_promotion_rows = _prompt_contract_promotion_rows(
        wave1_rows=prompt_contract_gate_rows,
        wave2_rows=prompt_contract_wave2_gate_rows,
        wave3_rows=prompt_contract_wave3_gate_rows,
        wave4_rows=prompt_contract_wave4_gate_rows,
        wave5_rows=prompt_contract_wave5_gate_rows,
        wave6_rows=prompt_contract_wave6_gate_rows,
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
    h1l_visual_executor_equivalence_rows = _csv_rows(
        Path(h1l_visual_executor_equivalence_packet) / "tool_contract_system_deltas.csv"
    )
    h1m_visual_alias_repeat_rows = _csv_rows(Path(h1m_visual_alias_repeat_packet) / "tool_contract_system_deltas.csv")
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
    catalog_live_comparisons = [
        (
            "visual role catalog vs no directive",
            json.loads(
                (Path(catalog_live_visual_vs_no_directive_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "visual role catalog vs visual initiation",
            json.loads(
                (
                    Path(catalog_live_visual_vs_visual_initiation_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "visual role catalog vs visual state tool",
            json.loads(
                (Path(catalog_live_visual_vs_visual_state_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
    ]
    catalog_live_summary_rows = _live_candidate_summary_rows(catalog_live_comparisons)
    catalog_live_case_rows = _live_candidate_case_rows(catalog_live_comparisons)
    argument_hints_live_comparisons = [
        (
            "visual argument hints vs no directive",
            json.loads(
                (
                    Path(argument_hints_live_visual_vs_no_directive_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "visual argument hints vs contracted",
            json.loads(
                (
                    Path(argument_hints_live_visual_vs_contracted_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "visual argument hints vs role catalog",
            json.loads(
                (
                    Path(argument_hints_live_visual_vs_role_catalog_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
    ]
    argument_hints_live_summary_rows = _live_candidate_summary_rows(argument_hints_live_comparisons)
    argument_hints_live_case_rows = _live_candidate_case_rows(argument_hints_live_comparisons)
    visual_hard_slice_live_comparisons = [
        (
            "contracted vs no directive",
            json.loads(
                (Path(visual_hard_slice_contracted_live_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "role catalog vs no directive",
            json.loads(
                (Path(visual_hard_slice_role_catalog_live_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "argument hints vs no directive",
            json.loads(
                (Path(visual_hard_slice_argument_hints_live_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "schema-field hints vs no directive",
            json.loads(
                (Path(visual_hard_slice_live_replay_comparison) / "live_replay_comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "schema literal targets vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_schema_literal_targets_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
    ]
    visual_hard_slice_live_replay_summary_rows = _live_candidate_summary_rows(visual_hard_slice_live_comparisons)
    visual_hard_slice_live_replay_case_rows = _live_candidate_case_rows(visual_hard_slice_live_comparisons)
    visual_hard_slice_stress_live_comparisons = [
        (
            "stress contracted vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_stress_contracted_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "stress role catalog vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_stress_role_catalog_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "stress argument hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_stress_argument_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "stress schema-field hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_stress_schema_field_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "stress schema literal targets vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_stress_schema_literal_targets_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
    ]
    visual_hard_slice_stress_live_summary_rows = _live_candidate_summary_rows(
        visual_hard_slice_stress_live_comparisons
    )
    visual_hard_slice_stress_live_case_rows = _live_candidate_case_rows(visual_hard_slice_stress_live_comparisons)
    visual_hard_slice_alias_repeat_live_comparisons = [
        (
            "alias-repeat contracted vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_repeat_contracted_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-repeat role catalog vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_repeat_role_catalog_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-repeat argument hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_repeat_argument_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-repeat schema-field hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_repeat_schema_field_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-repeat schema literal targets vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_repeat_schema_literal_targets_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
    ]
    visual_hard_slice_alias_repeat_live_summary_rows = _live_candidate_summary_rows(
        visual_hard_slice_alias_repeat_live_comparisons
    )
    visual_hard_slice_alias_repeat_live_case_rows = _live_candidate_case_rows(
        visual_hard_slice_alias_repeat_live_comparisons
    )
    visual_hard_slice_alias_transfer_live_comparisons = [
        (
            "alias-transfer contracted vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_contracted_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-transfer role catalog vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_role_catalog_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-transfer argument hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_argument_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-transfer schema-field hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_schema_field_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-transfer schema literal targets vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_schema_literal_targets_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
    ]
    visual_hard_slice_alias_transfer_live_summary_rows = _live_candidate_summary_rows(
        visual_hard_slice_alias_transfer_live_comparisons
    )
    visual_hard_slice_alias_transfer_live_case_rows = _live_candidate_case_rows(
        visual_hard_slice_alias_transfer_live_comparisons
    )
    visual_hard_slice_alias_transfer_oracle_live_comparisons = [
        (
            "alias-transfer oracle contracted vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_oracle_contracted_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-transfer oracle role catalog vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_oracle_role_catalog_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-transfer oracle argument hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_oracle_argument_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-transfer oracle schema-field hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_oracle_schema_field_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "alias-transfer oracle schema literal targets vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_alias_transfer_oracle_schema_literal_targets_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
    ]
    visual_hard_slice_alias_transfer_oracle_live_summary_rows = _live_candidate_summary_rows(
        visual_hard_slice_alias_transfer_oracle_live_comparisons
    )
    visual_hard_slice_alias_transfer_oracle_live_case_rows = _live_candidate_case_rows(
        visual_hard_slice_alias_transfer_oracle_live_comparisons
    )
    visual_hard_slice_post_repair_live_comparisons = [
        (
            "post-repair contracted vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_post_repair_contracted_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "post-repair argument hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_post_repair_argument_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "post-repair oblique code hints vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_post_repair_code_hints_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
        (
            "post-repair oblique code guard vs no directive",
            json.loads(
                (
                    Path(visual_hard_slice_post_repair_code_guard_live_comparison)
                    / "live_replay_comparison.json"
                ).read_text(encoding="utf-8")
            ),
        ),
    ]
    visual_hard_slice_post_repair_live_summary_rows = _live_candidate_summary_rows(
        visual_hard_slice_post_repair_live_comparisons
    )
    visual_hard_slice_post_repair_live_case_rows = _live_candidate_case_rows(
        visual_hard_slice_post_repair_live_comparisons
    )

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
    _write_csv(tables_dir / "prompt_contract_wave5_probe_gates.csv", prompt_contract_wave5_gate_rows)
    _write_csv(tables_dir / "prompt_contract_wave5_probe_failure_modes.csv", prompt_contract_wave5_failure_rows)
    _write_csv(tables_dir / "tool_catalog_profile_probe_gates.csv", tool_catalog_profile_gate_rows)
    _write_csv(
        tables_dir / "tool_catalog_argument_hints_vs_role_catalog_case_deltas.csv",
        tool_catalog_argument_hints_vs_role_catalog_case_rows,
    )
    _write_csv(
        tables_dir / "tool_catalog_split_selector_vs_argument_hints_case_deltas.csv",
        tool_catalog_split_selector_vs_argument_hints_case_rows,
    )
    _write_csv(
        tables_dir / "tool_catalog_split_selector_vs_role_catalog_case_deltas.csv",
        tool_catalog_split_selector_vs_role_catalog_case_rows,
    )
    _write_csv(
        tables_dir / "tool_catalog_split_selector_live_replay_decision.csv",
        tool_catalog_split_selector_live_decision_rows,
    )
    _write_csv(
        tables_dir / "tool_catalog_schema_field_hints_vs_argument_hints_case_deltas.csv",
        tool_catalog_schema_field_hints_vs_argument_hints_case_rows,
    )
    _write_csv(
        tables_dir / "tool_catalog_schema_field_hints_vs_split_selector_case_deltas.csv",
        tool_catalog_schema_field_hints_vs_split_selector_case_rows,
    )
    _write_csv(
        tables_dir / "tool_catalog_schema_field_hints_vs_role_catalog_case_deltas.csv",
        tool_catalog_schema_field_hints_vs_role_catalog_case_rows,
    )
    _write_csv(
        tables_dir / "tool_catalog_schema_field_hints_live_replay_decision.csv",
        tool_catalog_schema_field_hints_live_decision_rows,
    )
    _write_csv(tables_dir / "prompt_contract_wave6_probe_gates.csv", prompt_contract_wave6_gate_rows)
    _write_csv(tables_dir / "prompt_contract_wave6_probe_failure_modes.csv", prompt_contract_wave6_failure_rows)
    _write_csv(tables_dir / "visual_hard_slice_probe_gates.csv", visual_hard_slice_gate_rows)
    _write_csv(tables_dir / "visual_hard_slice_failure_modes.csv", visual_hard_slice_failure_rows)
    _write_csv(tables_dir / "visual_hard_slice_family_summary.csv", visual_hard_slice_family_rows)
    _write_csv(
        tables_dir / "visual_hard_slice_case_deltas_vs_no_directive.csv",
        visual_hard_slice_case_deltas_vs_no_directive_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_case_deltas_vs_contracted.csv",
        visual_hard_slice_case_deltas_vs_contracted_rows,
    )
    _write_csv(tables_dir / "visual_hard_slice_exactness_summary.csv", visual_hard_slice_exactness_summary_rows)
    _write_csv(tables_dir / "visual_hard_slice_exactness_gaps.csv", visual_hard_slice_exactness_gap_rows)
    _write_csv(tables_dir / "prompt_contract_promotion_decisions.csv", prompt_contract_promotion_rows)
    _write_csv(tables_dir / "h1i_prompt_contract_candidate_metrics.csv", h1i_prompt_contract_rows)
    _write_csv(tables_dir / "h1i_prompt_contract_repeat3_metrics.csv", h1i_prompt_contract_repeat_rows)
    _write_csv(tables_dir / "h1j_probe_derived_candidate_metrics.csv", h1j_prompt_contract_rows)
    _write_csv(tables_dir / "h1j_probe_derived_helper_metrics.csv", h1j_helper_rows)
    _write_csv(tables_dir / "h1k_parallel_audit_candidate_metrics.csv", h1k_parallel_audit_rows)
    _write_csv(tables_dir / "h1k_parallel_audit_helper_metrics.csv", h1k_parallel_audit_helper_rows)
    _write_csv(tables_dir / "h1l_visual_executor_equivalence_candidate_metrics.csv", h1l_visual_executor_equivalence_rows)
    _write_csv(tables_dir / "h1m_visual_alias_repeat_candidate_metrics.csv", h1m_visual_alias_repeat_rows)
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
    _write_csv(tables_dir / "visual_catalog_live_candidate_replay_summary.csv", catalog_live_summary_rows)
    _write_csv(tables_dir / "visual_catalog_live_candidate_case_deltas.csv", catalog_live_case_rows)
    _write_csv(
        tables_dir / "visual_catalog_argument_hints_live_candidate_replay_summary.csv",
        argument_hints_live_summary_rows,
    )
    _write_csv(
        tables_dir / "visual_catalog_argument_hints_live_candidate_case_deltas.csv",
        argument_hints_live_case_rows,
    )
    _write_csv(tables_dir / "visual_hard_slice_live_replay_summary.csv", visual_hard_slice_live_replay_summary_rows)
    _write_csv(tables_dir / "visual_hard_slice_live_replay_case_deltas.csv", visual_hard_slice_live_replay_case_rows)
    _write_csv(
        tables_dir / "visual_hard_slice_stress_live_replay_summary.csv",
        visual_hard_slice_stress_live_summary_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_stress_live_replay_case_deltas.csv",
        visual_hard_slice_stress_live_case_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_alias_repeat_live_replay_summary.csv",
        visual_hard_slice_alias_repeat_live_summary_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_alias_repeat_live_replay_case_deltas.csv",
        visual_hard_slice_alias_repeat_live_case_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_alias_transfer_live_replay_summary.csv",
        visual_hard_slice_alias_transfer_live_summary_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_alias_transfer_live_replay_case_deltas.csv",
        visual_hard_slice_alias_transfer_live_case_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv",
        visual_hard_slice_alias_transfer_oracle_live_summary_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_alias_transfer_oracle_live_replay_case_deltas.csv",
        visual_hard_slice_alias_transfer_oracle_live_case_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_post_repair_live_replay_summary.csv",
        visual_hard_slice_post_repair_live_summary_rows,
    )
    _write_csv(
        tables_dir / "visual_hard_slice_post_repair_live_replay_case_deltas.csv",
        visual_hard_slice_post_repair_live_case_rows,
    )

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
        figures_dir / "prompt_contract_wave5_probe_gate.svg",
        title="Prompt contract wave five probe gate",
        rows=prompt_contract_wave5_gate_rows,
        label_field="tool_prompt_contract_id",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
            ("delta_exact_vs_no_directive", "delta exact", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "tool_catalog_profile_probe_gate.svg",
        title="Tool catalog profile probe gate",
        rows=tool_catalog_profile_gate_rows,
        label_field="tool_catalog_profile_id",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
            ("delta_exact_vs_no_directive", "delta exact", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "prompt_contract_wave6_probe_gate.svg",
        title="Prompt contract wave six probe gate",
        rows=prompt_contract_wave6_gate_rows,
        label_field="tool_prompt_contract_id",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
            ("delta_exact_vs_no_directive", "delta exact", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "visual_hard_slice_probe_gate.svg",
        title="Visual hard-slice probe gate",
        rows=visual_hard_slice_gate_rows,
        label_field="label",
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
        figures_dir / "h1l_visual_executor_equivalence_burden.svg",
        title="H1l visual executor-equivalence candidate burden",
        rows=_label_system_rows(h1l_visual_executor_equivalence_rows),
        label_field="label",
        metrics=[
            ("controller_repair_avg", "repair", "#7C3AED"),
            ("controller_fallback_avg", "fallback", "#DC2626"),
            ("argument_repair_avg", "arg repair", "#0891B2"),
            ("raw_planning_clean_rate_avg", "raw clean", "#16A34A"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "h1m_visual_alias_repeat_burden.svg",
        title="H1m visual alias-repeat candidate burden",
        rows=_label_system_rows(h1m_visual_alias_repeat_rows),
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
    _write_grouped_metric_svg(
        figures_dir / "visual_catalog_live_candidate_replay_gate.svg",
        title="Visual catalog live replay gate",
        rows=catalog_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executable_rate", "candidate executable", "#059669"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "visual_catalog_argument_hints_live_candidate_replay_gate.svg",
        title="Visual catalog argument-hints live replay gate",
        rows=argument_hints_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executable_rate", "candidate executable", "#059669"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "visual_hard_slice_live_replay_gate.svg",
        title="Visual hard-slice live replay gate",
        rows=visual_hard_slice_live_replay_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executor_equivalence_rate", "candidate executor eq", "#059669"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "visual_hard_slice_stress_live_replay_gate.svg",
        title="Visual hard-slice stress live replay gate",
        rows=visual_hard_slice_stress_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executor_equivalence_rate", "candidate executor eq", "#059669"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "visual_hard_slice_alias_repeat_live_replay_gate.svg",
        title="Visual hard-slice alias-repeat live replay gate",
        rows=visual_hard_slice_alias_repeat_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executor_equivalence_rate", "candidate executor eq", "#059669"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "visual_hard_slice_alias_transfer_live_replay_gate.svg",
        title="Visual hard-slice alias-transfer live replay gate",
        rows=visual_hard_slice_alias_transfer_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executor_equivalence_rate", "candidate executor eq", "#059669"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "visual_hard_slice_alias_transfer_oracle_live_replay_gate.svg",
        title="Visual hard-slice alias-transfer oracle live replay gate",
        rows=visual_hard_slice_alias_transfer_oracle_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executor_equivalence_rate", "candidate executor eq", "#059669"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "visual_hard_slice_post_repair_live_replay_gate.svg",
        title="Visual hard-slice post-repair live replay gate",
        rows=visual_hard_slice_post_repair_live_summary_rows,
        label_field="comparison",
        metrics=[
            ("baseline_exact_rate", "baseline exact", "#2563EB"),
            ("candidate_exact_rate", "candidate exact", "#DC2626"),
            ("candidate_executor_equivalence_rate", "candidate executor eq", "#059669"),
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
        "prompt_contract_wave5_packet": str(Path(prompt_contract_wave5_packet).resolve()),
        "tool_catalog_profile_packet": str(Path(tool_catalog_profile_packet).resolve()),
        "tool_catalog_argument_hints_packet": str(Path(tool_catalog_argument_hints_packet).resolve()),
        "tool_catalog_argument_hints_vs_role_catalog_comparison": str(
            Path(tool_catalog_argument_hints_vs_role_catalog_comparison).resolve()
        ),
        "tool_catalog_split_selector_packet": str(Path(tool_catalog_split_selector_packet).resolve()),
        "tool_catalog_split_selector_vs_argument_hints_comparison": str(
            Path(tool_catalog_split_selector_vs_argument_hints_comparison).resolve()
        ),
        "tool_catalog_split_selector_vs_role_catalog_comparison": str(
            Path(tool_catalog_split_selector_vs_role_catalog_comparison).resolve()
        ),
        "tool_catalog_split_selector_live_decision": str(Path(tool_catalog_split_selector_live_decision).resolve()),
        "tool_catalog_schema_field_hints_packet": str(Path(tool_catalog_schema_field_hints_packet).resolve()),
        "tool_catalog_schema_field_hints_vs_argument_hints_comparison": str(
            Path(tool_catalog_schema_field_hints_vs_argument_hints_comparison).resolve()
        ),
        "tool_catalog_schema_field_hints_vs_split_selector_comparison": str(
            Path(tool_catalog_schema_field_hints_vs_split_selector_comparison).resolve()
        ),
        "tool_catalog_schema_field_hints_vs_role_catalog_comparison": str(
            Path(tool_catalog_schema_field_hints_vs_role_catalog_comparison).resolve()
        ),
        "tool_catalog_schema_field_hints_live_decision": str(Path(tool_catalog_schema_field_hints_live_decision).resolve()),
        "prompt_contract_wave6_packet": str(Path(prompt_contract_wave6_packet).resolve()),
        "visual_hard_slice_packet": str(Path(visual_hard_slice_packet).resolve()),
        "visual_hard_slice_exactness_diagnostic": str(Path(visual_hard_slice_exactness_diagnostic).resolve()),
        "h1i_prompt_contract_packet": str(Path(h1i_prompt_contract_packet).resolve()),
        "h1i_prompt_contract_repeat_packet": str(Path(h1i_prompt_contract_repeat_packet).resolve()),
        "h1j_prompt_contract_packet": str(Path(h1j_prompt_contract_packet).resolve()),
        "h1j_helper_packet": str(Path(h1j_helper_packet).resolve()),
        "h1k_parallel_audit_packet": str(Path(h1k_parallel_audit_packet).resolve()),
        "h1k_parallel_audit_helper_packet": str(Path(h1k_parallel_audit_helper_packet).resolve()),
        "h1l_visual_executor_equivalence_packet": str(Path(h1l_visual_executor_equivalence_packet).resolve()),
        "h1m_visual_alias_repeat_packet": str(Path(h1m_visual_alias_repeat_packet).resolve()),
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
        "catalog_live_visual_vs_no_directive_comparison": str(
            Path(catalog_live_visual_vs_no_directive_comparison).resolve()
        ),
        "catalog_live_visual_vs_visual_initiation_comparison": str(
            Path(catalog_live_visual_vs_visual_initiation_comparison).resolve()
        ),
        "catalog_live_visual_vs_visual_state_comparison": str(
            Path(catalog_live_visual_vs_visual_state_comparison).resolve()
        ),
        "argument_hints_live_visual_vs_no_directive_comparison": str(
            Path(argument_hints_live_visual_vs_no_directive_comparison).resolve()
        ),
        "argument_hints_live_visual_vs_contracted_comparison": str(
            Path(argument_hints_live_visual_vs_contracted_comparison).resolve()
        ),
        "argument_hints_live_visual_vs_role_catalog_comparison": str(
            Path(argument_hints_live_visual_vs_role_catalog_comparison).resolve()
        ),
        "visual_hard_slice_contracted_live_comparison": str(
            Path(visual_hard_slice_contracted_live_comparison).resolve()
        ),
        "visual_hard_slice_role_catalog_live_comparison": str(
            Path(visual_hard_slice_role_catalog_live_comparison).resolve()
        ),
        "visual_hard_slice_argument_hints_live_comparison": str(
            Path(visual_hard_slice_argument_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_live_replay_comparison": str(Path(visual_hard_slice_live_replay_comparison).resolve()),
        "visual_hard_slice_schema_literal_targets_live_comparison": str(
            Path(visual_hard_slice_schema_literal_targets_live_comparison).resolve()
        ),
        "visual_hard_slice_stress_contracted_live_comparison": str(
            Path(visual_hard_slice_stress_contracted_live_comparison).resolve()
        ),
        "visual_hard_slice_stress_role_catalog_live_comparison": str(
            Path(visual_hard_slice_stress_role_catalog_live_comparison).resolve()
        ),
        "visual_hard_slice_stress_argument_hints_live_comparison": str(
            Path(visual_hard_slice_stress_argument_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_stress_schema_field_hints_live_comparison": str(
            Path(visual_hard_slice_stress_schema_field_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_stress_schema_literal_targets_live_comparison": str(
            Path(visual_hard_slice_stress_schema_literal_targets_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_repeat_contracted_live_comparison": str(
            Path(visual_hard_slice_alias_repeat_contracted_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_repeat_role_catalog_live_comparison": str(
            Path(visual_hard_slice_alias_repeat_role_catalog_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_repeat_argument_hints_live_comparison": str(
            Path(visual_hard_slice_alias_repeat_argument_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_repeat_schema_field_hints_live_comparison": str(
            Path(visual_hard_slice_alias_repeat_schema_field_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_repeat_schema_literal_targets_live_comparison": str(
            Path(visual_hard_slice_alias_repeat_schema_literal_targets_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_contracted_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_contracted_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_role_catalog_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_role_catalog_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_argument_hints_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_argument_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_schema_field_hints_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_schema_field_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_schema_literal_targets_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_schema_literal_targets_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_oracle_contracted_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_oracle_contracted_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_oracle_role_catalog_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_oracle_role_catalog_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_oracle_argument_hints_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_oracle_argument_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_oracle_schema_field_hints_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_oracle_schema_field_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_alias_transfer_oracle_schema_literal_targets_live_comparison": str(
            Path(visual_hard_slice_alias_transfer_oracle_schema_literal_targets_live_comparison).resolve()
        ),
        "visual_hard_slice_post_repair_contracted_live_comparison": str(
            Path(visual_hard_slice_post_repair_contracted_live_comparison).resolve()
        ),
        "visual_hard_slice_post_repair_argument_hints_live_comparison": str(
            Path(visual_hard_slice_post_repair_argument_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_post_repair_code_hints_live_comparison": str(
            Path(visual_hard_slice_post_repair_code_hints_live_comparison).resolve()
        ),
        "visual_hard_slice_post_repair_code_guard_live_comparison": str(
            Path(visual_hard_slice_post_repair_code_guard_live_comparison).resolve()
        ),
        "registry_path": str(Path(registry_path).resolve()),
        "table_count": 70,
        "figure_count": 34,
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
        "prompt_contract_wave5_probe_gates": prompt_contract_wave5_gate_rows,
        "prompt_contract_wave5_probe_failure_modes": prompt_contract_wave5_failure_rows,
        "tool_catalog_profile_probe_gates": tool_catalog_profile_gate_rows,
        "tool_catalog_argument_hints_vs_role_catalog_comparison": tool_catalog_argument_hints_vs_role_catalog_payload,
        "tool_catalog_argument_hints_vs_role_catalog_case_deltas": tool_catalog_argument_hints_vs_role_catalog_case_rows,
        "tool_catalog_split_selector_vs_argument_hints_comparison": tool_catalog_split_selector_vs_argument_hints_payload,
        "tool_catalog_split_selector_vs_argument_hints_case_deltas": tool_catalog_split_selector_vs_argument_hints_case_rows,
        "tool_catalog_split_selector_vs_role_catalog_comparison": tool_catalog_split_selector_vs_role_catalog_payload,
        "tool_catalog_split_selector_vs_role_catalog_case_deltas": tool_catalog_split_selector_vs_role_catalog_case_rows,
        "tool_catalog_split_selector_live_replay_decision": tool_catalog_split_selector_live_decision_rows,
        "tool_catalog_schema_field_hints_vs_argument_hints_comparison": tool_catalog_schema_field_hints_vs_argument_hints_payload,
        "tool_catalog_schema_field_hints_vs_argument_hints_case_deltas": tool_catalog_schema_field_hints_vs_argument_hints_case_rows,
        "tool_catalog_schema_field_hints_vs_split_selector_comparison": tool_catalog_schema_field_hints_vs_split_selector_payload,
        "tool_catalog_schema_field_hints_vs_split_selector_case_deltas": tool_catalog_schema_field_hints_vs_split_selector_case_rows,
        "tool_catalog_schema_field_hints_vs_role_catalog_comparison": tool_catalog_schema_field_hints_vs_role_catalog_payload,
        "tool_catalog_schema_field_hints_vs_role_catalog_case_deltas": tool_catalog_schema_field_hints_vs_role_catalog_case_rows,
        "tool_catalog_schema_field_hints_live_replay_decision": tool_catalog_schema_field_hints_live_decision_rows,
        "prompt_contract_wave6_probe_gates": prompt_contract_wave6_gate_rows,
        "prompt_contract_wave6_probe_failure_modes": prompt_contract_wave6_failure_rows,
        "visual_hard_slice_probe_gates": visual_hard_slice_gate_rows,
        "visual_hard_slice_failure_modes": visual_hard_slice_failure_rows,
        "visual_hard_slice_family_summary": visual_hard_slice_family_rows,
        "visual_hard_slice_case_deltas_vs_no_directive": visual_hard_slice_case_deltas_vs_no_directive_rows,
        "visual_hard_slice_case_deltas_vs_contracted": visual_hard_slice_case_deltas_vs_contracted_rows,
        "visual_hard_slice_exactness_summary": visual_hard_slice_exactness_summary_rows,
        "visual_hard_slice_exactness_gaps": visual_hard_slice_exactness_gap_rows,
        "prompt_contract_promotion_decisions": prompt_contract_promotion_rows,
        "h1i_prompt_contract_candidate_metrics": h1i_prompt_contract_rows,
        "h1i_prompt_contract_repeat3_metrics": h1i_prompt_contract_repeat_rows,
        "h1j_probe_derived_candidate_metrics": h1j_prompt_contract_rows,
        "h1j_probe_derived_helper_metrics": h1j_helper_rows,
        "h1k_parallel_audit_candidate_metrics": h1k_parallel_audit_rows,
        "h1k_parallel_audit_helper_metrics": h1k_parallel_audit_helper_rows,
        "h1l_visual_executor_equivalence_candidate_metrics": h1l_visual_executor_equivalence_rows,
        "h1m_visual_alias_repeat_candidate_metrics": h1m_visual_alias_repeat_rows,
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
        "visual_catalog_live_candidate_replay_summary": catalog_live_summary_rows,
        "visual_catalog_live_candidate_case_deltas": catalog_live_case_rows,
        "visual_catalog_argument_hints_live_candidate_replay_summary": argument_hints_live_summary_rows,
        "visual_catalog_argument_hints_live_candidate_case_deltas": argument_hints_live_case_rows,
        "visual_hard_slice_live_replay_comparisons": [payload for _, payload in visual_hard_slice_live_comparisons],
        "visual_hard_slice_live_replay_summary": visual_hard_slice_live_replay_summary_rows,
        "visual_hard_slice_live_replay_case_deltas": visual_hard_slice_live_replay_case_rows,
        "visual_hard_slice_stress_live_replay_comparisons": [
            payload for _, payload in visual_hard_slice_stress_live_comparisons
        ],
        "visual_hard_slice_stress_live_replay_summary": visual_hard_slice_stress_live_summary_rows,
        "visual_hard_slice_stress_live_replay_case_deltas": visual_hard_slice_stress_live_case_rows,
        "visual_hard_slice_alias_repeat_live_replay_comparisons": [
            payload for _, payload in visual_hard_slice_alias_repeat_live_comparisons
        ],
        "visual_hard_slice_alias_repeat_live_replay_summary": visual_hard_slice_alias_repeat_live_summary_rows,
        "visual_hard_slice_alias_repeat_live_replay_case_deltas": visual_hard_slice_alias_repeat_live_case_rows,
        "visual_hard_slice_alias_transfer_live_replay_comparisons": [
            payload for _, payload in visual_hard_slice_alias_transfer_live_comparisons
        ],
        "visual_hard_slice_alias_transfer_live_replay_summary": visual_hard_slice_alias_transfer_live_summary_rows,
        "visual_hard_slice_alias_transfer_live_replay_case_deltas": visual_hard_slice_alias_transfer_live_case_rows,
        "visual_hard_slice_alias_transfer_oracle_live_replay_comparisons": [
            payload for _, payload in visual_hard_slice_alias_transfer_oracle_live_comparisons
        ],
        "visual_hard_slice_alias_transfer_oracle_live_replay_summary": (
            visual_hard_slice_alias_transfer_oracle_live_summary_rows
        ),
        "visual_hard_slice_alias_transfer_oracle_live_replay_case_deltas": (
            visual_hard_slice_alias_transfer_oracle_live_case_rows
        ),
        "visual_hard_slice_post_repair_live_replay_comparisons": [
            payload for _, payload in visual_hard_slice_post_repair_live_comparisons
        ],
        "visual_hard_slice_post_repair_live_replay_summary": (
            visual_hard_slice_post_repair_live_summary_rows
        ),
        "visual_hard_slice_post_repair_live_replay_case_deltas": (
            visual_hard_slice_post_repair_live_case_rows
        ),
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
    parser.add_argument("--prompt-contract-wave5-packet", default=str(DEFAULT_PROMPT_CONTRACT_WAVE5_PACKET))
    parser.add_argument("--tool-catalog-profile-packet", default=str(DEFAULT_TOOL_CATALOG_PROFILE_PACKET))
    parser.add_argument("--tool-catalog-argument-hints-packet", default=str(DEFAULT_TOOL_CATALOG_ARGUMENT_HINTS_PACKET))
    parser.add_argument(
        "--tool-catalog-argument-hints-vs-role-catalog-comparison",
        default=str(DEFAULT_TOOL_CATALOG_ARGUMENT_HINTS_VS_ROLE_CATALOG_COMPARISON),
    )
    parser.add_argument("--tool-catalog-split-selector-packet", default=str(DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_PACKET))
    parser.add_argument(
        "--tool-catalog-split-selector-vs-argument-hints-comparison",
        default=str(DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_VS_ARGUMENT_HINTS_COMPARISON),
    )
    parser.add_argument(
        "--tool-catalog-split-selector-vs-role-catalog-comparison",
        default=str(DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_VS_ROLE_CATALOG_COMPARISON),
    )
    parser.add_argument(
        "--tool-catalog-split-selector-live-decision",
        default=str(DEFAULT_TOOL_CATALOG_SPLIT_SELECTOR_LIVE_DECISION),
    )
    parser.add_argument("--tool-catalog-schema-field-hints-packet", default=str(DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_PACKET))
    parser.add_argument(
        "--tool-catalog-schema-field-hints-vs-argument-hints-comparison",
        default=str(DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_ARGUMENT_HINTS_COMPARISON),
    )
    parser.add_argument(
        "--tool-catalog-schema-field-hints-vs-split-selector-comparison",
        default=str(DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_SPLIT_SELECTOR_COMPARISON),
    )
    parser.add_argument(
        "--tool-catalog-schema-field-hints-vs-role-catalog-comparison",
        default=str(DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_VS_ROLE_CATALOG_COMPARISON),
    )
    parser.add_argument(
        "--tool-catalog-schema-field-hints-live-decision",
        default=str(DEFAULT_TOOL_CATALOG_SCHEMA_FIELD_HINTS_LIVE_DECISION),
    )
    parser.add_argument("--prompt-contract-wave6-packet", default=str(DEFAULT_PROMPT_CONTRACT_WAVE6_PACKET))
    parser.add_argument("--visual-hard-slice-packet", default=str(DEFAULT_VISUAL_HARD_SLICE_PACKET))
    parser.add_argument(
        "--visual-hard-slice-exactness-diagnostic",
        default=str(DEFAULT_VISUAL_HARD_SLICE_EXACTNESS_DIAGNOSTIC),
    )
    parser.add_argument("--h1i-prompt-contract-packet", default=str(DEFAULT_H1I_PROMPT_CONTRACT_PACKET))
    parser.add_argument("--h1i-prompt-contract-repeat-packet", default=str(DEFAULT_H1I_PROMPT_CONTRACT_REPEAT_PACKET))
    parser.add_argument("--h1j-prompt-contract-packet", default=str(DEFAULT_H1J_PROMPT_CONTRACT_PACKET))
    parser.add_argument("--h1j-helper-packet", default=str(DEFAULT_H1J_HELPER_PACKET))
    parser.add_argument("--h1k-parallel-audit-packet", default=str(DEFAULT_H1K_PARALLEL_AUDIT_PACKET))
    parser.add_argument(
        "--h1k-parallel-audit-helper-packet",
        default=str(DEFAULT_H1K_PARALLEL_AUDIT_HELPER_PACKET),
    )
    parser.add_argument(
        "--h1l-visual-executor-equivalence-packet",
        default=str(DEFAULT_H1L_VISUAL_EXECUTOR_EQUIVALENCE_PACKET),
    )
    parser.add_argument(
        "--h1m-visual-alias-repeat-packet",
        default=str(DEFAULT_H1M_VISUAL_ALIAS_REPEAT_PACKET),
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
    parser.add_argument(
        "--catalog-live-visual-vs-no-directive-comparison",
        default=str(DEFAULT_CATALOG_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON),
    )
    parser.add_argument(
        "--catalog-live-visual-vs-visual-initiation-comparison",
        default=str(DEFAULT_CATALOG_LIVE_VISUAL_VS_VISUAL_INITIATION_COMPARISON),
    )
    parser.add_argument(
        "--catalog-live-visual-vs-visual-state-comparison",
        default=str(DEFAULT_CATALOG_LIVE_VISUAL_VS_VISUAL_STATE_COMPARISON),
    )
    parser.add_argument(
        "--argument-hints-live-visual-vs-no-directive-comparison",
        default=str(DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_NO_DIRECTIVE_COMPARISON),
    )
    parser.add_argument(
        "--argument-hints-live-visual-vs-contracted-comparison",
        default=str(DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_CONTRACTED_COMPARISON),
    )
    parser.add_argument(
        "--argument-hints-live-visual-vs-role-catalog-comparison",
        default=str(DEFAULT_ARGUMENT_HINTS_LIVE_VISUAL_VS_ROLE_CATALOG_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-contracted-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_CONTRACTED_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-role-catalog-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ROLE_CATALOG_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-argument-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ARGUMENT_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-live-replay-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_LIVE_REPLAY_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-schema-literal-targets-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-stress-contracted-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_STRESS_CONTRACTED_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-stress-role-catalog-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_STRESS_ROLE_CATALOG_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-stress-argument-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_STRESS_ARGUMENT_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-stress-schema-field-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_STRESS_SCHEMA_FIELD_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-stress-schema-literal-targets-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_STRESS_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-repeat-contracted-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_CONTRACTED_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-repeat-role-catalog-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_ROLE_CATALOG_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-repeat-argument-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_ARGUMENT_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-repeat-schema-field-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_SCHEMA_FIELD_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-repeat-schema-literal-targets-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_REPEAT_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-contracted-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_CONTRACTED_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-role-catalog-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ROLE_CATALOG_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-argument-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ARGUMENT_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-schema-field-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_SCHEMA_FIELD_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-schema-literal-targets-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-oracle-contracted-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_CONTRACTED_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-oracle-role-catalog-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_ROLE_CATALOG_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-oracle-argument-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_ARGUMENT_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-oracle-schema-field-hints-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_SCHEMA_FIELD_HINTS_LIVE_COMPARISON),
    )
    parser.add_argument(
        "--visual-hard-slice-alias-transfer-oracle-schema-literal-targets-live-comparison",
        default=str(DEFAULT_VISUAL_HARD_SLICE_ALIAS_TRANSFER_ORACLE_SCHEMA_LITERAL_TARGETS_LIVE_COMPARISON),
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
        prompt_contract_wave5_packet=args.prompt_contract_wave5_packet,
        tool_catalog_profile_packet=args.tool_catalog_profile_packet,
        tool_catalog_argument_hints_packet=args.tool_catalog_argument_hints_packet,
        tool_catalog_argument_hints_vs_role_catalog_comparison=args.tool_catalog_argument_hints_vs_role_catalog_comparison,
        tool_catalog_split_selector_packet=args.tool_catalog_split_selector_packet,
        tool_catalog_split_selector_vs_argument_hints_comparison=args.tool_catalog_split_selector_vs_argument_hints_comparison,
        tool_catalog_split_selector_vs_role_catalog_comparison=args.tool_catalog_split_selector_vs_role_catalog_comparison,
        tool_catalog_split_selector_live_decision=args.tool_catalog_split_selector_live_decision,
        tool_catalog_schema_field_hints_packet=args.tool_catalog_schema_field_hints_packet,
        tool_catalog_schema_field_hints_vs_argument_hints_comparison=args.tool_catalog_schema_field_hints_vs_argument_hints_comparison,
        tool_catalog_schema_field_hints_vs_split_selector_comparison=args.tool_catalog_schema_field_hints_vs_split_selector_comparison,
        tool_catalog_schema_field_hints_vs_role_catalog_comparison=args.tool_catalog_schema_field_hints_vs_role_catalog_comparison,
        tool_catalog_schema_field_hints_live_decision=args.tool_catalog_schema_field_hints_live_decision,
        prompt_contract_wave6_packet=args.prompt_contract_wave6_packet,
        visual_hard_slice_packet=args.visual_hard_slice_packet,
        visual_hard_slice_exactness_diagnostic=args.visual_hard_slice_exactness_diagnostic,
        h1i_prompt_contract_packet=args.h1i_prompt_contract_packet,
        h1i_prompt_contract_repeat_packet=args.h1i_prompt_contract_repeat_packet,
        h1j_prompt_contract_packet=args.h1j_prompt_contract_packet,
        h1j_helper_packet=args.h1j_helper_packet,
        h1k_parallel_audit_packet=args.h1k_parallel_audit_packet,
        h1k_parallel_audit_helper_packet=args.h1k_parallel_audit_helper_packet,
        h1l_visual_executor_equivalence_packet=args.h1l_visual_executor_equivalence_packet,
        h1m_visual_alias_repeat_packet=args.h1m_visual_alias_repeat_packet,
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
        catalog_live_visual_vs_no_directive_comparison=args.catalog_live_visual_vs_no_directive_comparison,
        catalog_live_visual_vs_visual_initiation_comparison=args.catalog_live_visual_vs_visual_initiation_comparison,
        catalog_live_visual_vs_visual_state_comparison=args.catalog_live_visual_vs_visual_state_comparison,
        argument_hints_live_visual_vs_no_directive_comparison=args.argument_hints_live_visual_vs_no_directive_comparison,
        argument_hints_live_visual_vs_contracted_comparison=args.argument_hints_live_visual_vs_contracted_comparison,
        argument_hints_live_visual_vs_role_catalog_comparison=args.argument_hints_live_visual_vs_role_catalog_comparison,
        visual_hard_slice_contracted_live_comparison=args.visual_hard_slice_contracted_live_comparison,
        visual_hard_slice_role_catalog_live_comparison=args.visual_hard_slice_role_catalog_live_comparison,
        visual_hard_slice_argument_hints_live_comparison=args.visual_hard_slice_argument_hints_live_comparison,
        visual_hard_slice_live_replay_comparison=args.visual_hard_slice_live_replay_comparison,
        visual_hard_slice_schema_literal_targets_live_comparison=args.visual_hard_slice_schema_literal_targets_live_comparison,
        visual_hard_slice_stress_contracted_live_comparison=args.visual_hard_slice_stress_contracted_live_comparison,
        visual_hard_slice_stress_role_catalog_live_comparison=args.visual_hard_slice_stress_role_catalog_live_comparison,
        visual_hard_slice_stress_argument_hints_live_comparison=args.visual_hard_slice_stress_argument_hints_live_comparison,
        visual_hard_slice_stress_schema_field_hints_live_comparison=args.visual_hard_slice_stress_schema_field_hints_live_comparison,
        visual_hard_slice_stress_schema_literal_targets_live_comparison=args.visual_hard_slice_stress_schema_literal_targets_live_comparison,
        visual_hard_slice_alias_repeat_contracted_live_comparison=args.visual_hard_slice_alias_repeat_contracted_live_comparison,
        visual_hard_slice_alias_repeat_role_catalog_live_comparison=args.visual_hard_slice_alias_repeat_role_catalog_live_comparison,
        visual_hard_slice_alias_repeat_argument_hints_live_comparison=args.visual_hard_slice_alias_repeat_argument_hints_live_comparison,
        visual_hard_slice_alias_repeat_schema_field_hints_live_comparison=args.visual_hard_slice_alias_repeat_schema_field_hints_live_comparison,
        visual_hard_slice_alias_repeat_schema_literal_targets_live_comparison=args.visual_hard_slice_alias_repeat_schema_literal_targets_live_comparison,
        visual_hard_slice_alias_transfer_contracted_live_comparison=args.visual_hard_slice_alias_transfer_contracted_live_comparison,
        visual_hard_slice_alias_transfer_role_catalog_live_comparison=args.visual_hard_slice_alias_transfer_role_catalog_live_comparison,
        visual_hard_slice_alias_transfer_argument_hints_live_comparison=args.visual_hard_slice_alias_transfer_argument_hints_live_comparison,
        visual_hard_slice_alias_transfer_schema_field_hints_live_comparison=args.visual_hard_slice_alias_transfer_schema_field_hints_live_comparison,
        visual_hard_slice_alias_transfer_schema_literal_targets_live_comparison=args.visual_hard_slice_alias_transfer_schema_literal_targets_live_comparison,
        visual_hard_slice_alias_transfer_oracle_contracted_live_comparison=args.visual_hard_slice_alias_transfer_oracle_contracted_live_comparison,
        visual_hard_slice_alias_transfer_oracle_role_catalog_live_comparison=args.visual_hard_slice_alias_transfer_oracle_role_catalog_live_comparison,
        visual_hard_slice_alias_transfer_oracle_argument_hints_live_comparison=args.visual_hard_slice_alias_transfer_oracle_argument_hints_live_comparison,
        visual_hard_slice_alias_transfer_oracle_schema_field_hints_live_comparison=args.visual_hard_slice_alias_transfer_oracle_schema_field_hints_live_comparison,
        visual_hard_slice_alias_transfer_oracle_schema_literal_targets_live_comparison=args.visual_hard_slice_alias_transfer_oracle_schema_literal_targets_live_comparison,
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
                "tool_catalog_profile_id": controls.tool_catalog_profile_id,
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


def _live_decision_row(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "packet_run_id": manifest.get("packet_run_id", ""),
        "candidate_system_id": manifest.get("candidate_system_id", ""),
        "tool_catalog_profile_id": manifest.get("tool_catalog_profile_id", ""),
        "decision": manifest.get("decision", ""),
        "reason": manifest.get("reason", ""),
        "candidate_exact_match_rate": manifest.get("candidate_exact_match_rate", ""),
        "candidate_executable_match_rate": manifest.get("candidate_executable_match_rate", ""),
        "best_current_exact_candidate": manifest.get("best_current_exact_candidate", ""),
        "best_current_exact_candidate_rate": manifest.get("best_current_exact_candidate_rate", ""),
        "best_current_executable_routing_candidate": manifest.get("best_current_executable_routing_candidate", ""),
        "best_current_executable_routing_rate": manifest.get("best_current_executable_routing_rate", ""),
    }


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
                "baseline_executor_equivalence_rate": _none_to_blank(summary.get("baseline_executor_equivalence_rate")),
                "candidate_executor_equivalence_rate": _none_to_blank(summary.get("candidate_executor_equivalence_rate")),
                "delta_executor_equivalence_rate": _none_to_blank(summary.get("delta_executor_equivalence_rate")),
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
    wave5_rows: list[dict[str, Any]],
    wave6_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wave, gate_rows in [
        ("v1", wave1_rows),
        ("v2", wave2_rows),
        ("v3", wave3_rows),
        ("v4", wave4_rows),
        ("v5", wave5_rows),
        ("v6", wave6_rows),
    ]:
        for row in gate_rows:
            decision = _promotion_decision(row)
            rows.append(
                {
                    "wave": wave,
                    "tool_prompt_contract_id": row["tool_prompt_contract_id"],
                    "tool_catalog_profile_id": row.get("tool_catalog_profile_id", ""),
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
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_refine_selection": "visual refine",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog": "visual role catalog",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints": "catalog arg hints",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints": "catalog split selector",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints": "catalog schema fields",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets": "catalog schema target literals",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_literal_guard": "visual catalog literal",
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
    wave5_gate_rows = payload["prompt_contract_wave5_probe_gates"]
    catalog_profile_gate_rows = payload["tool_catalog_profile_probe_gates"]
    argument_hints_vs_catalog_case_rows = payload["tool_catalog_argument_hints_vs_role_catalog_case_deltas"]
    split_selector_vs_argument_hints_case_rows = payload["tool_catalog_split_selector_vs_argument_hints_case_deltas"]
    split_selector_vs_role_catalog_case_rows = payload["tool_catalog_split_selector_vs_role_catalog_case_deltas"]
    split_selector_live_decision_rows = payload["tool_catalog_split_selector_live_replay_decision"]
    schema_field_hints_vs_argument_hints_case_rows = payload["tool_catalog_schema_field_hints_vs_argument_hints_case_deltas"]
    schema_field_hints_vs_split_selector_case_rows = payload["tool_catalog_schema_field_hints_vs_split_selector_case_deltas"]
    schema_field_hints_live_decision_rows = payload["tool_catalog_schema_field_hints_live_replay_decision"]
    wave6_gate_rows = payload["prompt_contract_wave6_probe_gates"]
    visual_hard_slice_gate_rows = payload["visual_hard_slice_probe_gates"]
    visual_hard_slice_family_rows = payload["visual_hard_slice_family_summary"]
    visual_hard_slice_case_deltas_vs_no_directive = payload["visual_hard_slice_case_deltas_vs_no_directive"]
    visual_hard_slice_exactness_summary = payload["visual_hard_slice_exactness_summary"]
    visual_hard_slice_exactness_gaps = payload["visual_hard_slice_exactness_gaps"]
    h1i_prompt_contract_rows = payload["h1i_prompt_contract_candidate_metrics"]
    h1i_prompt_contract_repeat_rows = payload["h1i_prompt_contract_repeat3_metrics"]
    h1j_prompt_contract_rows = payload["h1j_probe_derived_candidate_metrics"]
    h1j_helper_rows = payload["h1j_probe_derived_helper_metrics"]
    h1k_parallel_audit_rows = payload["h1k_parallel_audit_candidate_metrics"]
    h1k_parallel_audit_helper_rows = payload["h1k_parallel_audit_helper_metrics"]
    h1l_visual_executor_equivalence_rows = payload["h1l_visual_executor_equivalence_candidate_metrics"]
    h1m_visual_alias_repeat_rows = payload["h1m_visual_alias_repeat_candidate_metrics"]
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
    catalog_live_summary_rows = payload["visual_catalog_live_candidate_replay_summary"]
    catalog_live_case_rows = payload["visual_catalog_live_candidate_case_deltas"]
    argument_hints_live_summary_rows = payload["visual_catalog_argument_hints_live_candidate_replay_summary"]
    argument_hints_live_case_rows = payload["visual_catalog_argument_hints_live_candidate_case_deltas"]
    visual_hard_slice_live_summary_rows = payload["visual_hard_slice_live_replay_summary"]
    visual_hard_slice_live_case_rows = payload["visual_hard_slice_live_replay_case_deltas"]
    visual_hard_slice_stress_live_summary_rows = payload["visual_hard_slice_stress_live_replay_summary"]
    visual_hard_slice_stress_live_case_rows = payload["visual_hard_slice_stress_live_replay_case_deltas"]
    visual_hard_slice_alias_repeat_live_summary_rows = payload[
        "visual_hard_slice_alias_repeat_live_replay_summary"
    ]
    visual_hard_slice_alias_repeat_live_case_rows = payload[
        "visual_hard_slice_alias_repeat_live_replay_case_deltas"
    ]
    visual_hard_slice_alias_transfer_live_summary_rows = payload[
        "visual_hard_slice_alias_transfer_live_replay_summary"
    ]
    visual_hard_slice_alias_transfer_live_case_rows = payload[
        "visual_hard_slice_alias_transfer_live_replay_case_deltas"
    ]
    visual_hard_slice_alias_transfer_oracle_live_summary_rows = payload[
        "visual_hard_slice_alias_transfer_oracle_live_replay_summary"
    ]
    visual_hard_slice_alias_transfer_oracle_live_case_rows = payload[
        "visual_hard_slice_alias_transfer_oracle_live_replay_case_deltas"
    ]
    visual_hard_slice_post_repair_live_summary_rows = payload[
        "visual_hard_slice_post_repair_live_replay_summary"
    ]
    visual_hard_slice_post_repair_live_case_rows = payload[
        "visual_hard_slice_post_repair_live_replay_case_deltas"
    ]
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
        "The visual catalog branch now includes an explicit negative-result loop. "
        "`visual_role_catalog_argument_hints_v2` remains the best focused-replay exact visual candidate. "
        "`visual_role_catalog_split_selector_hints_v3` was rejected before live replay because it regressed exact readback, `visual_role_catalog_schema_field_hints_v4` is the strongest fresh hard-slice profile, and `visual_role_catalog_schema_literal_targets_v5` is now negative evidence because it failed to fix the exact string misses while adding a wrong-tool regression.",
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
        "![Prompt contract wave five probe gate](figures/prompt_contract_wave5_probe_gate.svg)",
        "",
        "![Tool catalog profile probe gate](figures/tool_catalog_profile_probe_gate.svg)",
        "",
        "![Prompt contract wave six probe gate](figures/prompt_contract_wave6_probe_gate.svg)",
        "",
        "![Visual hard-slice probe gate](figures/visual_hard_slice_probe_gate.svg)",
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
        "![H1l visual executor-equivalence burden](figures/h1l_visual_executor_equivalence_burden.svg)",
        "",
        "![H1m visual alias-repeat burden](figures/h1m_visual_alias_repeat_burden.svg)",
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
        "![Visual catalog live replay gate](figures/visual_catalog_live_candidate_replay_gate.svg)",
        "",
        "![Visual catalog argument-hints live replay gate](figures/visual_catalog_argument_hints_live_candidate_replay_gate.svg)",
        "",
        "![Visual hard-slice live replay gate](figures/visual_hard_slice_live_replay_gate.svg)",
        "",
        "![Visual hard-slice stress live replay gate](figures/visual_hard_slice_stress_live_replay_gate.svg)",
        "",
        "![Visual hard-slice alias-repeat live replay gate](figures/visual_hard_slice_alias_repeat_live_replay_gate.svg)",
        "",
        "![Visual hard-slice alias-transfer live replay gate](figures/visual_hard_slice_alias_transfer_live_replay_gate.svg)",
        "",
        "![Visual hard-slice alias-transfer oracle live replay gate](figures/visual_hard_slice_alias_transfer_oracle_live_replay_gate.svg)",
        "",
        "![Visual hard-slice post-repair live replay gate](figures/visual_hard_slice_post_repair_live_replay_gate.svg)",
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
        "## Prompt-Contract Wave Five Probe Gate",
        "",
        _markdown_table(wave5_gate_rows),
        "",
        "`visual_refine_selection_v5` was more surgical: it targeted only latest-selection filtering and `refine_selection`. The raw probe rejected it before live replay: exact rate stayed `0.0`, executable rate stayed `0.0`, and the dominant failure shifted further toward `no_tool_call`. Under the current gate, this candidate should not spend CLI-live replay or H1 budget.",
        "",
        "## Tool-Catalog Profile Probe Gate",
        "",
        _markdown_table(catalog_profile_gate_rows),
        "",
        "`visual_role_catalog_v1` moves the intervention from standalone prompt-contract wording into the tool-catalog presentation. It keeps the exact directive disabled, improves raw exact rate from `0.0` to `0.125`, restores the visual executable target to `1.0`, and changes the live visual failure from wrong-tool/no-call into literal argument mismatch. `visual_role_catalog_argument_hints_v2` then tests the next narrow question: can field-level selector semantics fix that literal mismatch while preserving routing?",
        "",
        "## Tool-Catalog Argument-Hints vs Role-Catalog Probe Delta",
        "",
        _markdown_table(argument_hints_vs_catalog_case_rows),
        "",
        "The raw answer is mixed but materially informative. Argument hints raise probe exactness from `1 / 8` to `2 / 8` by making `visual_latest_filter_literal` exact, while preserving exact readback. The cost is that `visual_form_target_literal` drops from executable paraphrase to non-executable argument mismatch, so this is a candidate for visual referent exactness, not a complete visual recovery profile.",
        "",
        "## Tool-Catalog Split-Selector Negative Probe Delta",
        "",
        _markdown_table(split_selector_vs_argument_hints_case_rows),
        "",
        _markdown_table(split_selector_vs_role_catalog_case_rows),
        "",
        _markdown_table(split_selector_live_decision_rows),
        "",
        "`visual_role_catalog_split_selector_hints_v3` is useful as negative evidence. It preserved the v2 latest-filter exact call, but dropped overall raw exactness from `2 / 8` to `1 / 8` versus v2 and regressed readback by emitting `tool_name` instead of `name`. It also failed to recover the v1 executable form-target behavior, so focused live replay was intentionally skipped.",
        "",
        "## Tool-Catalog Schema-Field Negative Probe Delta",
        "",
        _markdown_table(schema_field_hints_vs_argument_hints_case_rows),
        "",
        _markdown_table(schema_field_hints_vs_split_selector_case_rows),
        "",
        _markdown_table(schema_field_hints_live_decision_rows),
        "",
        "`visual_role_catalog_schema_field_hints_v4` is cleaner than v3 because it avoids broad prose and restores the exact readback case. It still does not beat v2: raw exact stays `2 / 8`, executable visual-form recovery stays `0 / 1`, and the form-target case over-prefers `refine_selection` with `selection_id=\"latest\"`. Live replay was skipped because it tied the current best exact candidate while remaining below the executable routing baseline.",
        "",
        "## Prompt-Contract Wave Six Probe Gate",
        "",
        _markdown_table(wave6_gate_rows),
        "",
        "Wave six composes the visual role catalog with `literal_argument_guard_v1`. It keeps the same one-case exact gain but loses the catalog-only executable visual rescue and introduces no-call regressions on CLI/API cases. Treat it as a negative composition result: routing guidance and literal-copy wording interfere in this form.",
        "",
        "## Visual Hard-Slice Probe Gate",
        "",
        _markdown_table(visual_hard_slice_gate_rows),
        "",
        "The fresh visual hard slice breaks the earlier top-line saturation and gives a cleaner read on harness shape. Contracted MLX is the upper bound at `8 / 8` strict, executable, and executor-equivalent. The no-directive row falls to `1 / 8` on all three metrics, with no-tool-call as the dominant failure. The strongest no-directive profile remains `visual_role_catalog_schema_field_hints_v4`: `6 / 8` strict and `8 / 8` executor-equivalent. The attempted v5 target-literal repair drops to `5 / 8` strict and `7 / 8` executor-equivalent, so the hard-slice evidence now rejects that wording as an overcorrection rather than a promotion candidate.",
        "",
        "## Visual Hard-Slice Family Summary",
        "",
        _markdown_table(visual_hard_slice_family_rows),
        "",
        "The family breakdown explains the new signal. Schema-field hints preserve full executor-equivalent behavior across visible-region targeting, valid selection carryover, and region readback, but strict exactness still lags on visual argument-copying cases. The v5 target-literal repair did not improve that family and regressed the stale-selection decoy into a wrong-tool call, which suggests the next packaged H1 slice should treat strict protocol fidelity and executor-visible success as separate endpoints.",
        "",
        "## Visual Hard-Slice Exactness Diagnostic",
        "",
        _markdown_table(visual_hard_slice_exactness_summary),
        "",
        _markdown_table(visual_hard_slice_exactness_gaps),
        "",
        "The exactness diagnostic sharpens the v4/v5 interpretation. The two v4 non-exact rows are not executor-targeting failures: both reach the expected local visual regions, and the probe now scores them as executor-equivalent target matches. The v5 target-literal profile keeps those same two aliases and adds one true harness failure by choosing stale `refine_selection` instead of current-image `extract_layout`.",
        "",
        "## Visual Hard-Slice CLI-Live Replay",
        "",
        _markdown_table(visual_hard_slice_live_summary_rows),
        "",
        _markdown_table(visual_hard_slice_live_case_rows),
        "",
        "The live operator replay now preserves the hard-slice discriminator instead of smoothing it into staged packaged workflows. Contracted MLX is the upper bound at `2 / 2` strict and executor-equivalent. Role catalog v1 and argument hints v2 each recover only the stale-selection decoy (`1 / 2` strict, `1 / 2` executor-equivalent). Schema-field hints v4 keeps that exact stale-selection win and also recovers the metric-panel target as an executor-equivalent paraphrase (`1 / 2` strict, `2 / 2` executor-equivalent). Schema target literals v5 remains negative: `0 / 2` strict and `1 / 2` executor-equivalent, with the stale-selection decoy becoming a wrong-tool failure.",
        "",
        "## Visual Hard-Slice Stress CLI-Live Replay",
        "",
        _markdown_table(visual_hard_slice_stress_live_summary_rows),
        "",
        _markdown_table(visual_hard_slice_stress_live_case_rows),
        "",
        "The harder stress replay repeats the two mechanisms with fresh decoys. It no longer fully separates no-directive MLX: no-directive reaches `2 / 4` strict and `3 / 4` executor-equivalent. Contracted MLX remains the `4 / 4` strict upper bound. Schema-field hints v4 and schema target literals v5 do not improve strict exactness over no-directive (`2 / 4`), but both recover full executor-equivalence (`4 / 4`) by turning the hardest metric-panel decoy into an executor-valid selector alias. Role catalog v1 is negative on this stress slice, dropping to `1 / 4` strict and `2 / 4` executor-equivalent.",
        "",
        "## Visual Hard-Slice Alias-Repeat CLI-Live Replay",
        "",
        _markdown_table(visual_hard_slice_alias_repeat_live_summary_rows),
        "",
        _markdown_table(visual_hard_slice_alias_repeat_live_case_rows),
        "",
        "The eight-case alias-repeat packet makes the stress finding more publication-useful. No-directive MLX reaches `2 / 8` strict and `5 / 8` executor-equivalent. Schema-field hints v4 preserves the same strict count but improves executor-equivalence to `7 / 8`, while schema target literals v5 reaches `3 / 8` strict and full `8 / 8` executor-equivalence. Contracted MLX remains the strict upper bound at `7 / 8` and `8 / 8` executor-equivalent. Role catalog v1 and argument hints v2 are partial: they improve executor-equivalence to `6 / 8`, but do not match the schema-local profiles.",
        "",
        "## Visual Hard-Slice Alias-Transfer CLI-Live Replay",
        "",
        _markdown_table(visual_hard_slice_alias_transfer_live_summary_rows),
        "",
        _markdown_table(visual_hard_slice_alias_transfer_live_case_rows),
        "",
        "The six-case alias-transfer packet is the first post-packaging-gap discriminator. It uses fresh visual labels and decoys rather than repeating metric-panel/callout wording. No-directive MLX is `0 / 6` strict and `2 / 6` executor-equivalent. Argument hints v2 is the best executor-grounding row at `1 / 6` strict and `6 / 6` executor-equivalent. Schema target literals v5 reaches `1 / 6` strict and `4 / 6` executor-equivalent. Schema-field hints v4 improves strict exactness to `1 / 6` but does not improve executor-equivalence over no-directive. A follow-up contract-split diagnostic found that `5 / 6` generated expected-call contracts do not satisfy the packet's own expected-execution oracle. Contracted MLX's `5 / 6` strict score is therefore planner-call fidelity, not a clean target-success upper bound; it has `4` exact-but-not-executor rows. The publication-safe reading is that argument hints v2 is the H1n executor-target winner, and H1n should be rebuilt with oracle expected calls before strict exactness is used as a headline metric.",
        "",
        "## Visual Hard-Slice Alias-Transfer Oracle CLI-Live Replay",
        "",
        _markdown_table(visual_hard_slice_alias_transfer_oracle_live_summary_rows),
        "",
        _markdown_table(visual_hard_slice_alias_transfer_oracle_live_case_rows),
        "",
        "The oracle replay rebuild makes the packet's expected calls execute to the same visual targets as the packet's expected-execution oracle, and replay-live now preserves those serialized expected calls instead of recomputing planner-derived calls. That changes the H1n interpretation materially: no-directive MLX is `2 / 6` strict and executor-equivalent, contracted MLX falls to `1 / 6`, role catalog v1 reaches `3 / 6`, argument hints v2 reaches `5 / 6` strict and `6 / 6` executor-equivalent, schema-field hints v4 stays at `2 / 6`, and schema target literals v5 reaches `4 / 6`. The clean H1n winner is therefore argument hints, with schema target literals as the second-place transfer mechanism. Contracted prompting is not a useful upper bound on this oracle transfer slice.",
        "",
        "## Visual Hard-Slice Post-Repair CLI-Live Replay",
        "",
        _markdown_table(visual_hard_slice_post_repair_live_summary_rows),
        "",
        _markdown_table(visual_hard_slice_post_repair_live_case_rows),
        "",
        "The fresh post-repair holdout tests whether the v7 oblique code guard transfers beyond the repair packet. No-directive MLX is `2 / 8` strict and executor-equivalent, while contracted/default MLX is only `3 / 8`. Argument hints v2 and v6 code hints both reach `5 / 8`, but by different routes: argument hints is better on non-code labels, while code hints is better on code-like labels and stale-selection routing. The activation-gated v7 code guard is the current upper bound on this fresh packet at `6 / 8`, improving over no-directive by `+0.50`, over contracted/default by `+0.375`, and over both argument hints and v6 by `+0.125`. The remaining misses, `chip l90` and `status pill`, define the next held-out micro-slice.",
        "",
        "## Visual Hard-Slice Case Deltas vs No Directive",
        "",
        _markdown_table(visual_hard_slice_case_deltas_vs_no_directive),
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
        "## Visual Catalog CLI-Live Candidate Replay",
        "",
        _markdown_table(catalog_live_summary_rows),
        "",
        "`visual_role_catalog_v1` matches wave three's `1 / 3` exact ceiling, beats wave four on executable visual-form recovery, and converts the remaining latest-filter failure from `wrong_tool` to `argument_mismatch`. The next useful move is not more broad visual state wording; it is a narrow argument-literal mechanism that preserves the catalog routing win.",
        "",
        _markdown_table(catalog_live_case_rows),
        "",
        "## Visual Catalog Argument-Hints CLI-Live Candidate Replay",
        "",
        _markdown_table(argument_hints_live_summary_rows),
        "",
        "`visual_role_catalog_argument_hints_v2` is the first no-directive candidate to match contracted MLX on this focused visual exact replay: `2 / 3` exact. It fixes `visual_latest_filter_literal` exactly and preserves exact readback. The remaining gap is important: the candidate loses the contracted/v1 executable visual-form rescue, turning `visual_form_target_literal` into a non-executable argument mismatch. This is progress on selector literalness, but not yet a full replacement for controller-backed visual recovery.",
        "",
        _markdown_table(argument_hints_live_case_rows),
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
        "## H1l Visual Executor-Equivalence Candidate Packet",
        "",
        _markdown_table(h1l_visual_executor_equivalence_rows),
        "",
        "H1l promotes the visual hard-slice executor-equivalence result into five packaged visual live workflows. The packet is currently negative as a discriminator: contracted MLX, no-directive MLX, role catalog v1, argument hints v2, schema-field hints v4, and schema target literals v5 all tie at readiness `0.90406`, strict interface `0.85`, recovered execution `0.8`, raw clean `1.0`, and zero repair/fallback burden. That means the v4 hard-slice executor-equivalence gain is still a probe-level signal, not yet a packaged-workflow signal.",
        "",
        "## H1m Visual Alias-Repeat Candidate Packet",
        "",
        _markdown_table(h1m_visual_alias_repeat_rows),
        "",
        "H1m promotes the harder eight-case visual alias-repeat replay signal into three new packaged visual workflows. It is another negative packaged-workflow result: contracted MLX, no-directive MLX, role catalog v1, argument hints v2, schema-field hints v4, and schema target literals v5 all tie at readiness `0.87783`, strict interface `0.75`, recovered execution `0.667`, raw clean `1.0`, and zero repair/fallback/argument-repair burden. The replay-shaped alias-repeat signal is real, but this packaged surface is still too staged to attribute improvement.",
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
        "- The visual catalog path now gives a sharper positive result than the prompt-contract path: argument-hints cataloging reaches `2 / 3` live exact visual replay without the exact directive, but still misses executable form-target recovery.",
        "- The fresh visual hard slice updates that picture: schema-field hints preserve full executor-equivalent behavior on independently authored visual cases, but still trail contracted strict exactness.",
        "- The visual hard-slice live replay now confirms the same distinction in the CLI operator path when the raw case shape is preserved.",
        "- H1l then shows the current packaged visual workflows are too staged to preserve that distinction: all visual catalog rows tie on readiness, strict interface, recovered execution, raw clean rate, and controller burden.",
        "- H1m repeats that lesson on a harder alias-repeat packaged surface: even the rows that improved replay executor-equivalence tie once the task is staged into packaged workflows.",
        "- The next experiment should use preserved replay-shaped live packets, repeated alias packets, or less staged non-packaged live tasks before spending helper-ablation budget on current packaged H1 visual workflows.",
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
            f"- Prompt-contract wave five packet: `{payload['manifest']['prompt_contract_wave5_packet']}`",
            f"- Tool catalog profile packet: `{payload['manifest']['tool_catalog_profile_packet']}`",
            f"- Tool catalog argument-hints packet: `{payload['manifest']['tool_catalog_argument_hints_packet']}`",
            f"- Tool catalog argument-hints vs role-catalog comparison: `{payload['manifest']['tool_catalog_argument_hints_vs_role_catalog_comparison']}`",
            f"- Tool catalog split-selector packet: `{payload['manifest']['tool_catalog_split_selector_packet']}`",
            f"- Tool catalog split-selector vs argument-hints comparison: `{payload['manifest']['tool_catalog_split_selector_vs_argument_hints_comparison']}`",
            f"- Tool catalog split-selector vs role-catalog comparison: `{payload['manifest']['tool_catalog_split_selector_vs_role_catalog_comparison']}`",
            f"- Tool catalog split-selector live decision: `{payload['manifest']['tool_catalog_split_selector_live_decision']}`",
            f"- Tool catalog schema-field packet: `{payload['manifest']['tool_catalog_schema_field_hints_packet']}`",
            f"- Tool catalog schema-field vs argument-hints comparison: `{payload['manifest']['tool_catalog_schema_field_hints_vs_argument_hints_comparison']}`",
            f"- Tool catalog schema-field vs split-selector comparison: `{payload['manifest']['tool_catalog_schema_field_hints_vs_split_selector_comparison']}`",
            f"- Tool catalog schema-field vs role-catalog comparison: `{payload['manifest']['tool_catalog_schema_field_hints_vs_role_catalog_comparison']}`",
            f"- Tool catalog schema-field live decision: `{payload['manifest']['tool_catalog_schema_field_hints_live_decision']}`",
            f"- Prompt-contract wave six packet: `{payload['manifest']['prompt_contract_wave6_packet']}`",
            f"- Visual hard-slice packet: `{payload['manifest']['visual_hard_slice_packet']}`",
            f"- Visual hard-slice exactness diagnostic: `{payload['manifest']['visual_hard_slice_exactness_diagnostic']}`",
            f"- H1i prompt-contract packet: `{payload['manifest']['h1i_prompt_contract_packet']}`",
            f"- H1i prompt-contract repeat packet: `{payload['manifest']['h1i_prompt_contract_repeat_packet']}`",
            f"- H1j probe-derived prompt-contract packet: `{payload['manifest']['h1j_prompt_contract_packet']}`",
            f"- H1j probe-derived helper packet: `{payload['manifest']['h1j_helper_packet']}`",
            f"- H1k parallel-audit prompt-contract packet: `{payload['manifest']['h1k_parallel_audit_packet']}`",
            f"- H1k parallel-audit helper packet: `{payload['manifest']['h1k_parallel_audit_helper_packet']}`",
            f"- H1l visual executor-equivalence packet: `{payload['manifest']['h1l_visual_executor_equivalence_packet']}`",
            f"- H1m visual alias-repeat packet: `{payload['manifest']['h1m_visual_alias_repeat_packet']}`",
            f"- Exact replay comparison: `{payload['manifest']['exact_replay_comparison']}`",
            f"- Canonical argument replay comparison: `{payload['manifest']['canonical_argument_replay_comparison']}`",
            f"- Visual replay comparison: `{payload['manifest']['visual_replay_comparison']}`",
            f"- Parallel replay comparison: `{payload['manifest']['parallel_replay_comparison']}`",
            f"- CLI-live parallel replay comparison: `{payload['manifest']['live_parallel_replay_comparison']}`",
            f"- CLI-live visual replay comparison: `{payload['manifest']['live_visual_replay_comparison']}`",
            f"- CLI-live canonical replay comparison: `{payload['manifest']['live_canonical_replay_comparison']}`",
            f"- Wave four live visual vs no-directive comparison: `{payload['manifest']['wave4_live_visual_vs_no_directive_comparison']}`",
            f"- Wave four live visual vs contracted comparison: `{payload['manifest']['wave4_live_visual_vs_contracted_comparison']}`",
            f"- Argument-hints live visual vs no-directive comparison: `{payload['manifest']['argument_hints_live_visual_vs_no_directive_comparison']}`",
            f"- Argument-hints live visual vs contracted comparison: `{payload['manifest']['argument_hints_live_visual_vs_contracted_comparison']}`",
            f"- Argument-hints live visual vs role-catalog comparison: `{payload['manifest']['argument_hints_live_visual_vs_role_catalog_comparison']}`",
            f"- Visual hard-slice contracted live comparison: `{payload['manifest']['visual_hard_slice_contracted_live_comparison']}`",
            f"- Visual hard-slice role-catalog live comparison: `{payload['manifest']['visual_hard_slice_role_catalog_live_comparison']}`",
            f"- Visual hard-slice argument-hints live comparison: `{payload['manifest']['visual_hard_slice_argument_hints_live_comparison']}`",
            f"- Visual hard-slice live replay comparison: `{payload['manifest']['visual_hard_slice_live_replay_comparison']}`",
            f"- Visual hard-slice schema-literal-targets live comparison: `{payload['manifest']['visual_hard_slice_schema_literal_targets_live_comparison']}`",
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
