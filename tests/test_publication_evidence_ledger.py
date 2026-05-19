from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_publication_evidence_ledger.py"
SPEC = importlib.util.spec_from_file_location("build_publication_evidence_ledger_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_publication_evidence_ledger_writes_claims_and_sources(tmp_path: Path) -> None:
    payload = SCRIPT.build_ledger(output_dir=tmp_path)

    assert payload["manifest"]["claim_count"] >= 6
    assert payload["manifest"]["missing_source_count"] == 0

    claims = {row["claim_id"]: row for row in payload["claims"]}
    assert claims["C2_final_tool_directive_causal_for_protocol"]["status"] == "supported_current_packets"
    assert claims["C6_split_selector_wording_is_negative_evidence"]["status"] == "negative_result_current_packets"
    assert claims["C8_visual_hard_slice_targets_remaining_uncertainty"]["status"] == "supported_current_packets"
    assert claims["C9_schema_literal_targets_v5_is_negative_evidence"]["status"] == "negative_result_current_packets"
    assert claims["C10_v4_exact_misses_are_executor_success_aliases"]["status"] == "supported_current_packets"
    assert claims["C11_h1l_packaged_visual_workflows_remain_saturated"]["status"] == "negative_result_current_packets"
    assert claims["C12_replay_shaped_live_preserves_visual_hard_slice_signal"]["status"] == "supported_current_packets"
    assert claims["C13_visual_live_stress_separates_executor_grounding_from_strict_fidelity"][
        "status"
    ] == "supported_current_packets"
    assert claims["C14_h1m_packaged_alias_repeat_saturates"]["status"] == "negative_result_current_packets"
    assert claims["C15_packaged_visual_surfaces_wash_out_replay_discrimination"][
        "status"
    ] == "supported_current_packets"
    assert claims["C16_visual_alias_transfer_favors_argument_hints_executor_grounding"][
        "status"
    ] == "supported_current_packets"
    assert claims["C17_h1n_strict_exactness_matches_planner_not_oracle"][
        "status"
    ] == "benchmark_contract_issue_current_packets"
    assert claims["C18_h1n_oracle_transfer_identifies_argument_hints_as_clean_winner"][
        "status"
    ] == "supported_current_packets"
    assert claims["C19_h1n_argument_hints_gain_is_not_controller_helper_artifact"][
        "status"
    ] == "supported_current_packets"
    assert claims["C20_h1n_oracle_repeat_confirms_catalog_transfer_not_contracted_upper_bound"][
        "status"
    ] == "supported_current_packets"
    assert claims["C21_h1n_two_packet_oracle_synthesis_narrows_next_visual_question"][
        "status"
    ] == "supported_current_packets"
    assert claims["C22_h1n_oblique_labels_favor_argument_hints_over_schema_literals"][
        "status"
    ] == "supported_current_packets"
    assert claims["C23_h1n_oblique_argument_hints_misses_are_code_and_negation_errors"][
        "status"
    ] == "supported_current_packets"
    assert claims["C24_h1n_oblique_code_hints_repair_two_misses_with_one_regression"][
        "status"
    ] == "supported_current_packets"
    assert claims["C25_h1n_oblique_code_hints_is_localized_not_general"][
        "status"
    ] == "negative_result_current_packets"
    assert claims["C26_h1n_oblique_code_guard_fixes_v6_regression"][
        "status"
    ] == "supported_current_packets"
    assert claims["C27_h1n_code_guard_improves_v6_but_not_argument_hints"][
        "status"
    ] == "supported_current_packets"
    assert claims["C28_h1n_post_repair_holdout_favors_code_guard"][
        "status"
    ] == "supported_current_packets"
    assert claims["C29_h1n_residual_holdout_favors_hybrid_label_guard"][
        "status"
    ] == "supported_current_packets"
    assert claims["C30_component_value_guard_is_negative_evidence"][
        "status"
    ] == "negative_result_current_packets"
    assert claims["C31_no_call_control_rescue_is_current_component_value_upper_bound"][
        "status"
    ] == "supported_current_packets"
    assert claims["C32_no_call_rescue_is_scoped_not_general"][
        "status"
    ] == "supported_current_packets"
    assert claims["C33_h1o_factorial_identifies_component_value_residue"][
        "status"
    ] == "supported_current_packets"
    assert claims["C34_h1p_component_holdout_supports_component_value_domain"][
        "status"
    ] == "supported_current_packets"
    assert claims["C35_h1q_component_label_guard_is_strongest_transfer_candidate"][
        "status"
    ] == "supported_current_packets"
    assert claims["C36_h1s_residual_guard_is_targeted_not_global"][
        "status"
    ] == "supported_current_packets"
    assert claims["C37_h1x_breaks_v11_saturation_but_supports_routing"][
        "status"
    ] == "supported_current_packets"
    assert claims["C38_h2a_controller_stale_selection_gate_is_causal"][
        "status"
    ] == "supported_current_packets"
    assert claims["C39_h2a_stale_selection_gate_transfers_with_better_executor_profile"][
        "status"
    ] == "supported_current_packets"
    assert claims["C40_h2b_residual_exactness_favors_scoped_v12_not_global_h2a"][
        "status"
    ] == "supported_current_packets"
    assert claims["C41_h2c_scoped_residual_gate_saturates_h2b_but_needs_transfer"][
        "status"
    ] == "supported_current_packets"
    assert claims["C42_h2d_class_preserving_route_repairs_h2c_transfer_but_costs_h2b_exactness"][
        "status"
    ] == "supported_current_packets"
    assert claims["C43_h2e_route_arbitration_reconciles_h2c_h2d_tradeoff"][
        "status"
    ] == "supported_current_packets"
    assert claims["C44_h2f_holdout_breaks_h2e_global_promotion"][
        "status"
    ] == "negative_result_current_packets"
    assert claims["C45_h2g_component_identity_contract_is_partial_executor_gain"][
        "status"
    ] == "negative_result_current_packets"
    assert claims["C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer"][
        "status"
    ] == "supported_scoped_negative_global_promotion"
    assert claims["C47_h2i_conditional_component_arbitration_does_not_preserve_h2f_repair"][
        "status"
    ] == "negative_result_current_packets"
    assert claims["C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"][
        "status"
    ] == "supported_current_packets_next_harder_holdout"
    assert claims["C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"][
        "status"
    ] == "supported_current_packets_helper_ablation_passed"
    assert claims["C50_h2l_overreach_holdout_supports_target_normalization_scope"][
        "status"
    ] == "supported_current_packets_next_harder_holdout"
    assert claims["C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"][
        "status"
    ] == "negative_result_current_packets"
    assert claims["C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"][
        "status"
    ] == "supported_current_packets_scope_candidate"
    assert claims["C53_h2o_value_bearing_target_synthesis_repairs_h2m_strict_with_contextual_alias_residue"][
        "status"
    ] == "supported_current_packets_scope_candidate"
    assert claims["C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression"][
        "status"
    ] == "supported_current_packets_scope_candidate"
    assert claims["C55_h2q_composed_surface_value_stale_breaks_h2p_saturation"][
        "status"
    ] == "supported_current_packets_boundary"
    assert claims["C56_h2r_composed_route_gating_solves_h2q_locally"][
        "status"
    ] == "supported_current_packets_transfer_backtested"
    assert claims["C57_h2r_transfer_backtest_preserves_current_gates"][
        "status"
    ] == "supported_current_packets_transfer_positive_requires_fresh_holdout"
    assert claims["C58_h2s_fresh_holdout_confirms_h2r_composed_route_gating"][
        "status"
    ] == "supported_fresh_holdout_requires_h2t_or_packaged_transfer"
    assert claims["C59_h2t_overreach_independence_breaks_h2r_via_negation_scope_normalization"][
        "status"
    ] == "supported_fresh_holdout_requires_h2u_negation_aware_normalization"
    assert claims["C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"][
        "status"
    ] == "supported_current_full_transfer_needs_harder_semantic_holdout"
    assert claims["C61_h2v_semantic_negation_breaks_h2u_transfer_saturation"][
        "status"
    ] == "supported_fresh_semantic_holdout_needs_h2w_repair"
    assert claims["C62_h2w_semantic_target_preservation_repairs_h2v"][
        "status"
    ] == "supported_h2v_repair_transfer_backtested_separately"
    assert claims["C63_h2w_transfer_backtest_preserves_current_gates"][
        "status"
    ] == "supported_replay_transfer_clean_packaged_workflow_unproven"
    assert claims["C64_h2x_cli_semantic_pressure_separates_semantic_preservation_from_fallback"][
        "status"
    ] == "supported_packaged_cli_gate_fallback_independent"
    assert claims["C65_h2y_scaled_cli_semantic_pressure_exposes_h2w_boundary"][
        "status"
    ] == "supported_boundary_result_fallback_independent"
    assert claims["C66_h2z_boundary_ablation_closes_h2y_with_separable_controller_helpers"][
        "status"
    ] == "supported_boundary_ablation_requires_harder_holdout"
    assert claims["C67_h3_cli_controller_holdout_blocks_h2z_global_promotion"][
        "status"
    ] == "supported_fresh_holdout_negative_global_promotion"
    assert claims["C68_h3a_controller_repair_closes_h3_with_separable_helpers"][
        "status"
    ] == "supported_fresh_holdout_factorial_repair"
    assert claims["C69_h3a_preserves_h2z_h2y_transfer_gate"][
        "status"
    ] == "supported_first_transfer_regression_gate"
    assert claims["C70_h3a_preserves_h2w_transfer_backcompat_gate"][
        "status"
    ] == "supported_broad_internal_transfer_gate"
    assert "7/8" in claims["C2_final_tool_directive_causal_for_protocol"]["primary_metric"]
    assert "8/8 strict and 8/8 executor-equivalent versus H2p at 3/8" in claims[
        "C56_h2r_composed_route_gating_solves_h2q_locally"
    ]["primary_metric"]
    assert "81/81 strict and 81/81 executor-equivalent" in claims[
        "C57_h2r_transfer_backtest_preserves_current_gates"
    ]["primary_metric"]
    assert "10/10 strict and 10/10 executor-equivalent" in claims[
        "C58_h2s_fresh_holdout_confirms_h2r_composed_route_gating"
    ]["primary_metric"]
    assert "H2r/H2p/H2o/H2j all reach 8/10 strict" in claims[
        "C59_h2t_overreach_independence_breaks_h2r_via_negation_scope_normalization"
    ]["primary_metric"]
    assert "2 H2r misses are raw-exact outputs rewritten" in claims[
        "C59_h2t_overreach_independence_breaks_h2r_via_negation_scope_normalization"
    ]["primary_metric"]
    assert "improves H2t from H2r's 8/10 strict" in claims[
        "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
    ]["primary_metric"]
    assert "preserves 26/26 strict exactness" in claims[
        "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
    ]["primary_metric"]
    assert "another 39/39 strict exactness" in claims[
        "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
    ]["primary_metric"]
    assert "34/34 across H1y/H1o/H1p" in claims[
        "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
    ]["primary_metric"]
    assert "99/99" in claims[
        "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
    ]["primary_metric"]
    assert "H2u reaches 4/10 strict and 5/10 executor-equivalent" in claims[
        "C61_h2v_semantic_negation_breaks_h2u_transfer_saturation"
    ]["primary_metric"]
    assert "both stale-example rows and all three genuine negated-target rows" in claims[
        "C61_h2v_semantic_negation_breaks_h2u_transfer_saturation"
    ]["primary_metric"]
    assert "H2w reaches 10/10 strict and 10/10 executor-equivalent" in claims[
        "C62_h2w_semantic_target_preservation_repairs_h2v"
    ]["primary_metric"]
    assert "+0.60 exact-rate and +0.50 executor-equivalence-rate" in claims[
        "C62_h2w_semantic_target_preservation_repairs_h2v"
    ]["primary_metric"]
    assert "109/109 strict exactness and 109/109 executor-equivalence" in claims[
        "C63_h2w_transfer_backtest_preserves_current_gates"
    ]["primary_metric"]
    assert "zero strict regressions versus H2u" in claims[
        "C63_h2w_transfer_backtest_preserves_current_gates"
    ]["primary_metric"]
    assert "H2u reaches 3/8 strict and 4/8 executor-equivalent" in claims[
        "C64_h2x_cli_semantic_pressure_separates_semantic_preservation_from_fallback"
    ]["primary_metric"]
    assert "Matched no-fallback controls have 0.0 exact and executor-equivalence deltas" in claims[
        "C64_h2x_cli_semantic_pressure_separates_semantic_preservation_from_fallback"
    ]["primary_metric"]
    assert "H2u reaches 4/16 strict and 5/16 executor-equivalent" in claims[
        "C65_h2y_scaled_cli_semantic_pressure_exposes_h2w_boundary"
    ]["primary_metric"]
    assert "H2w reaches 12/16 strict and 12/16 executor-equivalent" in claims[
        "C65_h2y_scaled_cli_semantic_pressure_exposes_h2w_boundary"
    ]["primary_metric"]
    assert "all three stale-selection negation rows remain wrong-tool failures" in claims[
        "C65_h2y_scaled_cli_semantic_pressure_exposes_h2w_boundary"
    ]["limitation"]
    assert "H2z stale-selection negation alone reaches 15/16" in claims[
        "C66_h2z_boundary_ablation_closes_h2y_with_separable_controller_helpers"
    ]["primary_metric"]
    assert "H2z combined reaches 16/16 strict and 16/16 executor-equivalent" in claims[
        "C66_h2z_boundary_ablation_closes_h2y_with_separable_controller_helpers"
    ]["primary_metric"]
    assert "not yet evidence that the combined helper should be globally promoted" in claims[
        "C66_h2z_boundary_ablation_closes_h2y_with_separable_controller_helpers"
    ]["limitation"]
    assert "all reach 15/20 strict exactness and 15/20 executor-equivalence" in claims[
        "C67_h3_cli_controller_holdout_blocks_h2z_global_promotion"
    ]["primary_metric"]
    assert "0 fixed cases, and 0 H2z helper interventions" in claims[
        "C67_h3_cli_controller_holdout_blocks_h2z_global_promotion"
    ]["primary_metric"]
    assert "varies workflow family" in claims[
        "C67_h3_cli_controller_holdout_blocks_h2z_global_promotion"
    ]["limitation"]
    assert "H3a combined reaches 20/20" in claims[
        "C68_h3a_controller_repair_closes_h3_with_separable_helpers"
    ]["primary_metric"]
    assert "4 stale-paraphrase and 1 negative-value helper interventions" in claims[
        "C68_h3a_controller_repair_closes_h3_with_separable_helpers"
    ]["primary_metric"]
    assert "H3a combined also reaches 16/16" in claims[
        "C69_h3a_preserves_h2z_h2y_transfer_gate"
    ]["primary_metric"]
    assert "0.0 exact and executor-equivalence deltas versus H2z" in claims[
        "C69_h3a_preserves_h2z_h2y_transfer_gate"
    ]["primary_metric"]
    assert "109/109 strict" in claims[
        "C70_h3a_preserves_h2w_transfer_backcompat_gate"
    ]["primary_metric"]
    assert "0 H3a-specific helper interventions" in claims[
        "C70_h3a_preserves_h2w_transfer_backcompat_gate"
    ]["primary_metric"]
    assert "v3 raw exact falls" in claims["C6_split_selector_wording_is_negative_evidence"]["primary_metric"]
    assert "schema-field hints reach 6/8 strict and 8/8 executor-equivalent" in claims[
        "C8_visual_hard_slice_targets_remaining_uncertainty"
    ]["primary_metric"]
    assert "v5 reaches 5/8 strict and 7/8 executor-equivalent" in claims[
        "C9_schema_literal_targets_v5_is_negative_evidence"
    ]["primary_metric"]
    assert "true harness failure count is 0" in claims["C10_v4_exact_misses_are_executor_success_aliases"]["primary_metric"]
    assert "H1l candidate rows tie" in claims[
        "C11_h1l_packaged_visual_workflows_remain_saturated"
    ]["primary_metric"]
    assert "schema-field hints is the strongest no-directive row" in claims[
        "C12_replay_shaped_live_preserves_visual_hard_slice_signal"
    ]["primary_metric"]
    assert "schema-field hints and schema target literals are 2/4 strict but 4/4 executor-equivalent" in claims[
        "C13_visual_live_stress_separates_executor_grounding_from_strict_fidelity"
    ]["primary_metric"]
    assert "improves executor-equivalence from 5/8 to 7/8" in claims[
        "C13_visual_live_stress_separates_executor_grounding_from_strict_fidelity"
    ]["primary_metric"]
    assert "schema target literals reach 3/8 strict and 8/8 executor-equivalent" in claims[
        "C13_visual_live_stress_separates_executor_grounding_from_strict_fidelity"
    ]["primary_metric"]
    assert "H1m candidate rows tie at readiness 0.87783" in claims[
        "C14_h1m_packaged_alias_repeat_saturates"
    ]["primary_metric"]
    assert "zero controller repair/fallback/argument repair" in claims[
        "C14_h1m_packaged_alias_repeat_saturates"
    ]["primary_metric"]
    assert "2/2 visual promotion surfaces" in claims[
        "C15_packaged_visual_surfaces_wash_out_replay_discrimination"
    ]["primary_metric"]
    assert "H1m max replay executor-equivalence delta 0.375" in claims[
        "C15_packaged_visual_surfaces_wash_out_replay_discrimination"
    ]["primary_metric"]
    assert "argument hints v2 is 1/6 strict and 6/6 executor-equivalent" in claims[
        "C16_visual_alias_transfer_favors_argument_hints_executor_grounding"
    ]["primary_metric"]
    assert "contracted MLX is 5/6 strict but 1/6 executor-equivalent" in claims[
        "C16_visual_alias_transfer_favors_argument_hints_executor_grounding"
    ]["primary_metric"]
    assert "5/6 generated expected-call contracts fail" in claims[
        "C17_h1n_strict_exactness_matches_planner_not_oracle"
    ]["primary_metric"]
    assert "4 exact-but-not-executor rows" in claims[
        "C17_h1n_strict_exactness_matches_planner_not_oracle"
    ]["primary_metric"]
    assert "argument hints v2 is 5/6 exact and 6/6 executor-equivalent" in claims[
        "C18_h1n_oracle_transfer_identifies_argument_hints_as_clean_winner"
    ]["primary_metric"]
    assert "contracted is 1/6" in claims[
        "C18_h1n_oracle_transfer_identifies_argument_hints_as_clean_winner"
    ]["primary_metric"]
    assert "Argument hints remains 5/6 exact and 6/6 executor-equivalent" in claims[
        "C19_h1n_argument_hints_gain_is_not_controller_helper_artifact"
    ]["primary_metric"]
    assert "0.0 exact and executor-equivalence deltas" in claims[
        "C19_h1n_argument_hints_gain_is_not_controller_helper_artifact"
    ]["primary_metric"]
    assert "argument hints v2 and schema target literals v5 are 5/6 exact" in claims[
        "C20_h1n_oracle_repeat_confirms_catalog_transfer_not_contracted_upper_bound"
    ]["primary_metric"]
    assert "contracted is 0/6" in claims[
        "C20_h1n_oracle_repeat_confirms_catalog_transfer_not_contracted_upper_bound"
    ]["primary_metric"]
    assert "argument hints is executor-equivalent in both packets at 6/6 and 6/6" in claims[
        "C21_h1n_two_packet_oracle_synthesis_narrows_next_visual_question"
    ]["primary_metric"]
    assert "schema target literals rises from 4/6 to 6/6" in claims[
        "C21_h1n_two_packet_oracle_synthesis_narrows_next_visual_question"
    ]["primary_metric"]
    assert "argument hints v2 is 4/6" in claims[
        "C22_h1n_oblique_labels_favor_argument_hints_over_schema_literals"
    ]["primary_metric"]
    assert "schema target literals v5 is 0/6" in claims[
        "C22_h1n_oblique_labels_favor_argument_hints_over_schema_literals"
    ]["primary_metric"]
    assert "`cell r42` is truncated to `cell`" in claims[
        "C23_h1n_oblique_argument_hints_misses_are_code_and_negation_errors"
    ]["primary_metric"]
    assert "`alert p55` is replaced with the negated decoy `consent toggle`" in claims[
        "C23_h1n_oblique_argument_hints_misses_are_code_and_negation_errors"
    ]["primary_metric"]
    assert "Oblique code hints reaches 5/6 exact and executor-equivalent" in claims[
        "C24_h1n_oblique_code_hints_repair_two_misses_with_one_regression"
    ]["primary_metric"]
    assert "loses `field e19` as a wrong-tool case" in claims[
        "C24_h1n_oblique_code_hints_repair_two_misses_with_one_regression"
    ]["primary_metric"]
    assert any(
        row["claim_id"] == "C24_h1n_oblique_code_hints_repair_two_misses_with_one_regression"
        and row["path"] == "results/reports/h1n_oblique_code_hints_delta/diagnostic.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert "argument hints has 14/18 exact and 16/18 executor-equivalent successes" in claims[
        "C25_h1n_oblique_code_hints_is_localized_not_general"
    ]["primary_metric"]
    assert "code hints improves only the oblique packet" in claims[
        "C25_h1n_oblique_code_hints_is_localized_not_general"
    ]["primary_metric"]
    assert "Oblique code guard v7 reaches 6/6 exact and 6/6 executor-equivalent" in claims[
        "C26_h1n_oblique_code_guard_fixes_v6_regression"
    ]["primary_metric"]
    assert "over v6 code hints by +0.167" in claims[
        "C26_h1n_oblique_code_guard_fixes_v6_regression"
    ]["primary_metric"]
    assert "code guard reaches 14/18 exact and 15/18 executor-equivalent" in claims[
        "C27_h1n_code_guard_improves_v6_but_not_argument_hints"
    ]["primary_metric"]
    assert "argument hints remains 14/18 exact and 16/18 executor-equivalent" in claims[
        "C27_h1n_code_guard_improves_v6_but_not_argument_hints"
    ]["primary_metric"]
    assert "code guard reaches 6/8 exact and executor-equivalent" in claims[
        "C28_h1n_post_repair_holdout_favors_code_guard"
    ]["primary_metric"]
    assert "argument hints at 5/8" in claims[
        "C28_h1n_post_repair_holdout_favors_code_guard"
    ]["primary_metric"]
    assert "v8 hybrid label guard reaches 7/8 exact" in claims[
        "C29_h1n_residual_holdout_favors_hybrid_label_guard"
    ]["primary_metric"]
    assert "argument hints at 5/8 exact and 7/8 executor-equivalent" in claims[
        "C29_h1n_residual_holdout_favors_hybrid_label_guard"
    ]["primary_metric"]
    assert "argument hints v2 and hybrid label guard v8 both reach 6/8 exact" in claims[
        "C30_component_value_guard_is_negative_evidence"
    ]["primary_metric"]
    assert "v9 component-value guard falls to 4/8 exact" in claims[
        "C30_component_value_guard_is_negative_evidence"
    ]["primary_metric"]
    assert "v10 reaches 7/8 exact and 8/8 executor-equivalent" in claims[
        "C31_no_call_control_rescue_is_current_component_value_upper_bound"
    ]["primary_metric"]
    assert "+0.125 exact/+0.125 executor-equivalence" in claims[
        "C31_no_call_control_rescue_is_current_component_value_upper_bound"
    ]["primary_metric"]
    assert "v10 reaches 22/30 exact" in claims[
        "C32_no_call_rescue_is_scoped_not_general"
    ]["primary_metric"]
    assert "trails incumbents at 25/30 exact and 26/30 executor-equivalent" in claims[
        "C32_no_call_rescue_is_scoped_not_general"
    ]["primary_metric"]
    assert "argument hints v2 and component-value guard v9 tie the strict upper bound" in claims[
        "C33_h1o_factorial_identifies_component_value_residue"
    ]["primary_metric"]
    assert "no-directive is already 4/4 exact on activation/no-call" in claims[
        "C33_h1o_factorial_identifies_component_value_residue"
    ]["primary_metric"]
    assert "component-value guard v9 reaches 10/12 exact and 11/12" in claims[
        "C34_h1p_component_holdout_supports_component_value_domain"
    ]["primary_metric"]
    assert "argument hints v2 at 6/12" in claims[
        "C34_h1p_component_holdout_supports_component_value_domain"
    ]["primary_metric"]
    assert "component-label guard v11 reaches 26/32 exact and 29/32" in claims[
        "C35_h1q_component_label_guard_is_strongest_transfer_candidate"
    ]["primary_metric"]
    assert "component-value guard v9 at 23/32 exact and 25/32" in claims[
        "C35_h1q_component_label_guard_is_strongest_transfer_candidate"
    ]["primary_metric"]
    assert "v12 improves strict exactness from v11's 26/32 to 27/32" in claims[
        "C36_h1s_residual_guard_is_targeted_not_global"
    ]["primary_metric"]
    assert "lowers executor-equivalence from v11's 29/32 to 27/32" in claims[
        "C36_h1s_residual_guard_is_targeted_not_global"
    ]["primary_metric"]
    assert "v12 reaches 8/8" in claims[
        "C37_h1x_breaks_v11_saturation_but_supports_routing"
    ]["primary_metric"]
    assert "H2a reaches 8/10" in claims[
        "C38_h2a_controller_stale_selection_gate_is_causal"
    ]["primary_metric"]
    assert "H2a reaches 35/40 strict exact and 38/40 executor-equivalent" in claims[
        "C39_h2a_stale_selection_gate_transfers_with_better_executor_profile"
    ]["primary_metric"]
    assert "v12 reaches 4/5 strict exact and 4/5 executor-equivalent" in claims[
        "C40_h2b_residual_exactness_favors_scoped_v12_not_global_h2a"
    ]["primary_metric"]
    assert "H2a reaches 0/5 strict and 3/5 executor-equivalent" in claims[
        "C40_h2b_residual_exactness_favors_scoped_v12_not_global_h2a"
    ]["primary_metric"]
    assert "H2c reaches 5/5 strict exact and 5/5 executor-equivalent" in claims[
        "C41_h2c_scoped_residual_gate_saturates_h2b_but_needs_transfer"
    ]["primary_metric"]
    assert "H2d reaches 8/8 strict exact and 8/8 executor-equivalent on H1x" in claims[
        "C42_h2d_class_preserving_route_repairs_h2c_transfer_but_costs_h2b_exactness"
    ]["primary_metric"]
    assert "H2d reaches 4/5 strict exact on H2b versus H2c at 5/5" in claims[
        "C42_h2d_class_preserving_route_repairs_h2c_transfer_but_costs_h2b_exactness"
    ]["primary_metric"]
    assert "H2e reaches 5/5 strict exact and 5/5 executor-equivalent on H2b" in claims[
        "C43_h2e_route_arbitration_reconciles_h2c_h2d_tradeoff"
    ]["primary_metric"]
    assert "zero non-exact rows" in claims[
        "C43_h2e_route_arbitration_reconciles_h2c_h2d_tradeoff"
    ]["primary_metric"]
    assert "H2e reaches 6/10 strict exact and 6/10 executor-equivalent" in claims[
        "C44_h2f_holdout_breaks_h2e_global_promotion"
    ]["primary_metric"]
    assert "ties H2c at 6/10" in claims[
        "C44_h2f_holdout_breaks_h2e_global_promotion"
    ]["primary_metric"]
    assert "component-identity query contract" in claims[
        "C44_h2f_holdout_breaks_h2e_global_promotion"
    ]["next_test"]
    assert "H2g stays at 6/10 strict exact" in claims[
        "C45_h2g_component_identity_contract_is_partial_executor_gain"
    ]["primary_metric"]
    assert "improves executor-equivalence from 6/10 to 7/10" in claims[
        "C45_h2g_component_identity_contract_is_partial_executor_gain"
    ]["primary_metric"]
    assert "Build H2h with explicit negative examples" in claims[
        "C45_h2g_component_identity_contract_is_partial_executor_gain"
    ]["next_test"]
    assert "H2h improves H2f from H2e/H2g's 6/10 strict exactness to 9/10" in claims[
        "C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer"
    ]["primary_metric"]
    assert "falls to 3/5 on H2b" in claims[
        "C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer"
    ]["primary_metric"]
    assert "conditional arbitration profile" in claims[
        "C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer"
    ]["next_test"]
    assert "H2i reaches 6/10 strict" in claims[
        "C47_h2i_conditional_component_arbitration_does_not_preserve_h2f_repair"
    ]["primary_metric"]
    assert "`result tile` -> `result tile for Blocked`" in claims[
        "C47_h2i_conditional_component_arbitration_does_not_preserve_h2f_repair"
    ]["primary_metric"]
    assert "stopped at the H2f gate" in claims[
        "C47_h2i_conditional_component_arbitration_does_not_preserve_h2f_repair"
    ]["limitation"]
    assert "H2j reaches 10/10 strict and executor-equivalent on H2f" in claims[
        "C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"
    ]["primary_metric"]
    assert "5/5 on H2b and 8/8 on H1x" in claims[
        "C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"
    ]["primary_metric"]
    assert "same visual label appears as both a requested target" in claims[
        "C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"
    ]["limitation"]
    assert "Build an H2k harder holdout" in claims[
        "C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"
    ]["next_test"]
    assert "H2j reaches 8/8 strict and executor-equivalent" in claims[
        "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
    ]["primary_metric"]
    assert "H2e reaches 3/8 strict with 6/8 executor-equivalent" in claims[
        "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
    ]["primary_metric"]
    assert "5 target-query-normalization interventions and 0 stale-selection" in claims[
        "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
    ]["primary_metric"]
    assert "matched H2j-without-stale-selection ablation also reaches 8/8" in claims[
        "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
    ]["primary_metric"]
    assert "remaining limitation is broader transfer and over-normalization pressure" in claims[
        "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
    ]["limitation"]
    assert "target-query-normalization overreach" in claims[
        "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
    ]["next_test"]
    assert "full H2j and H2j without the stale-selection gate both reach 8/8" in claims[
        "C50_h2l_overreach_holdout_supports_target_normalization_scope"
    ]["primary_metric"]
    assert "`critical chip` to `status badge`" in claims[
        "C50_h2l_overreach_holdout_supports_target_normalization_scope"
    ]["primary_metric"]
    assert "explicit target-is wording" in claims[
        "C50_h2l_overreach_holdout_supports_target_normalization_scope"
    ]["limitation"]
    assert "Build H2m with less direct target phrasing" in claims[
        "C50_h2l_overreach_holdout_supports_target_normalization_scope"
    ]["next_test"]
    assert "full H2j and H2j without stale-selection both reach 3/8 strict" in claims[
        "C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"
    ]["primary_metric"]
    assert "3 value-bearing over-strip rows" in claims[
        "C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"
    ]["primary_metric"]
    assert "less-direct packet" in claims[
        "C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"
    ]["limitation"]
    assert "Build H2n as a scoped target-normalization policy" in claims[
        "C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"
    ]["next_test"]
    assert "ties H2j at 3/8 strict exactness but improves executor-equivalence from 3/8 to 5/8" in claims[
        "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
    ]["primary_metric"]
    assert "preserves H2k at 8/8, H2l at 8/8, and H2f at 10/10" in claims[
        "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
    ]["primary_metric"]
    assert "not a canonical target-query construction policy" in claims[
        "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
    ]["limitation"]
    assert "Build H2o as a canonical value-bearing target-query synthesis gate" in claims[
        "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
    ]["next_test"]
    assert "improves strict exactness from H2n's 3/8 to 7/8" in claims[
        "C53_h2o_value_bearing_target_synthesis_repairs_h2m_strict_with_contextual_alias_residue"
    ]["primary_metric"]
    assert "remaining H2m miss is not a value-bearing label construction miss" in claims[
        "C53_h2o_value_bearing_target_synthesis_repairs_h2m_strict_with_contextual_alias_residue"
    ]["limitation"]
    assert "Build H2p as a contextual surface-type alias routing slice" in claims[
        "C53_h2o_value_bearing_target_synthesis_repairs_h2m_strict_with_contextual_alias_residue"
    ]["next_test"]
    assert "from H2o's 7/8 to 8/8" in claims[
        "C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression"
    ]["primary_metric"]
    assert "preserves H2k at 8/8, H2l at 8/8, and H2f at 10/10" in claims[
        "C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression"
    ]["primary_metric"]
    assert "surface-class aliases" in claims[
        "C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression"
    ]["limitation"]
    assert "Define a harder post-H2p H1/H2 slice" in claims[
        "C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression"
    ]["next_test"]
    assert "3/8 strict and 3/8 executor-equivalent" in claims[
        "C55_h2q_composed_surface_value_stale_breaks_h2p_saturation"
    ]["primary_metric"]
    assert "H2o reaches 2/8" in claims[
        "C55_h2q_composed_surface_value_stale_breaks_h2p_saturation"
    ]["primary_metric"]
    assert "H2r around composed route gating" in claims[
        "C55_h2q_composed_surface_value_stale_breaks_h2p_saturation"
    ]["next_test"]
    assert any(
        row["claim_id"] == "C28_h1n_post_repair_holdout_favors_code_guard"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_post_repair_live_replay_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C29_h1n_residual_holdout_favors_hybrid_label_guard"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_residual_live_replay_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C30_component_value_guard_is_negative_evidence"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_component_value_live_replay_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C31_no_call_control_rescue_is_current_component_value_upper_bound"
        and row["path"]
        == "results/tool_probe_replay_live/20260510T_h1n_component_value_no_call_control_rescue_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C32_no_call_rescue_is_scoped_not_general"
        and row["path"] == "results/reports/h1n_no_call_rescue_transfer_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C33_h1o_factorial_identifies_component_value_residue"
        and row["path"] == "results/reports/h1o_control_factorial_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C34_h1p_component_holdout_supports_component_value_domain"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1p_live_replay_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C35_h1q_component_label_guard_is_strongest_transfer_candidate"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/h1q_component_label_guard_aggregate_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C36_h1s_residual_guard_is_targeted_not_global"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/h1s_component_residual_transfer_aggregate.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C37_h1x_breaks_v11_saturation_but_supports_routing"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/h1x_v11_breaker_packet_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C38_h2a_controller_stale_selection_gate_is_causal"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/h1y_routed_residual_packet_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C39_h2a_stale_selection_gate_transfers_with_better_executor_profile"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/h2a_stale_selection_transfer_aggregate_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C39_h2a_stale_selection_gate_transfers_with_better_executor_profile"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/figures/h2a_stale_selection_transfer_gate.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C40_h2b_residual_exactness_favors_scoped_v12_not_global_h2a"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/h2b_residual_exactness_packet_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C40_h2b_residual_exactness_favors_scoped_v12_not_global_h2a"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/figures/h2b_residual_exactness_gate.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C41_h2c_scoped_residual_gate_saturates_h2b_but_needs_transfer"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/tables/h2c_scoped_residual_packet_summary.csv"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C41_h2c_scoped_residual_gate_saturates_h2b_but_needs_transfer"
        and row["path"]
        == "results/reports/mlx_tool_contract_harnessing/figures/h2c_scoped_residual_gate.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C42_h2d_class_preserving_route_repairs_h2c_transfer_but_costs_h2b_exactness"
        and row["path"] == "results/reports/h2d_transfer_tradeoff_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C42_h2d_class_preserving_route_repairs_h2c_transfer_but_costs_h2b_exactness"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_h2c_on_h1x_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C43_h2e_route_arbitration_reconciles_h2c_h2d_tradeoff"
        and row["path"] == "results/reports/h2e_route_arbitration_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C43_h2e_route_arbitration_reconciles_h2c_h2d_tradeoff"
        and row["path"] == "results/reports/h2e_route_arbitration_synthesis/figures/h2e_route_arbitration_gate.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C44_h2f_holdout_breaks_h2e_global_promotion"
        and row["path"] == "results/reports/h2f_route_arbitration_holdout_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C44_h2f_holdout_breaks_h2e_global_promotion"
        and row["path"]
        == "results/reports/h2f_route_arbitration_holdout_synthesis/figures/h2f_holdout_profile_bars.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C44_h2f_holdout_breaks_h2e_global_promotion"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260510T_h2f_route_arbitration_h2e_vs_no_directive_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C45_h2g_component_identity_contract_is_partial_executor_gain"
        and row["path"]
        == "results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C45_h2g_component_identity_contract_is_partial_executor_gain"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260510T_h2g_component_identity_query_contract_vs_h2e_on_h2f_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer"
        and row["path"] == "results/reports/h2h_component_identity_tradeoff_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h2b_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h1x_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C47_h2i_conditional_component_arbitration_does_not_preserve_h2f_repair"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2h_on_h2f_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C47_h2i_conditional_component_arbitration_does_not_preserve_h2f_repair"
        and row["path"] == "results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"
        and row["path"] == "results/reports/h2j_target_query_normalization_transfer_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"
        and row["path"]
        == "results/reports/h2j_target_query_normalization_transfer_synthesis/figures/h2j_transfer_gate.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2e_on_h2f_v2"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2h_on_h1x_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
        and row["path"] == "results/reports/h2k_target_decoy_overlap_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
        and row["path"]
        == "results/reports/h2k_target_decoy_overlap_synthesis/figures/h2k_target_decoy_overlap_gate.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_h2h_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
        and row["path"]
        == "results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_no_stale_gate_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_no_stale_gate_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C50_h2l_overreach_holdout_supports_target_normalization_scope"
        and row["path"] == "results/reports/h2l_target_normalization_overreach_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C50_h2l_overreach_holdout_supports_target_normalization_scope"
        and row["path"]
        == "results/reports/h2l_target_normalization_overreach_synthesis/figures/h2l_target_normalization_overreach_gate.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C50_h2l_overreach_holdout_supports_target_normalization_scope"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2l_target_normalization_overreach_h2j_vs_h2e_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C50_h2l_overreach_holdout_supports_target_normalization_scope"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2l_target_normalization_overreach_h2j_vs_no_stale_gate_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"
        and row["path"] == "results/reports/h2m_less_direct_overreach_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"
        and row["path"]
        == "results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2j_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_h2e_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_no_stale_gate_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"]
        == "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
        and row["path"] == "results/reports/h2n_scoped_target_normalization_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"]
        == "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
        and row["path"]
        == "results/reports/h2n_scoped_target_normalization_synthesis/figures/h2n_scoped_target_normalization_gate.svg"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"]
        == "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
        and row["path"] == "results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2m_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"]
        == "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2m_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"]
        == "C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2f_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"]
        == "C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression"
        and row["path"] == "results/reports/h2p_contextual_surface_alias_routing_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"]
        == "C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression"
        and row["path"] == "results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2m_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"]
        == "C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2m_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C55_h2q_composed_surface_value_stale_breaks_h2p_saturation"
        and row["path"] == "results/reports/h2q_composed_surface_value_stale_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C55_h2q_composed_surface_value_stale_breaks_h2p_saturation"
        and row["path"] == "results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2p_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C55_h2q_composed_surface_value_stale_breaks_h2p_saturation"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2q_composed_surface_value_stale_h2p_vs_h2o_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C58_h2s_fresh_holdout_confirms_h2r_composed_route_gating"
        and row["path"] == "results/reports/h2s_fresh_composed_holdout_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C58_h2s_fresh_holdout_confirms_h2r_composed_route_gating"
        and row["path"] == "results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2r_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C58_h2s_fresh_holdout_confirms_h2r_composed_route_gating"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2s_fresh_composed_holdout_h2r_vs_h2p_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C59_h2t_overreach_independence_breaks_h2r_via_negation_scope_normalization"
        and row["path"] == "results/reports/h2t_overreach_independence_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C59_h2t_overreach_independence_breaks_h2r_via_negation_scope_normalization"
        and row["path"] == "results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2r_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C59_h2t_overreach_independence_breaks_h2r_via_negation_scope_normalization"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2e_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
        and row["path"] == "results/reports/h2u_negation_guard_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
        and row["path"] == "results/tool_probe_replay_live/20260513T_h2t_overreach_independence_h2u_execute_v2"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260513T_h2u_negation_guard_vs_h2r_on_h2t_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
        and row["path"] == "results/tool_probe_replay_live/20260513T_h2u_negation_guard_on_h2k_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260513T_h2u_negation_guard_vs_h2r_on_h1x_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
        and row["path"] == "results/tool_probe_replay_live/20260513T_h2u_negation_guard_on_h1y_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260513T_h2u_negation_guard_vs_h2r_on_h1p_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C61_h2v_semantic_negation_breaks_h2u_transfer_saturation"
        and row["path"] == "results/reports/h2v_semantic_negation_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C61_h2v_semantic_negation_breaks_h2u_transfer_saturation"
        and row["path"] == "results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2u_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C61_h2v_semantic_negation_breaks_h2u_transfer_saturation"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260513T_h2v_semantic_negation_h2u_vs_h2r_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C62_h2w_semantic_target_preservation_repairs_h2v"
        and row["path"] == "results/reports/h2w_semantic_target_preservation_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C62_h2w_semantic_target_preservation_repairs_h2v"
        and row["path"] == "results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2w_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C62_h2w_semantic_target_preservation_repairs_h2v"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260513T_h2v_semantic_negation_h2w_vs_h2u_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C63_h2w_transfer_backtest_preserves_current_gates"
        and row["path"] == "results/reports/h2w_transfer_backtest_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C63_h2w_transfer_backtest_preserves_current_gates"
        and row["path"] == "results/tool_probe_replay_live/20260513T_h2w_semantic_target_preservation_on_h1p_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C63_h2w_transfer_backtest_preserves_current_gates"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2t_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C64_h2x_cli_semantic_pressure_separates_semantic_preservation_from_fallback"
        and row["path"] == "results/reports/h2x_cli_semantic_pressure_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C64_h2x_cli_semantic_pressure_separates_semantic_preservation_from_fallback"
        and row["path"] == "results/tool_probe_replay_live/20260517T_h2x_cli_semantic_pressure_h2w_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C64_h2x_cli_semantic_pressure_separates_semantic_preservation_from_fallback"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260517T_h2x_cli_semantic_pressure_h2u_no_fallback_vs_h2u_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C65_h2y_scaled_cli_semantic_pressure_exposes_h2w_boundary"
        and row["path"] == "results/reports/h2y_scaled_cli_semantic_pressure_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C65_h2y_scaled_cli_semantic_pressure_exposes_h2w_boundary"
        and row["path"] == "results/tool_probe_replay_live/20260519T_h2y_scaled_cli_semantic_pressure_h2w_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C65_h2y_scaled_cli_semantic_pressure_exposes_h2w_boundary"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260519T_h2y_scaled_cli_semantic_pressure_h2w_vs_h2u_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C66_h2z_boundary_ablation_closes_h2y_with_separable_controller_helpers"
        and row["path"] == "results/reports/h2z_boundary_ablation_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C66_h2z_boundary_ablation_closes_h2y_with_separable_controller_helpers"
        and row["path"] == "results/tool_probe_replay_live/20260519T_h2y_scaled_cli_semantic_pressure_h2z_combined_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C66_h2z_boundary_ablation_closes_h2y_with_separable_controller_helpers"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260519T_h2y_scaled_cli_semantic_pressure_h2z_combined_vs_h2w_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C68_h3a_controller_repair_closes_h3_with_separable_helpers"
        and row["path"] == "results/reports/h3a_controller_repair_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C68_h3a_controller_repair_closes_h3_with_separable_helpers"
        and row["path"] == "results/tool_probe_replay_live/20260519T_h3_cli_controller_holdout_h3a_combined_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C69_h3a_preserves_h2z_h2y_transfer_gate"
        and row["path"] == "results/reports/h3a_h2y_transfer_gate_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C69_h3a_preserves_h2z_h2y_transfer_gate"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260519T_h2y_scaled_cli_semantic_pressure_h3a_combined_vs_h2z_combined_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C70_h3a_preserves_h2w_transfer_backcompat_gate"
        and row["path"] == "results/reports/h3a_transfer_backtest_synthesis/report.md"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C70_h3a_preserves_h2w_transfer_backcompat_gate"
        and row["path"] == "results/tool_probe_replay_live/20260519T_h3a_boundary_combined_on_h1p_execute_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )
    assert any(
        row["claim_id"] == "C70_h3a_preserves_h2w_transfer_backcompat_gate"
        and row["path"]
        == "results/tool_probe_replay_live_comparisons/20260519T_h3a_boundary_combined_vs_h2w_on_h2t_v1"
        and row["exists"]
        for row in payload["evidence_sources"]
    )

    source_types = {row["artifact_type"] for row in payload["evidence_sources"]}
    assert "h1_ablation_packet" in source_types
    assert "visual_hard_slice_probe_packet" in source_types
    assert "visual_hard_slice_profile_comparison" in source_types
    assert "visual_hard_slice_exactness_diagnostic" in source_types
    assert "diagnostic_report" in source_types
    assert "design_packet" in source_types
    assert "live_replay_decision" in source_types
    assert all(row["exists"] for row in payload["evidence_sources"])

    assert (tmp_path / "ledger.md").exists()
    assert (tmp_path / "ledger.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "claim_ledger.csv").exists()
    assert (tmp_path / "tables" / "evidence_sources.csv").exists()
