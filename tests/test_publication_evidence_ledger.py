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
    assert "7/8" in claims["C2_final_tool_directive_causal_for_protocol"]["primary_metric"]
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
