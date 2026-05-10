# H2f Route Arbitration Holdout Synthesis

Generated: `2026-05-10T22:01:38.599299+00:00`

## Summary

H2f is the fresh holdout that was supposed to test whether H2e route arbitration generalized beyond the saturated H2b/H1x gates. It does not. H2e keeps a large advantage over the no-directive floor, but it ties H2c and fails four cases by calling the right tool with the wrong query. The residual problem is component-identity binding under displayed-value decoys. H2g improves executor-equivalence by one row but does not improve strict exactness. H2h then repairs most of the fresh holdout at 9/10 strict and executor-equivalent, leaving only the state marker alias. H2i conditionalization does not preserve that lift: it returns to 6/10, tying H2e while trailing H2h by three rows. This makes H2h a strong scoped repair and H2i a negative prompt-conditionalization result.

![H2f holdout profile bars](figures/h2f_holdout_profile_bars.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executable_success_count | executable_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_no_directive_execute_v1 | 10 | 1 | 0.1 | 1 | 0.1 | 1 | 0.1 |
| h2a_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2a_execute_v1 | 10 | 4 | 0.4 | 4 | 0.4 | 4 | 0.4 |
| component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_component_residual_guard_execute_v1 | 10 | 5 | 0.5 | 6 | 0.6 | 6 | 0.6 |
| h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2c_execute_v1 | 10 | 6 | 0.6 | 6 | 0.6 | 6 | 0.6 |
| h2d_class_preserving_route | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_class_preserving_residual_route_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2d_execute_v1 | 10 | 5 | 0.5 | 5 | 0.5 | 5 | 0.5 |
| h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1 | 10 | 6 | 0.6 | 6 | 0.6 | 6 | 0.6 |
| h2g_component_identity_query_contract | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_query_contract_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1 | 10 | 6 | 0.6 | 7 | 0.7 | 7 | 0.7 |
| h2h_component_identity_negative_examples | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1 | 10 | 9 | 0.9 | 9 | 0.9 | 9 | 0.9 |
| h2i_conditional_component_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_component_identity_arbitration_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1 | 10 | 6 | 0.6 | 6 | 0.6 | 6 | 0.6 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executable_rate | candidate_executable_rate | delta_executable_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2f_h2e_vs_h2c | results/tool_probe_replay_live_comparisons/20260510T_h2f_route_arbitration_h2e_vs_h2c_v1 | 10 | 0.6 | 0.6 | 0.0 | 0.6 | 0.6 | 0.0 | 0.6 | 0.6 | 0.0 |
| h2f_h2e_vs_h2d | results/tool_probe_replay_live_comparisons/20260510T_h2f_route_arbitration_h2e_vs_h2d_v1 | 10 | 0.5 | 0.6 | 0.09999999999999998 | 0.5 | 0.6 | 0.09999999999999998 | 0.5 | 0.6 | 0.09999999999999998 |
| h2f_h2e_vs_h2a | results/tool_probe_replay_live_comparisons/20260510T_h2f_route_arbitration_h2e_vs_h2a_v1 | 10 | 0.4 | 0.6 | 0.19999999999999996 | 0.4 | 0.6 | 0.19999999999999996 | 0.4 | 0.6 | 0.19999999999999996 |
| h2f_h2e_vs_component_residual_guard | results/tool_probe_replay_live_comparisons/20260510T_h2f_route_arbitration_h2e_vs_component_residual_guard_v1 | 10 | 0.5 | 0.6 | 0.09999999999999998 | 0.6 | 0.6 | 0.0 | 0.6 | 0.6 | 0.0 |
| h2f_h2e_vs_no_directive | results/tool_probe_replay_live_comparisons/20260510T_h2f_route_arbitration_h2e_vs_no_directive_v1 | 10 | 0.1 | 0.6 | 0.5 | 0.1 | 0.6 | 0.5 | 0.1 | 0.6 | 0.5 |
| h2f_h2g_vs_h2e | results/tool_probe_replay_live_comparisons/20260510T_h2g_component_identity_query_contract_vs_h2e_on_h2f_v1 | 10 | 0.6 | 0.6 | 0.0 | 0.6 | 0.7 | 0.09999999999999998 | 0.6 | 0.7 | 0.09999999999999998 |
| h2f_h2g_vs_h2c | results/tool_probe_replay_live_comparisons/20260510T_h2g_component_identity_query_contract_vs_h2c_on_h2f_v1 | 10 | 0.6 | 0.6 | 0.0 | 0.6 | 0.7 | 0.09999999999999998 | 0.6 | 0.7 | 0.09999999999999998 |
| h2f_h2g_vs_no_directive | results/tool_probe_replay_live_comparisons/20260510T_h2g_component_identity_query_contract_vs_no_directive_on_h2f_v1 | 10 | 0.1 | 0.6 | 0.5 | 0.1 | 0.7 | 0.6 | 0.1 | 0.7 | 0.6 |
| h2f_h2h_vs_h2e | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h2f_v1 | 10 | 0.6 | 0.9 | 0.30000000000000004 | 0.6 | 0.9 | 0.30000000000000004 | 0.6 | 0.9 | 0.30000000000000004 |
| h2f_h2h_vs_h2g | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2g_on_h2f_v1 | 10 | 0.6 | 0.9 | 0.30000000000000004 | 0.7 | 0.9 | 0.20000000000000007 | 0.7 | 0.9 | 0.20000000000000007 |
| h2f_h2h_vs_h2c | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2c_on_h2f_v1 | 10 | 0.6 | 0.9 | 0.30000000000000004 | 0.6 | 0.9 | 0.30000000000000004 | 0.6 | 0.9 | 0.30000000000000004 |
| h2f_h2h_vs_no_directive | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_no_directive_on_h2f_v1 | 10 | 0.1 | 0.9 | 0.8 | 0.1 | 0.9 | 0.8 | 0.1 | 0.9 | 0.8 |
| h2f_h2i_vs_h2e | results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2e_on_h2f_v1 | 10 | 0.6 | 0.6 | 0.0 | 0.6 | 0.6 | 0.0 | 0.6 | 0.6 | 0.0 |
| h2f_h2i_vs_h2h | results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2h_on_h2f_v1 | 10 | 0.9 | 0.6 | -0.30000000000000004 | 0.9 | 0.6 | -0.30000000000000004 | 0.9 | 0.6 | -0.30000000000000004 |

## H2e Non-Exact Rows

| profile_label | packet_dir | case_id | family | failure_mode | executable_match | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query | query_error_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2e_route_arbitration | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1 | h2f_result_tile_comment_value_decoy | h2f_route_component_class_transfer | argument_mismatch | False | False | extract_layout | result tile | extract_layout | Blocked | value_or_alias_query_substitution |
| h2e_route_arbitration | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1 | h2f_resolution_badge_log_result_decoy | h2f_route_component_class_transfer | argument_mismatch | False | False | extract_layout | resolution badge | extract_layout | Deferred | value_or_alias_query_substitution |
| h2e_route_arbitration | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1 | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | argument_mismatch | False | False | extract_layout | state marker | extract_layout | lifecycle state marker | value_or_alias_query_substitution |
| h2e_route_arbitration | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1 | h2f_mode_switch_note_value_decoy | h2f_route_nonstandard_class | argument_mismatch | False | False | extract_layout | mode switch | extract_layout | mode toggle | value_or_alias_query_substitution |

## H2g Non-Exact Rows

| profile_label | packet_dir | case_id | family | failure_mode | executable_match | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query | query_error_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2g_component_identity_query_contract | results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1 | h2f_result_tile_comment_value_decoy | h2f_route_component_class_transfer | argument_mismatch | False | False | extract_layout | result tile | extract_layout | Blocked | value_or_alias_query_substitution |
| h2g_component_identity_query_contract | results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1 | h2f_resolution_badge_log_result_decoy | h2f_route_component_class_transfer | executable_paraphrase | True | True | extract_layout | resolution badge | extract_layout | resolution badge Deferred | value_or_alias_query_substitution |
| h2g_component_identity_query_contract | results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1 | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | argument_mismatch | False | False | extract_layout | state marker | extract_layout | lifecycle state marker | value_or_alias_query_substitution |
| h2g_component_identity_query_contract | results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1 | h2f_mode_switch_note_value_decoy | h2f_route_nonstandard_class | argument_mismatch | False | False | extract_layout | mode switch | extract_layout | mode toggle | value_or_alias_query_substitution |

## H2h Non-Exact Rows

| profile_label | packet_dir | case_id | family | failure_mode | executable_match | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query | query_error_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2h_component_identity_negative_examples | results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1 | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | argument_mismatch | False | False | extract_layout | state marker | extract_layout | lifecycle state marker | value_or_alias_query_substitution |

## H2i Non-Exact Rows

| profile_label | packet_dir | case_id | family | failure_mode | executable_match | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query | query_error_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2i_conditional_component_arbitration | results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1 | h2f_alert_t47_negated_switch_decoy | h2f_route_code_label | argument_mismatch | False | False | extract_layout | alert t47 | extract_layout | Escalated | value_or_alias_query_substitution |
| h2i_conditional_component_arbitration | results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1 | h2f_result_tile_comment_value_decoy | h2f_route_component_class_transfer | argument_mismatch | False | False | extract_layout | result tile | extract_layout | result tile for Blocked | value_or_alias_query_substitution |
| h2i_conditional_component_arbitration | results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1 | h2f_resolution_badge_log_result_decoy | h2f_route_component_class_transfer | argument_mismatch | False | False | extract_layout | resolution badge | extract_layout | resolution badge for Deferred | value_or_alias_query_substitution |
| h2i_conditional_component_arbitration | results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1 | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | argument_mismatch | False | False | extract_layout | state marker | extract_layout | lifecycle state marker | value_or_alias_query_substitution |

## Family Rows

| profile_label | family | case_count | exact_success_count | executor_success_count | exact_rate | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| no_directive | h2f_route_code_label | 2 | 0 | 0 | 0.0 | 0.0 |
| no_directive | h2f_route_component_class_transfer | 2 | 0 | 0 | 0.0 | 0.0 |
| no_directive | h2f_route_nonstandard_class | 2 | 0 | 0 | 0.0 | 0.0 |
| no_directive | h2f_route_stale_field | 2 | 1 | 1 | 0.5 | 0.5 |
| no_directive | h2f_activation_panel_notice | 2 | 0 | 0 | 0.0 | 0.0 |
| h2a_component_label_guard | h2f_route_code_label | 2 | 0 | 0 | 0.0 | 0.0 |
| h2a_component_label_guard | h2f_route_component_class_transfer | 2 | 0 | 0 | 0.0 | 0.0 |
| h2a_component_label_guard | h2f_route_nonstandard_class | 2 | 0 | 0 | 0.0 | 0.0 |
| h2a_component_label_guard | h2f_route_stale_field | 2 | 2 | 2 | 1.0 | 1.0 |
| h2a_component_label_guard | h2f_activation_panel_notice | 2 | 2 | 2 | 1.0 | 1.0 |
| component_residual_guard_v12 | h2f_route_code_label | 2 | 2 | 2 | 1.0 | 1.0 |
| component_residual_guard_v12 | h2f_route_component_class_transfer | 2 | 0 | 0 | 0.0 | 0.0 |
| component_residual_guard_v12 | h2f_route_nonstandard_class | 2 | 0 | 0 | 0.0 | 0.0 |
| component_residual_guard_v12 | h2f_route_stale_field | 2 | 2 | 2 | 1.0 | 1.0 |
| component_residual_guard_v12 | h2f_activation_panel_notice | 2 | 1 | 2 | 0.5 | 1.0 |
| h2c_scoped_residual_gate | h2f_route_code_label | 2 | 2 | 2 | 1.0 | 1.0 |
| h2c_scoped_residual_gate | h2f_route_component_class_transfer | 2 | 0 | 0 | 0.0 | 0.0 |
| h2c_scoped_residual_gate | h2f_route_nonstandard_class | 2 | 0 | 0 | 0.0 | 0.0 |
| h2c_scoped_residual_gate | h2f_route_stale_field | 2 | 2 | 2 | 1.0 | 1.0 |
| h2c_scoped_residual_gate | h2f_activation_panel_notice | 2 | 2 | 2 | 1.0 | 1.0 |
| h2d_class_preserving_route | h2f_route_code_label | 2 | 1 | 1 | 0.5 | 0.5 |
| h2d_class_preserving_route | h2f_route_component_class_transfer | 2 | 0 | 0 | 0.0 | 0.0 |
| h2d_class_preserving_route | h2f_route_nonstandard_class | 2 | 0 | 0 | 0.0 | 0.0 |
| h2d_class_preserving_route | h2f_route_stale_field | 2 | 2 | 2 | 1.0 | 1.0 |
| h2d_class_preserving_route | h2f_activation_panel_notice | 2 | 2 | 2 | 1.0 | 1.0 |
| h2e_route_arbitration | h2f_route_code_label | 2 | 2 | 2 | 1.0 | 1.0 |
| h2e_route_arbitration | h2f_route_component_class_transfer | 2 | 0 | 0 | 0.0 | 0.0 |
| h2e_route_arbitration | h2f_route_nonstandard_class | 2 | 0 | 0 | 0.0 | 0.0 |
| h2e_route_arbitration | h2f_route_stale_field | 2 | 2 | 2 | 1.0 | 1.0 |
| h2e_route_arbitration | h2f_activation_panel_notice | 2 | 2 | 2 | 1.0 | 1.0 |
| h2g_component_identity_query_contract | h2f_route_code_label | 2 | 2 | 2 | 1.0 | 1.0 |
| h2g_component_identity_query_contract | h2f_route_component_class_transfer | 2 | 0 | 1 | 0.0 | 0.5 |
| h2g_component_identity_query_contract | h2f_route_nonstandard_class | 2 | 0 | 0 | 0.0 | 0.0 |
| h2g_component_identity_query_contract | h2f_route_stale_field | 2 | 2 | 2 | 1.0 | 1.0 |
| h2g_component_identity_query_contract | h2f_activation_panel_notice | 2 | 2 | 2 | 1.0 | 1.0 |
| h2h_component_identity_negative_examples | h2f_route_code_label | 2 | 2 | 2 | 1.0 | 1.0 |
| h2h_component_identity_negative_examples | h2f_route_component_class_transfer | 2 | 2 | 2 | 1.0 | 1.0 |
| h2h_component_identity_negative_examples | h2f_route_nonstandard_class | 2 | 1 | 1 | 0.5 | 0.5 |
| h2h_component_identity_negative_examples | h2f_route_stale_field | 2 | 2 | 2 | 1.0 | 1.0 |
| h2h_component_identity_negative_examples | h2f_activation_panel_notice | 2 | 2 | 2 | 1.0 | 1.0 |
| h2i_conditional_component_arbitration | h2f_route_code_label | 2 | 1 | 1 | 0.5 | 0.5 |
| h2i_conditional_component_arbitration | h2f_route_component_class_transfer | 2 | 0 | 0 | 0.0 | 0.0 |
| h2i_conditional_component_arbitration | h2f_route_nonstandard_class | 2 | 1 | 1 | 0.5 | 0.5 |
| h2i_conditional_component_arbitration | h2f_route_stale_field | 2 | 2 | 2 | 1.0 | 1.0 |
| h2i_conditional_component_arbitration | h2f_activation_panel_notice | 2 | 2 | 2 | 1.0 | 1.0 |

## Failure Mode Rows

| profile_label | failure_mode | count |
| --- | --- | --- |
| no_directive | argument_mismatch | 3 |
| no_directive | exact | 1 |
| no_directive | no_tool_call | 5 |
| no_directive | wrong_tool | 1 |
| h2a_component_label_guard | argument_mismatch | 6 |
| h2a_component_label_guard | exact | 4 |
| component_residual_guard_v12 | argument_mismatch | 4 |
| component_residual_guard_v12 | exact | 5 |
| component_residual_guard_v12 | executable_paraphrase | 1 |
| h2c_scoped_residual_gate | argument_mismatch | 4 |
| h2c_scoped_residual_gate | exact | 6 |
| h2d_class_preserving_route | argument_mismatch | 5 |
| h2d_class_preserving_route | exact | 5 |
| h2e_route_arbitration | argument_mismatch | 4 |
| h2e_route_arbitration | exact | 6 |
| h2g_component_identity_query_contract | argument_mismatch | 3 |
| h2g_component_identity_query_contract | exact | 6 |
| h2g_component_identity_query_contract | executable_paraphrase | 1 |
| h2h_component_identity_negative_examples | argument_mismatch | 1 |
| h2h_component_identity_negative_examples | exact | 9 |
| h2i_conditional_component_arbitration | argument_mismatch | 4 |
| h2i_conditional_component_arbitration | exact | 6 |

## Findings

| finding_id | finding |
| --- | --- |
| h2f_breaks_h2e_saturation | H2e reaches only 6/10 exact and 6/10 executor-equivalent on the fresh H2f holdout, after previously saturating H2b and H1x. |
| route_arbitration_does_not_beat_h2c_on_h2f | H2e ties H2c on H2f: delta exact=0.0 and delta executor-equivalence=0.0. |
| controllers_remain_causal_against_floor | No-directive reaches 1/10 exact while H2e reaches 6/10, a 0.5 exact-rate lift. Intermediate rows are H2a=4/10, v12=5/10, H2d=5/10, and H2c=6/10. |
| remaining_failure_is_component_identity_binding | All H2e non-exact rows are argument mismatches in h2f_route_component_class_transfer, h2f_route_nonstandard_class. The model preserved the right tool but substituted displayed values or aliases for requested component identities: result tile->Blocked, resolution badge->Deferred, state marker->lifecycle state marker, mode switch->mode toggle. |
| next_slice | H2g keeps strict exactness at 6/10 but improves executor-equivalence to 7/10, a 0.09999999999999998 executor lift over H2e. This is partial evidence, not promotion evidence, because the H2g non-exact table still has 4 rows. |
| h2h_repairs_h2f_component_identity | H2h reaches 9/10 exact and 9/10 executor-equivalent on H2f, lifting exactness by 0.30000000000000004 versus H2e and 0.30000000000000004 versus H2g. |
| h2h_residual_state_marker_alias | H2h leaves one H2f residual. The remaining target-query substitution is state marker->lifecycle state marker, so the next contract work should isolate marker-prefix alias expansion rather than broad value-substitution prose. |
| next_contract | Do not promote H2h globally from H2f alone. Use the H2b/H1x transfer tradeoff packets to test whether the negative examples preserve prior residual-exactness and route-arbitration wins. |
| h2i_conditionalization_is_negative | H2i conditional arbitration falls back to 6/10 exact on H2f, tying H2e with delta exact=0.0 and trailing H2h by -0.30000000000000004. Its non-exact rows are alert t47->Escalated, result tile->result tile for Blocked, resolution badge->resolution badge for Deferred, state marker->lifecycle state marker. |
