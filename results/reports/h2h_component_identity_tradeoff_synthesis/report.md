# H2h Component-Identity Tradeoff Synthesis

Generated: `2026-05-10T21:48:15.577500+00:00`

## Summary

H2h is a strong scoped repair for the fresh H2f holdout but not a global successor. It raises H2f from H2e/H2g's 6/10 strict exactness to 9/10, yet regresses the prior H2b and H1x gates that H2e had saturated. The research interpretation is that explicit negative examples can causally repair displayed-value component-identity failures, but the same prose can over-constrain related component classes and code-label rows.

![H2h tradeoff gate](figures/h2h_tradeoff_gate.svg)

## Packet Rows

| suite | profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2f | h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1 | 10 | 6 | 0.6 | 6 | 0.6 |
| h2f | h2g_component_identity_query_contract | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_query_contract_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1 | 10 | 6 | 0.6 | 7 | 0.7 |
| h2f | h2h_component_identity_negative_examples | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1 | 10 | 9 | 0.9 | 9 | 0.9 |
| h2b | h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 5 | 1.0 | 5 | 1.0 |
| h2b | h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h2b_execute_v1 | 5 | 5 | 1.0 | 5 | 1.0 |
| h2b | h2h_component_identity_negative_examples | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2b_execute_v1 | 5 | 3 | 0.6 | 3 | 0.6 |
| h1x | h2d_class_preserving_route | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_class_preserving_residual_route_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x | h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x | h2h_component_identity_negative_examples | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h1x_execute_v1 | 8 | 6 | 0.75 | 6 | 0.75 |

## Comparison Rows

| suite | comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2f | h2h_vs_h2e | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h2f_v1 | 10 | 0.6 | 0.9 | 0.30000000000000004 | 0.6 | 0.9 | 0.30000000000000004 |
| h2f | h2h_vs_h2g | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2g_on_h2f_v1 | 10 | 0.6 | 0.9 | 0.30000000000000004 | 0.7 | 0.9 | 0.20000000000000007 |
| h2b | h2h_vs_h2e | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h2b_v1 | 5 | 1.0 | 0.6 | -0.4 | 1.0 | 0.6 | -0.4 |
| h2b | h2h_vs_h2c | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2c_on_h2b_v1 | 5 | 1.0 | 0.6 | -0.4 | 1.0 | 0.6 | -0.4 |
| h1x | h2h_vs_h2e | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h1x_v1 | 8 | 1.0 | 0.75 | -0.25 | 1.0 | 0.75 | -0.25 |
| h1x | h2h_vs_h2d | results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2d_on_h1x_v1 | 8 | 1.0 | 0.75 | -0.25 | 1.0 | 0.75 | -0.25 |

## H2h Non-Exact Rows

| suite | case_id | family | failure_mode | expected_tool | expected_target_query | actual_tool | actual_target_query | query_error_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2f | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | argument_mismatch | extract_layout | state marker | extract_layout | lifecycle state marker | component_class_or_value_substitution |
| h2b | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | argument_mismatch | extract_layout | result pill | extract_layout | result tile | component_class_or_value_substitution |
| h2b | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | argument_mismatch | extract_layout | badge c08 | extract_layout | badge m31 c08 | component_class_or_value_substitution |
| h1x | h1x_resolution_chip_comment_result_decoy | h1x_oblique_surface_value | argument_mismatch | extract_layout | result chip | extract_layout | result tile | component_class_or_value_substitution |
| h1x | h1x_error_notice_history_activation_decoy | h1x_oblique_activation_no_call | argument_mismatch | extract_layout | error banner | extract_layout | error notice | component_class_or_value_substitution |

## Findings

| finding_id | finding |
| --- | --- |
| h2h_repairs_fresh_h2f | H2h reaches 9/10 strict and 9/10 executor-equivalent on H2f, a 0.30000000000000004 exact-rate lift over H2e. |
| h2h_regresses_prior_transfer_gates | H2h falls to 3/5 on H2b (-0.4 versus H2e) and 6/8 on H1x (-0.25 versus H2e). |
| h2h_failure_boundary | The residual substitutions are concentrated in component-class transfer and alias expansion: h2f:state marker->lifecycle state marker; h2b:result pill->result tile; h2b:badge c08->badge m31 c08; h1x:result chip->result tile; h1x:error banner->error notice. |
| next_slice | The next hypothesis should be conditional arbitration, not a broader negative-example paragraph: keep H2e's route arbitration as the default and activate H2h-style negative examples only for explicit displayed-value component-identity prompts. |
