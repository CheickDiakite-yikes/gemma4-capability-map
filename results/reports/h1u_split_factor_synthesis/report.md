# H1u Split-Factor Synthesis

Generated: `2026-05-10T18:15:33.663610+00:00`

## Summary

H1u splits the failed H1t compact conditional route into independent factors. V14 confirms that nonstandard component-class wording repairs tag/toggle value collapse, but v15 is the stronger local candidate: it saturates H1r at `6 / 6` and should be transfer-tested next.

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | no_tool_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_no_directive_execute_v1 | `6` | `0` | `0.00000` | `1` | `0.16667` | `2` | `0` | `3` |
| component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_label_guard_execute_v1 | `6` | `5` | `0.83333` | `5` | `0.83333` | `1` | `0` | `0` |
| component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_residual_guard_execute_v1 | `6` | `6` | `1.00000` | `6` | `1.00000` | `0` | `0` | `0` |
| conditional_residual_route_v13 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route | results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1 | `6` | `3` | `0.50000` | `3` | `0.50000` | `3` | `0` | `0` |
| nonstandard_component_class_guard_v14 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_nonstandard_component_class_guard | results/tool_probe_replay_live/20260510T_h1u_nonstandard_component_class_guard_on_h1r_component_residual_execute_v1 | `6` | `5` | `0.83333` | `5` | `0.83333` | `1` | `0` | `0` |
| code_label_exact_guard_v15 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | results/tool_probe_replay_live/20260510T_h1u_code_label_exact_guard_on_h1r_component_residual_execute_v1 | `6` | `6` | `1.00000` | `6` | `1.00000` | `0` | `0` | `0` |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| no_directive | h1r_code_label_exactness | `2` | `0` | `0.00000` | `1` | `0.50000` |
| no_directive | h1r_nonstandard_component_class | `2` | `0` | `0.00000` | `0` | `0.00000` |
| no_directive | h1r_stale_selection_component_label | `2` | `0` | `0.00000` | `0` | `0.00000` |
| component_label_guard_v11 | h1r_code_label_exactness | `2` | `1` | `0.50000` | `1` | `0.50000` |
| component_label_guard_v11 | h1r_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_label_guard_v11 | h1r_stale_selection_component_label | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1r_code_label_exactness | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1r_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1r_stale_selection_component_label | `2` | `2` | `1.00000` | `2` | `1.00000` |
| conditional_residual_route_v13 | h1r_code_label_exactness | `2` | `1` | `0.50000` | `1` | `0.50000` |
| conditional_residual_route_v13 | h1r_nonstandard_component_class | `2` | `0` | `0.00000` | `0` | `0.00000` |
| conditional_residual_route_v13 | h1r_stale_selection_component_label | `2` | `2` | `1.00000` | `2` | `1.00000` |
| nonstandard_component_class_guard_v14 | h1r_code_label_exactness | `2` | `1` | `0.50000` | `1` | `0.50000` |
| nonstandard_component_class_guard_v14 | h1r_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| nonstandard_component_class_guard_v14 | h1r_stale_selection_component_label | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1r_code_label_exactness | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1r_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1r_stale_selection_component_label | `2` | `2` | `1.00000` | `2` | `1.00000` |

## Comparison Rows

| comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tool_probe_replay_live_comparisons/20260510T_h1u_nonstandard_component_guard_h1r_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_nonstandard_component_class_guard | `6` | `0.83333` | `0.83333` | `0.00000` | `0.83333` | `0.83333` | `0.00000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1u_nonstandard_component_guard_h1r_vs_component_residual_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_nonstandard_component_class_guard | `6` | `1.00000` | `0.83333` | `-0.16667` | `1.00000` | `0.83333` | `-0.16667` |
| results/tool_probe_replay_live_comparisons/20260510T_h1u_nonstandard_component_guard_h1r_vs_conditional_residual_route_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_nonstandard_component_class_guard | `6` | `0.50000` | `0.83333` | `0.33333` | `0.50000` | `0.83333` | `0.33333` |
| results/tool_probe_replay_live_comparisons/20260510T_h1u_code_label_exact_guard_h1r_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `6` | `0.83333` | `1.00000` | `0.16667` | `0.83333` | `1.00000` | `0.16667` |
| results/tool_probe_replay_live_comparisons/20260510T_h1u_code_label_exact_guard_h1r_vs_component_residual_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `6` | `1.00000` | `1.00000` | `0.00000` | `1.00000` | `1.00000` | `0.00000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1u_code_label_exact_guard_h1r_vs_conditional_residual_route_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `6` | `0.50000` | `1.00000` | `0.50000` | `0.50000` | `1.00000` | `0.50000` |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match |
| --- | --- | --- | --- | --- |
| no_directive | h1r_owner_field_stale_selection_note_decoy | h1r_stale_selection_component_label | no_tool_call | `false` |
| no_directive | h1r_assignee_field_previous_selection_summary_decoy | h1r_stale_selection_component_label | no_tool_call | `false` |
| no_directive | h1r_state_tag_log_value_decoy | h1r_nonstandard_component_class | argument_mismatch | `false` |
| no_directive | h1r_mode_toggle_note_value_decoy | h1r_nonstandard_component_class | argument_mismatch | `false` |
| no_directive | h1r_alert_s92_toggle_negation_decoy | h1r_code_label_exactness | executable_paraphrase | `true` |
| no_directive | h1r_badge_c08_note_decoy | h1r_code_label_exactness | no_tool_call | `false` |
| component_label_guard_v11 | h1r_alert_s92_toggle_negation_decoy | h1r_code_label_exactness | argument_mismatch | `false` |
| conditional_residual_route_v13 | h1r_state_tag_log_value_decoy | h1r_nonstandard_component_class | argument_mismatch | `false` |
| conditional_residual_route_v13 | h1r_mode_toggle_note_value_decoy | h1r_nonstandard_component_class | argument_mismatch | `false` |
| conditional_residual_route_v13 | h1r_alert_s92_toggle_negation_decoy | h1r_code_label_exactness | argument_mismatch | `false` |
| nonstandard_component_class_guard_v14 | h1r_alert_s92_toggle_negation_decoy | h1r_code_label_exactness | argument_mismatch | `false` |

## Findings

| finding_id | finding |
| --- | --- |
| v14_is_factor_positive_but_incomplete | Nonstandard component-class guard v14 reaches 5/6 and fixes tag/toggle value-collapse, but still misses h1r_alert_s92_toggle_negation_decoy. |
| v15_saturates_h1r | Code-label exact guard v15 reaches 6/6 exact and 6/6 executor-equivalent, matching v12 on H1r. |
| v15_beats_v11_ties_v12 | v15 improves over v11 by 0.167 exact-rate and ties v12 with delta 0.000. |
| transfer_decision | Transfer-test v15 across H1n/H1o/H1p next. It may keep the H1r repair while using less broad component-residual prose than v12. |
