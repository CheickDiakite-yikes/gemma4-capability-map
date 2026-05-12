# H2m Less-Direct Target-Normalization Overreach Synthesis

Generated: `2026-05-12T20:15:33.038410+00:00`

## Summary

H2m removes H2l's explicit target-is wording while preserving the same value-bearing, alias, and regression-guard families. It breaks the H2l saturation: full H2j and H2j without stale-selection both fall to 3/8 strict and executor-equivalent. H2e reaches 1/8 strict and 3/8 executor-equivalent. The mechanism is mixed: H2j still repairs some contextual labels, but it also over-strips less-direct value-bearing labels such as `result badge Blocked` and `state tag Closed` into shorter component labels.

![H2m less-direct overreach gate](figures/h2m_less_direct_overreach_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2e_execute_v1 | 8 | 1 | 0.125 | 3 | 0.375 |
| h2j_target_query_normalization_no_stale_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_target_query_normalization_no_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2j_no_stale_gate_execute_v1 | 8 | 3 | 0.375 | 3 | 0.375 |
| h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2j_execute_v1 | 8 | 3 | 0.375 | 3 | 0.375 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2j_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_h2e_v1 | 8 | 0.125 | 0.375 | 0.25 | 0.375 | 0.375 | 0.0 |
| h2j_vs_no_stale_gate | results/tool_probe_replay_live_comparisons/20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_no_stale_gate_v1 | 8 | 0.375 | 0.375 | 0.0 | 0.375 | 0.375 | 0.0 |

## Family Rows

| family | case_count | expected_target_queries |
| --- | --- | --- |
| h2m_contextual_alias_is_target | 2 | error notice; result tile |
| h2m_h2k_regression_guard_less_direct | 2 | status badge; mode field |
| h2m_less_direct_value_bearing_target | 4 | result badge Blocked; state tag Closed; mode toggle Manual; priority badge Critical |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2e_route_arbitration | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | result badge Blocked | extract_layout | result chip |
| h2e_route_arbitration | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | executable_paraphrase | True | extract_layout | state tag Closed | extract_layout | Closed state tag |
| h2e_route_arbitration | h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | mode toggle Manual | extract_layout | mode toggle |
| h2e_route_arbitration | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | executable_paraphrase | True | extract_layout | priority badge Critical | extract_layout | priority badge critical |
| h2e_route_arbitration | h2m_error_notice_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | error notice | extract_layout | archive panel |
| h2e_route_arbitration | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | result tile | extract_layout | Blocked |
| h2e_route_arbitration | h2m_mode_field_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | argument_mismatch | False | extract_layout | mode field | extract_layout | mode switch |
| h2j_target_query_normalization_no_stale_gate | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | result badge Blocked | extract_layout | result badge |
| h2j_target_query_normalization_no_stale_gate | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | state tag Closed | extract_layout | state tag |
| h2j_target_query_normalization_no_stale_gate | h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | mode toggle Manual | extract_layout | mode toggle |
| h2j_target_query_normalization_no_stale_gate | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | priority badge Critical | extract_layout | priority badge |
| h2j_target_query_normalization_no_stale_gate | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | result tile | extract_layout | Blocked |
| h2j_target_query_normalization | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | result badge Blocked | extract_layout | result badge |
| h2j_target_query_normalization | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | state tag Closed | extract_layout | state tag |
| h2j_target_query_normalization | h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | mode toggle Manual | extract_layout | mode toggle |
| h2j_target_query_normalization | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | priority badge Critical | extract_layout | priority badge |
| h2j_target_query_normalization | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | result tile | extract_layout | Blocked |

## Controller Intervention Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2j_target_query_normalization_no_stale_gate | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result chip"} | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result badge"} | result badge |
| h2j_target_query_normalization_no_stale_gate | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"Closed state tag"} | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"state tag"} | state tag |
| h2j_target_query_normalization_no_stale_gate | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge critical"} | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge"} | priority badge |
| h2j_target_query_normalization_no_stale_gate | h2m_error_notice_contextual_alias | h2m_contextual_alias_is_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"archive panel"} | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"error notice"} | error notice |
| h2j_target_query_normalization_no_stale_gate | h2m_mode_field_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode switch"} | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode field"} | mode field |
| h2j_target_query_normalization | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result chip"} | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result badge"} | result badge |
| h2j_target_query_normalization | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"Closed state tag"} | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"state tag"} | state tag |
| h2j_target_query_normalization | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge critical"} | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge"} | priority badge |
| h2j_target_query_normalization | h2m_error_notice_contextual_alias | h2m_contextual_alias_is_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"archive panel"} | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"error notice"} | error notice |
| h2j_target_query_normalization | h2m_mode_field_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode switch"} | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode field"} | mode field |

## Overstrip Rows

| case_id | family | expected_target_query | actual_target_query | from_arguments | to_arguments | prompt_state_label |
| --- | --- | --- | --- | --- | --- | --- |
| h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | result badge Blocked | result badge | {"image_id":"img-h2m-result-badge-blocked","target_query":"result chip"} | {"image_id":"img-h2m-result-badge-blocked","target_query":"result badge"} | result badge |
| h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | state tag Closed | state tag | {"image_id":"img-h2m-state-tag-closed","target_query":"Closed state tag"} | {"image_id":"img-h2m-state-tag-closed","target_query":"state tag"} | state tag |
| h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | priority badge Critical | priority badge | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge critical"} | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge"} | priority badge |

## Findings

| finding_id | finding |
| --- | --- |
| h2m_breaks_h2l_saturation | H2m breaks the H2l saturation: H2j reaches 3/8 exact and executor-equivalent, H2j-no-stale also reaches 3/8, and H2e reaches 1/8 exact. |
| h2m_target_normalization_is_mixed | H2j improves exact-rate over H2e by 0.25 but does not improve executor-equivalence over H2e (0.0); it ties the no-stale ablation with 0.0 exact-rate delta. |
| h2m_exposes_overstrip | H2j records 5 target-query-normalization interventions, but 3 of them over-strip less-direct value-bearing targets to shorter component labels. |
| next_gate_should_scope_normalization | The next H2n move should make target-query normalization conditional on evidence that the shorter component label is explicitly requested, while preserving the H2k/H2l regression-guard repairs. |
