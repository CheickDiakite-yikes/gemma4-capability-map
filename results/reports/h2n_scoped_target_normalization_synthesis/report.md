# H2n Scoped Target-Normalization Synthesis

Generated: `2026-05-12T20:39:57.878339+00:00`

## Summary

H2n converts the H2m negative result into a scoped controller policy. The normalizer still performs the contextual-label repairs that H2j needed, but it refuses to shorten value-bearing labels when the prompt evidence implies that the displayed value is part of the requested component identity. On H2m, this does not improve strict exactness over H2j: both remain 3/8. It does improve executor-equivalence from 3/8 to 5/8, and it keeps the H2k, H2l, and H2f transfer gates saturated. The remaining H2m misses are therefore no longer just an over-strip problem; they need canonical value-bearing target synthesis.

![H2n scoped target-normalization gate](figures/h2n_scoped_target_normalization_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2e_execute_v1 | 8 | 1 | 0.125 | 3 | 0.375 |
| h2m_h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2j_execute_v1 | 8 | 3 | 0.375 | 3 | 0.375 |
| h2m_h2n_scoped_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_scoped_target_query_normalization | results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2m_execute_v1 | 8 | 3 | 0.375 | 5 | 0.625 |
| h2k_h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2k_h2n_scoped_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_scoped_target_query_normalization | results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2k_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2l_h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2j_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2l_h2n_scoped_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_scoped_target_query_normalization | results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2l_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2f_h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2f_execute_v2 | 10 | 10 | 1.0 | 10 | 1.0 |
| h2f_h2n_scoped_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_scoped_target_query_normalization | results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2f_execute_v1 | 10 | 10 | 1.0 | 10 | 1.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2n_vs_h2j | results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2m_v1 | 8 | 0.375 | 0.375 | 0.0 | 0.375 | 0.625 | 0.25 |
| h2m_h2n_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2e_on_h2m_v1 | 8 | 0.125 | 0.375 | 0.25 | 0.375 | 0.625 | 0.25 |
| h2k_h2n_vs_h2j | results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2k_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2l_h2n_vs_h2j | results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2l_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2f_h2n_vs_h2j | results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2f_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2e_route_arbitration | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | result badge Blocked | extract_layout | result chip |
| h2m_h2e_route_arbitration | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | executable_paraphrase | True | extract_layout | state tag Closed | extract_layout | Closed state tag |
| h2m_h2e_route_arbitration | h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | mode toggle Manual | extract_layout | mode toggle |
| h2m_h2e_route_arbitration | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | executable_paraphrase | True | extract_layout | priority badge Critical | extract_layout | priority badge critical |
| h2m_h2e_route_arbitration | h2m_error_notice_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | error notice | extract_layout | archive panel |
| h2m_h2e_route_arbitration | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | result tile | extract_layout | Blocked |
| h2m_h2e_route_arbitration | h2m_mode_field_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | argument_mismatch | False | extract_layout | mode field | extract_layout | mode switch |
| h2m_h2j_target_query_normalization | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | result badge Blocked | extract_layout | result badge |
| h2m_h2j_target_query_normalization | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | state tag Closed | extract_layout | state tag |
| h2m_h2j_target_query_normalization | h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | mode toggle Manual | extract_layout | mode toggle |
| h2m_h2j_target_query_normalization | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | priority badge Critical | extract_layout | priority badge |
| h2m_h2j_target_query_normalization | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | result tile | extract_layout | Blocked |
| h2m_h2n_scoped_target_query_normalization | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | result badge Blocked | extract_layout | result chip |
| h2m_h2n_scoped_target_query_normalization | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | executable_paraphrase | True | extract_layout | state tag Closed | extract_layout | Closed state tag |
| h2m_h2n_scoped_target_query_normalization | h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | argument_mismatch | False | extract_layout | mode toggle Manual | extract_layout | mode toggle |
| h2m_h2n_scoped_target_query_normalization | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | executable_paraphrase | True | extract_layout | priority badge Critical | extract_layout | priority badge critical |
| h2m_h2n_scoped_target_query_normalization | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | result tile | extract_layout | Blocked |

## Controller Intervention Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label | preserved_target_query | value_bearing_label | value_suffix | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2n_scoped_target_query_normalization | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization_blocked | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result chip"} |  | {} | result badge | result chip | result badge Blocked | blocked | value_bearing_label_requested |
| h2m_h2n_scoped_target_query_normalization | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization_blocked | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"Closed state tag"} |  | {} | state tag | Closed state tag | state tag Closed | closed | value_bearing_label_requested |
| h2m_h2n_scoped_target_query_normalization | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization_blocked | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge critical"} |  | {} | priority badge | priority badge critical | priority badge Critical | critical | value_bearing_label_requested |
| h2m_h2n_scoped_target_query_normalization | h2m_error_notice_contextual_alias | h2m_contextual_alias_is_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"archive panel"} | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"error notice"} | error notice |  |  |  |  |
| h2m_h2n_scoped_target_query_normalization | h2m_mode_field_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode switch"} | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode field"} | mode field |  |  |  |  |
| h2k_h2n_scoped_target_query_normalization | h2k_mode_toggle_negated_consent_toggle_decoy | h2k_negated_same_component_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-mode-toggle","target_query":"mode toggle Manual"} | extract_layout | {"image_id":"img-h2k-mode-toggle","target_query":"mode toggle"} | mode toggle |  |  |  |  |
| h2k_h2n_scoped_target_query_normalization | h2k_result_badge_negated_result_tile_decoy | h2k_negated_same_component_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-result-badge","target_query":"result badge Blocked"} | extract_layout | {"image_id":"img-h2k-result-badge","target_query":"result badge"} | result badge |  |  |  |  |
| h2k_h2n_scoped_target_query_normalization | h2k_error_banner_archived_error_notice_decoy | h2k_transfer_regression_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-error-banner","target_query":"error notice"} | extract_layout | {"image_id":"img-h2k-error-banner","target_query":"error banner"} | error banner |  |  |  |  |
| h2k_h2n_scoped_target_query_normalization | h2k_state_tag_before_reading_state_marker_decoy | h2k_before_reading_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-state-tag","target_query":"state tag Closed"} | extract_layout | {"image_id":"img-h2k-state-tag","target_query":"state tag"} | state tag |  |  |  |  |
| h2k_h2n_scoped_target_query_normalization | h2k_mode_field_before_reading_mode_switch_decoy | h2k_before_reading_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-mode-field","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2k-mode-field","target_query":"mode field"} | mode field |  |  |  |  |
| h2l_h2n_scoped_target_query_normalization | h2l_status_badge_short_label_regression_guard | h2l_h2k_regression_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2l-status-badge-short","target_query":"critical chip"} | extract_layout | {"image_id":"img-h2l-status-badge-short","target_query":"status badge"} | status badge |  |  |  |  |
| h2f_h2n_scoped_target_query_normalization | h2f_result_tile_comment_value_decoy | h2f_route_component_class_transfer | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-result-tile","target_query":"Blocked"} | extract_layout | {"image_id":"img-h2f-result-tile","target_query":"result tile"} | result tile |  |  |  |  |
| h2f_h2n_scoped_target_query_normalization | h2f_resolution_badge_log_result_decoy | h2f_route_component_class_transfer | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-resolution-badge","target_query":"Deferred"} | extract_layout | {"image_id":"img-h2f-resolution-badge","target_query":"resolution badge"} | resolution badge |  |  |  |  |
| h2f_h2n_scoped_target_query_normalization | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-state-marker","target_query":"lifecycle state marker"} | extract_layout | {"image_id":"img-h2f-state-marker","target_query":"state marker"} | state marker |  |  |  |  |
| h2f_h2n_scoped_target_query_normalization | h2f_mode_switch_note_value_decoy | h2f_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-mode-switch","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2f-mode-switch","target_query":"mode switch"} | mode switch |  |  |  |  |
| h2f_h2n_scoped_target_query_normalization | h2f_owner_field_previous_memo_decoy | h2f_route_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"owner field","selection_id":"sel-h2f-owner-memo"} | extract_layout | {"image_id":"img-h2f-owner-field","target_query":"owner field"} |  |  |  |  |  |

## Scoped Block Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label | preserved_target_query | value_bearing_label | value_suffix | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2n_scoped_target_query_normalization | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization_blocked | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result chip"} |  | {} | result badge | result chip | result badge Blocked | blocked | value_bearing_label_requested |
| h2m_h2n_scoped_target_query_normalization | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization_blocked | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"Closed state tag"} |  | {} | state tag | Closed state tag | state tag Closed | closed | value_bearing_label_requested |
| h2m_h2n_scoped_target_query_normalization | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization_blocked | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge critical"} |  | {} | priority badge | priority badge critical | priority badge Critical | critical | value_bearing_label_requested |

## Findings

| finding_id | finding |
| --- | --- |
| h2n_improves_h2m_executor_equivalence_not_strict | H2n ties H2j strict exactness on H2m at 3/8 but improves executor-equivalence from 3/8 to 5/8, a 0.25 executor-equivalence-rate gain. |
| h2n_keeps_h2e_exact_gain | Against H2e, H2n improves H2m strict exactness from 1/8 to 3/8 and executor-equivalence from 3/8 to 5/8. |
| h2n_scoping_blocks_value_bearing_overstrip | H2n records 3 scoped target-query-normalization blocks on H2m value-bearing rows while preserving 2 contextual-label rewrites. |
| h2n_transfers_without_regression | H2n preserves the previous H2j transfer gates: 8/8 on H2k, 8/8 on H2l, and 10/10 on H2f with zero exact-rate delta versus H2j on each packet. |
| next_gate_needs_canonical_value_bearing_target_synthesis | H2n still leaves 5 non-exact H2m rows, so the next H2o question is whether the controller can synthesize canonical value-bearing target queries only when the longer label is recoverable. |
