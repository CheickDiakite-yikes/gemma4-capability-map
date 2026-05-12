# H2p Contextual Surface Alias Routing

Generated: `2026-05-12T21:23:26.684268+00:00`

## Summary

H2p separates a surface-class alias from a displayed value when the prompt explicitly asks for a surface shape and demotes the nearby value-bearing components to context. This closes the remaining H2m residue left by H2o: the `tile-style result surface for Blocked` row now targets `result tile` instead of the visible value `Blocked`. H2p reaches 8/8 strict and executor-equivalent on H2m while preserving the saturated H2k, H2l, and H2f transfer packets.

![H2p contextual surface alias routing gate](figures/h2p_contextual_surface_alias_routing_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2e_execute_v1 | 8 | 1 | 0.125 | 3 | 0.375 |
| h2m_h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2j_execute_v1 | 8 | 3 | 0.375 | 3 | 0.375 |
| h2m_h2n_scoped_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_scoped_target_query_normalization | results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2m_execute_v1 | 8 | 3 | 0.375 | 5 | 0.625 |
| h2m_h2o_value_bearing_target_query_synthesis | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis | results/tool_probe_replay_live/20260512T_h2o_value_bearing_target_synthesis_on_h2m_execute_v1 | 8 | 7 | 0.875 | 7 | 0.875 |
| h2m_h2p_contextual_surface_alias_routing | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing | results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2m_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2k_h2o_value_bearing_target_query_synthesis | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis | results/tool_probe_replay_live/20260512T_h2o_value_bearing_target_synthesis_on_h2k_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2k_h2p_contextual_surface_alias_routing | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing | results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2k_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2l_h2o_value_bearing_target_query_synthesis | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis | results/tool_probe_replay_live/20260512T_h2o_value_bearing_target_synthesis_on_h2l_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2l_h2p_contextual_surface_alias_routing | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing | results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2l_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h2f_h2o_value_bearing_target_query_synthesis | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis | results/tool_probe_replay_live/20260512T_h2o_value_bearing_target_synthesis_on_h2f_execute_v1 | 10 | 10 | 1.0 | 10 | 1.0 |
| h2f_h2p_contextual_surface_alias_routing | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing | results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2f_execute_v1 | 10 | 10 | 1.0 | 10 | 1.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2p_vs_h2o | results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2m_v1 | 8 | 0.875 | 1.0 | 0.125 | 0.875 | 1.0 | 0.125 |
| h2m_h2p_vs_h2n | results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2n_on_h2m_v1 | 8 | 0.375 | 1.0 | 0.625 | 0.625 | 1.0 | 0.375 |
| h2m_h2p_vs_h2j | results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2j_on_h2m_v1 | 8 | 0.375 | 1.0 | 0.625 | 0.375 | 1.0 | 0.625 |
| h2m_h2p_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2e_on_h2m_v1 | 8 | 0.125 | 1.0 | 0.875 | 0.375 | 1.0 | 0.625 |
| h2k_h2p_vs_h2o | results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2k_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2l_h2p_vs_h2o | results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2l_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2f_h2p_vs_h2o | results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2f_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |

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
| h2m_h2o_value_bearing_target_query_synthesis | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | argument_mismatch | False | extract_layout | result tile | extract_layout | Blocked |

## Controller Intervention Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label | value_bearing_label | value_suffix | matched_phrase | display_value | surface_label | surface_text | surface_region_id | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2p_contextual_surface_alias_routing | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result chip"} | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result badge Blocked"} | result badge | result badge Blocked | blocked | blocked result badge |  |  |  |  | value_bearing_label_recoverable |
| h2m_h2p_contextual_surface_alias_routing | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"Closed state tag"} | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"state tag Closed"} | state tag | state tag Closed | closed | closed state tag |  |  |  |  | value_bearing_label_recoverable |
| h2m_h2p_contextual_surface_alias_routing | h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2m-mode-toggle-manual","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2m-mode-toggle-manual","target_query":"mode toggle Manual"} | mode toggle | mode toggle Manual | manual | manual mode toggle |  |  |  |  | value_bearing_label_recoverable |
| h2m_h2p_contextual_surface_alias_routing | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge critical"} | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge Critical"} | priority badge | priority badge Critical | critical | critical priority badge |  |  |  |  | value_bearing_label_recoverable |
| h2m_h2p_contextual_surface_alias_routing | h2m_error_notice_contextual_alias | h2m_contextual_alias_is_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"archive panel"} | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"error notice"} | error notice |  |  |  |  |  |  |  |  |
| h2m_h2p_contextual_surface_alias_routing | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | visual_contextual_surface_alias_routing | extract_layout | {"image_id":"img-h2m-result-tile","target_query":"Blocked"} | extract_layout | {"image_id":"img-h2m-result-tile","target_query":"result tile"} |  |  |  |  | Blocked | result tile | Blocked | h2m-result-tile-12052 | contextual_surface_alias_recoverable |
| h2m_h2p_contextual_surface_alias_routing | h2m_mode_field_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode switch"} | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode field"} | mode field |  |  |  |  |  |  |  |  |
| h2k_h2p_contextual_surface_alias_routing | h2k_mode_toggle_negated_consent_toggle_decoy | h2k_negated_same_component_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-mode-toggle","target_query":"mode toggle Manual"} | extract_layout | {"image_id":"img-h2k-mode-toggle","target_query":"mode toggle"} | mode toggle |  |  |  |  |  |  |  |  |
| h2k_h2p_contextual_surface_alias_routing | h2k_result_badge_negated_result_tile_decoy | h2k_negated_same_component_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-result-badge","target_query":"result badge Blocked"} | extract_layout | {"image_id":"img-h2k-result-badge","target_query":"result badge"} | result badge |  |  |  |  |  |  |  |  |
| h2k_h2p_contextual_surface_alias_routing | h2k_error_banner_archived_error_notice_decoy | h2k_transfer_regression_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-error-banner","target_query":"error notice"} | extract_layout | {"image_id":"img-h2k-error-banner","target_query":"error banner"} | error banner |  |  |  |  |  |  |  |  |
| h2k_h2p_contextual_surface_alias_routing | h2k_state_tag_before_reading_state_marker_decoy | h2k_before_reading_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-state-tag","target_query":"state tag Closed"} | extract_layout | {"image_id":"img-h2k-state-tag","target_query":"state tag"} | state tag |  |  |  |  |  |  |  |  |
| h2k_h2p_contextual_surface_alias_routing | h2k_mode_field_before_reading_mode_switch_decoy | h2k_before_reading_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-mode-field","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2k-mode-field","target_query":"mode field"} | mode field |  |  |  |  |  |  |  |  |
| h2l_h2p_contextual_surface_alias_routing | h2l_status_badge_short_label_regression_guard | h2l_h2k_regression_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2l-status-badge-short","target_query":"critical chip"} | extract_layout | {"image_id":"img-h2l-status-badge-short","target_query":"status badge"} | status badge |  |  |  |  |  |  |  |  |
| h2f_h2p_contextual_surface_alias_routing | h2f_result_tile_comment_value_decoy | h2f_route_component_class_transfer | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-result-tile","target_query":"Blocked"} | extract_layout | {"image_id":"img-h2f-result-tile","target_query":"result tile"} | result tile |  |  |  |  |  |  |  |  |
| h2f_h2p_contextual_surface_alias_routing | h2f_resolution_badge_log_result_decoy | h2f_route_component_class_transfer | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-resolution-badge","target_query":"Deferred"} | extract_layout | {"image_id":"img-h2f-resolution-badge","target_query":"resolution badge"} | resolution badge |  |  |  |  |  |  |  |  |
| h2f_h2p_contextual_surface_alias_routing | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-state-marker","target_query":"lifecycle state marker"} | extract_layout | {"image_id":"img-h2f-state-marker","target_query":"state marker"} | state marker |  |  |  |  |  |  |  |  |
| h2f_h2p_contextual_surface_alias_routing | h2f_mode_switch_note_value_decoy | h2f_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-mode-switch","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2f-mode-switch","target_query":"mode switch"} | mode switch |  |  |  |  |  |  |  |  |
| h2f_h2p_contextual_surface_alias_routing | h2f_owner_field_previous_memo_decoy | h2f_route_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"owner field","selection_id":"sel-h2f-owner-memo"} | extract_layout | {"image_id":"img-h2f-owner-field","target_query":"owner field"} |  |  |  |  |  |  |  |  |  |

## Contextual Surface Alias Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label | value_bearing_label | value_suffix | matched_phrase | display_value | surface_label | surface_text | surface_region_id | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2m_h2p_contextual_surface_alias_routing | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | visual_contextual_surface_alias_routing | extract_layout | {"image_id":"img-h2m-result-tile","target_query":"Blocked"} | extract_layout | {"image_id":"img-h2m-result-tile","target_query":"result tile"} |  |  |  |  | Blocked | result tile | Blocked | h2m-result-tile-12052 | contextual_surface_alias_recoverable |

## Findings

| finding_id | finding |
| --- | --- |
| h2p_saturates_h2m_surface_alias_boundary | H2p improves H2m strict exactness from H2o's 7/8 to 8/8, adding 0.125 exact-rate and 0.125 executor-equivalence-rate. |
| h2p_adds_large_deltas_over_non_constructive_controls | Relative to H2n, H2p adds 0.625 exact-rate and 0.375 executor-equivalence-rate on H2m. |
| h2p_mechanism_is_single_alias_gate | H2p records 1 contextual surface-alias intervention on H2m, alongside 4 value-bearing syntheses and 2 ordinary contextual rewrites. |
| h2p_preserves_transfer_gates | H2p ties H2o on H2k at 8/8, H2l at 8/8, and H2f at 10/10 with zero exact-rate deltas. |
| h2p_closes_h2m_current_exact_boundary | H2p leaves 0 non-exact H2m rows under the current H2m packet. |
