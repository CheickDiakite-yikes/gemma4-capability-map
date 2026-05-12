# H2r Composed Route-Gating Synthesis

Generated: `2026-05-12T23:11:47.280773+00:00`

## Summary

H2r is the first positive repair of the H2q composition boundary. It adds a narrow controller-side composed route gate after H2p: stale `refine_selection` calls are rewritten when the latest prompt explicitly says to ignore old selections, and requested surface classes are restored when same-value comments, banners, switches, or archived labels are marked as nearby context.

On H2q, H2r reaches `8 / 8` strict and executor-equivalent while H2p was `3 / 8`. This is strong local mechanism evidence. Transfer backtests are now positive on the current packet set, so the next promotion gate is a fresh H2s composed holdout.

![H2r composed route-gating gate](figures/h2r_composed_route_gating_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2q_h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2e_execute_v1 | 8 | 1 | 0.125 | 2 | 0.25 |
| h2q_h2n_scoped_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_scoped_target_query_normalization | results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2n_execute_v1 | 8 | 0 | 0.0 | 1 | 0.125 |
| h2q_h2o_value_bearing_target_query_synthesis | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis | results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2o_execute_v1 | 8 | 2 | 0.25 | 2 | 0.25 |
| h2q_h2p_contextual_surface_alias_routing | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing | results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2p_execute_v1 | 8 | 3 | 0.375 | 3 | 0.375 |
| h2q_h2r_composed_route_gating | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating | results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h2q_execute_v2 | 8 | 8 | 1.0 | 8 | 1.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2q_h2r_vs_h2p | results/tool_probe_replay_live_comparisons/20260512T_h2r_composed_route_gating_vs_h2p_on_h2q_v2 | 8 | 0.375 | 1.0 | 0.625 | 0.375 | 1.0 | 0.625 |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| h2q_h2e_route_arbitration | h2q_contextual_alias_decoy_overlap | 2 | 1 | 0.5 | 1 | 0.5 |
| h2q_h2e_route_arbitration | h2q_stale_surface_alias | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2e_route_arbitration | h2q_surface_alias_value_decoy | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2e_route_arbitration | h2q_value_bearing_stale_decoy | 2 | 0 | 0.0 | 1 | 0.5 |
| h2q_h2n_scoped_target_query_normalization | h2q_contextual_alias_decoy_overlap | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2n_scoped_target_query_normalization | h2q_stale_surface_alias | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2n_scoped_target_query_normalization | h2q_surface_alias_value_decoy | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2n_scoped_target_query_normalization | h2q_value_bearing_stale_decoy | 2 | 0 | 0.0 | 1 | 0.5 |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_contextual_alias_decoy_overlap | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_stale_surface_alias | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_surface_alias_value_decoy | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_value_bearing_stale_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2p_contextual_surface_alias_routing | h2q_contextual_alias_decoy_overlap | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2p_contextual_surface_alias_routing | h2q_stale_surface_alias | 2 | 0 | 0.0 | 0 | 0.0 |
| h2q_h2p_contextual_surface_alias_routing | h2q_surface_alias_value_decoy | 2 | 1 | 0.5 | 1 | 0.5 |
| h2q_h2p_contextual_surface_alias_routing | h2q_value_bearing_stale_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2r_composed_route_gating | h2q_contextual_alias_decoy_overlap | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2r_composed_route_gating | h2q_stale_surface_alias | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2r_composed_route_gating | h2q_surface_alias_value_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2r_composed_route_gating | h2q_value_bearing_stale_decoy | 2 | 2 | 1.0 | 2 | 1.0 |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2q_h2e_route_arbitration | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | argument_mismatch | False | extract_layout | result tile | extract_layout | Blocked badge |
| h2q_h2e_route_arbitration | h2q_state_panel_closed_value_tag_decoy | h2q_surface_alias_value_decoy | argument_mismatch | False | extract_layout | state panel | extract_layout | Closed |
| h2q_h2e_route_arbitration | h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | executable_paraphrase | True | extract_layout | priority badge Critical | extract_layout | Critical priority badge |
| h2q_h2e_route_arbitration | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | argument_mismatch | False | extract_layout | owner field Amina | extract_layout | owner field |
| h2q_h2e_route_arbitration | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | error notice | extract_layout | archived exception panel |
| h2q_h2e_route_arbitration | h2q_result_tile_stale_selection_hint | h2q_stale_surface_alias | wrong_tool | False | extract_layout | result tile | refine_selection |  |
| h2q_h2e_route_arbitration | h2q_state_panel_stale_selection_hint | h2q_stale_surface_alias | wrong_tool | False | extract_layout | state panel | refine_selection |  |
| h2q_h2n_scoped_target_query_normalization | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | argument_mismatch | False | extract_layout | result tile | extract_layout | result comment |
| h2q_h2n_scoped_target_query_normalization | h2q_state_panel_closed_value_tag_decoy | h2q_surface_alias_value_decoy | argument_mismatch | False | extract_layout | state panel | extract_layout | Closed |
| h2q_h2n_scoped_target_query_normalization | h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | executable_paraphrase | True | extract_layout | priority badge Critical | extract_layout | Critical priority badge |
| h2q_h2n_scoped_target_query_normalization | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | argument_mismatch | False | extract_layout | owner field Amina | extract_layout | owner field |
| h2q_h2n_scoped_target_query_normalization | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | error notice | extract_layout | error banner |
| h2q_h2n_scoped_target_query_normalization | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | mode field | extract_layout | mode switch |
| h2q_h2n_scoped_target_query_normalization | h2q_result_tile_stale_selection_hint | h2q_stale_surface_alias | wrong_tool | False | extract_layout | result tile | refine_selection |  |
| h2q_h2n_scoped_target_query_normalization | h2q_state_panel_stale_selection_hint | h2q_stale_surface_alias | wrong_tool | False | extract_layout | state panel | refine_selection |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | argument_mismatch | False | extract_layout | result tile | extract_layout | result comment |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_state_panel_closed_value_tag_decoy | h2q_surface_alias_value_decoy | argument_mismatch | False | extract_layout | state panel | extract_layout | Closed |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | error notice | extract_layout | error banner |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | mode field | extract_layout | mode switch |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_result_tile_stale_selection_hint | h2q_stale_surface_alias | wrong_tool | False | extract_layout | result tile | refine_selection |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_state_panel_stale_selection_hint | h2q_stale_surface_alias | wrong_tool | False | extract_layout | state panel | refine_selection |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | argument_mismatch | False | extract_layout | result tile | extract_layout | result comment |
| h2q_h2p_contextual_surface_alias_routing | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | error notice | extract_layout | error banner |
| h2q_h2p_contextual_surface_alias_routing | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | mode field | extract_layout | mode switch |
| h2q_h2p_contextual_surface_alias_routing | h2q_result_tile_stale_selection_hint | h2q_stale_surface_alias | wrong_tool | False | extract_layout | result tile | refine_selection |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_state_panel_stale_selection_hint | h2q_stale_surface_alias | wrong_tool | False | extract_layout | state panel | refine_selection |  |

## Controller Intervention Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label | value_bearing_label | display_value | surface_label | requested_label | requested_region_id | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2q_h2e_route_arbitration | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |
| h2q_h2n_scoped_target_query_normalization | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"Blocked badge"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | result comment |  |  |  |  |  |  |
| h2q_h2n_scoped_target_query_normalization | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |
| h2q_h2n_scoped_target_query_normalization | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"archived exception panel"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | error banner |  |  |  |  |  |  |
| h2q_h2n_scoped_target_query_normalization | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode switch"} | mode switch |  |  |  |  |  |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"Blocked badge"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | result comment |  |  |  |  |  |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"Critical priority badge"} | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"priority badge Critical"} | priority badge | priority badge Critical |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field Amina"} | owner field | owner field Amina |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"archived exception panel"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | error banner |  |  |  |  |  |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode switch"} | mode switch |  |  |  |  |  |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"Blocked badge"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | result comment |  |  |  |  |  |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_state_panel_closed_value_tag_decoy | h2q_surface_alias_value_decoy | visual_contextual_surface_alias_routing | extract_layout | {"image_id":"img-h2q-state-panel-closed","target_query":"Closed"} | extract_layout | {"image_id":"img-h2q-state-panel-closed","target_query":"state panel"} |  |  | Closed | state panel |  |  | contextual_surface_alias_recoverable |
| h2q_h2p_contextual_surface_alias_routing | h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"Critical priority badge"} | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"priority badge Critical"} | priority badge | priority badge Critical |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2p_contextual_surface_alias_routing | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field Amina"} | owner field | owner field Amina |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2p_contextual_surface_alias_routing | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"archived exception panel"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | error banner |  |  |  |  |  |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode switch"} | mode switch |  |  |  |  |  |  |
| h2q_h2r_composed_route_gating | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_composed_route_gating | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result tile"} |  |  |  |  | result tile | h2q-result-tile-blocked-13002 | requested_surface_over_deprioritized_decoy |
| h2q_h2r_composed_route_gating | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"Blocked badge"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | result comment |  |  |  |  |  |  |
| h2q_h2r_composed_route_gating | h2q_state_panel_closed_value_tag_decoy | h2q_surface_alias_value_decoy | visual_contextual_surface_alias_routing | extract_layout | {"image_id":"img-h2q-state-panel-closed","target_query":"Closed"} | extract_layout | {"image_id":"img-h2q-state-panel-closed","target_query":"state panel"} |  |  | Closed | state panel |  |  | contextual_surface_alias_recoverable |
| h2q_h2r_composed_route_gating | h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"Critical priority badge"} | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"priority badge Critical"} | priority badge | priority badge Critical |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2r_composed_route_gating | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field Amina"} | owner field | owner field Amina |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2r_composed_route_gating | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |
| h2q_h2r_composed_route_gating | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_composed_route_gating | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error notice"} |  |  |  |  | error notice | h2q-error-notice-13042 | requested_surface_over_deprioritized_decoy |
| h2q_h2r_composed_route_gating | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"archived exception panel"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | error banner |  |  |  |  |  |  |
| h2q_h2r_composed_route_gating | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_composed_route_gating | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode switch"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} |  |  |  |  | mode field | h2q-mode-field-13052 | requested_surface_over_deprioritized_decoy |
| h2q_h2r_composed_route_gating | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode switch"} | mode switch |  |  |  |  |  |  |
| h2q_h2r_composed_route_gating | h2q_result_tile_stale_selection_hint | h2q_stale_surface_alias | visual_composed_route_gating | refine_selection | {"filter_query":"Blocked","selection_id":"sel-archived-result-badge"} | extract_layout | {"image_id":"img-h2q-result-tile-stale-selection","target_query":"result tile"} |  |  |  |  | result tile | h2q-current-result-tile-13062 | stale_selection_to_requested_surface |
| h2q_h2r_composed_route_gating | h2q_state_panel_stale_selection_hint | h2q_stale_surface_alias | visual_composed_route_gating | refine_selection | {"filter_query":"Closed","selection_id":"sel-archived-state-tag"} | extract_layout | {"image_id":"img-h2q-state-panel-stale-selection","target_query":"state panel"} |  |  |  |  | state panel | h2q-current-state-panel-13072 | stale_selection_to_requested_surface |

## Findings

| finding_id | finding |
| --- | --- |
| h2r_solves_h2q_local_boundary | H2r reaches 8/8 strict and 8/8 executor-equivalent on H2q, improving over H2p by 0.625 exact-rate and executor-equivalence. |
| h2r_matches_h2q_failure_cardinality | H2p left 5 non-exact rows; H2r records 5 composed-route interventions and leaves 0 non-exact rows. |
| h2r_mechanism_splits_stale_selection_and_decoy_surface_routes | Composed route gating fires on 2 stale-selection rows and 3 same-value decoy surface rows, showing the boundary was a route-selection problem rather than only label spelling. |
| h2r_transfer_backtested_but_needs_fresh_h2s | H2r is now transfer-positive on the current packet set, but it remains a local H2q-derived repair until a fresh H2s composed holdout confirms the policy without further tuning. |
