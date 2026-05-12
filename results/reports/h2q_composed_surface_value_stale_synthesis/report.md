# H2q Composed Surface/Value/Stale Synthesis

Generated: `2026-05-12T21:51:14.487948+00:00`

## Summary

H2q is the first post-H2p saturation breaker. It composes surface aliases, value-bearing labels, stale-selection hints, and decoy overlap in one replay packet. H2p remains the strongest current controller stack, but reaches only 3/8 strict and executor-equivalent, so the research target has moved from isolated surface/value repair to composed route gating.

![H2q composed surface/value/stale gate](figures/h2q_composed_surface_value_stale_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2q_h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2e_execute_v1 | 8 | 1 | 0.125 | 2 | 0.25 |
| h2q_h2n_scoped_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_scoped_target_query_normalization | results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2n_execute_v1 | 8 | 0 | 0.0 | 1 | 0.125 |
| h2q_h2o_value_bearing_target_query_synthesis | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis | results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2o_execute_v1 | 8 | 2 | 0.25 | 2 | 0.25 |
| h2q_h2p_contextual_surface_alias_routing | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing | results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2p_execute_v1 | 8 | 3 | 0.375 | 3 | 0.375 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2q_h2p_vs_h2o | results/tool_probe_replay_live_comparisons/20260512T_h2q_composed_surface_value_stale_h2p_vs_h2o_v1 | 8 | 0.25 | 0.375 | 0.125 | 0.25 | 0.375 | 0.125 |
| h2q_h2p_vs_h2n | results/tool_probe_replay_live_comparisons/20260512T_h2q_composed_surface_value_stale_h2p_vs_h2n_v1 | 8 | 0.0 | 0.375 | 0.375 | 0.125 | 0.375 | 0.25 |
| h2q_h2p_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2q_composed_surface_value_stale_h2p_vs_h2e_v1 | 8 | 0.125 | 0.375 | 0.25 | 0.25 | 0.375 | 0.125 |

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

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label | value_bearing_label | value_suffix | matched_phrase | display_value | surface_label | surface_text | surface_region_id | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2q_h2e_route_arbitration | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |  |  |
| h2q_h2n_scoped_target_query_normalization | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"Blocked badge"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | result comment |  |  |  |  |  |  |  |  |
| h2q_h2n_scoped_target_query_normalization | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |  |  |
| h2q_h2n_scoped_target_query_normalization | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"archived exception panel"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | error banner |  |  |  |  |  |  |  |  |
| h2q_h2n_scoped_target_query_normalization | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode switch"} | mode switch |  |  |  |  |  |  |  |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"Blocked badge"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | result comment |  |  |  |  |  |  |  |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"Critical priority badge"} | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"priority badge Critical"} | priority badge | priority badge Critical | critical | critical priority badge |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field Amina"} | owner field | owner field Amina | amina | amina owner field |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |  |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"archived exception panel"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | error banner |  |  |  |  |  |  |  |  |
| h2q_h2o_value_bearing_target_query_synthesis | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode switch"} | mode switch |  |  |  |  |  |  |  |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"Blocked badge"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | result comment |  |  |  |  |  |  |  |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_state_panel_closed_value_tag_decoy | h2q_surface_alias_value_decoy | visual_contextual_surface_alias_routing | extract_layout | {"image_id":"img-h2q-state-panel-closed","target_query":"Closed"} | extract_layout | {"image_id":"img-h2q-state-panel-closed","target_query":"state panel"} |  |  |  |  | Closed | state panel | Closed | h2q-state-panel-closed-13012 | contextual_surface_alias_recoverable |
| h2q_h2p_contextual_surface_alias_routing | h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"Critical priority badge"} | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"priority badge Critical"} | priority badge | priority badge Critical | critical | critical priority badge |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2p_contextual_surface_alias_routing | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field Amina"} | owner field | owner field Amina | amina | amina owner field |  |  |  |  | value_bearing_label_recoverable |
| h2q_h2p_contextual_surface_alias_routing | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field"} |  |  |  |  |  |  |  |  |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"archived exception panel"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | error banner |  |  |  |  |  |  |  |  |
| h2q_h2p_contextual_surface_alias_routing | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode switch"} | mode switch |  |  |  |  |  |  |  |  |

## Findings

| finding_id | finding |
| --- | --- |
| h2q_breaks_h2p_saturation | H2q breaks the post-H2p H2m saturation: H2p reaches only 3/8 strict and 3/8 executor-equivalent on the composed surface/value/stale packet. |
| h2q_h2p_remains_directionally_best | H2p is still the best current row: H2o is 2/8, H2n is 0/8, and H2e is 1/8 strict. H2p adds 0.125 strict over H2o, 0.375 over H2n, and 0.25 over H2e. |
| h2q_failures_are_tool_route_and_decoy_selection_failures | H2p leaves 5 non-exact rows: 3 argument mismatches and 2 wrong-tool rows, so remaining error is not merely strict spelling drift. |
| h2q_composition_exposes_incomplete_helper_interaction | H2p records 1 contextual surface-alias, 2 value-bearing, 3 target-normalization, and 1 stale-selection interventions, but still fails five rows under composed pressure. |
| next_slice_should_target_composed_route_gating | The next slice should target composed route gating: refuse stale refine_selection calls when the prompt says ignore old selection IDs, and prioritize requested surface classes over nearby same-value comments, banners, controls, and history context. |
