# H2s Fresh Composed Holdout Synthesis

Generated: `2026-05-12T23:36:50.694894+00:00`

## Summary

H2s is the first fresh holdout built after H2r passed the current transfer backtest. H2r was frozen for the first run, then H2p, H2o, and H2j controls were executed on the same packet.

H2r reaches `10 / 10` strict and `10 / 10` executor-equivalent. H2p and H2o each reach `3 / 10`; H2j reaches `1 / 10`. The H2r-vs-H2p gain is `0.7` exact-rate and executor-equivalence-rate.

This is fresh positive evidence for composed route gating. It should still be kept as scoped internal evidence: the next step is a harder H2t holdout or packaged workflow transfer, not another edit to H2r.

![H2s fresh composed holdout gate](figures/h2s_fresh_composed_holdout_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2s_h2j_target_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2j_execute_v1 | 10 | 1 | 0.1 | 1 | 0.1 |
| h2s_h2o_value_bearing_synthesis | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis | results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2o_execute_v1 | 10 | 3 | 0.3 | 3 | 0.3 |
| h2s_h2p_contextual_surface_alias_routing | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing | results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2p_execute_v1 | 10 | 3 | 0.3 | 3 | 0.3 |
| h2s_h2r_composed_route_gating | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating | results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2r_execute_v1 | 10 | 10 | 1.0 | 10 | 1.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2s_h2r_vs_h2p | results/tool_probe_replay_live_comparisons/20260512T_h2s_fresh_composed_holdout_h2r_vs_h2p_v1 | 10 | 0.3 | 1.0 | 0.7 | 0.3 | 1.0 | 0.7 |
| h2s_h2r_vs_h2o | results/tool_probe_replay_live_comparisons/20260512T_h2s_fresh_composed_holdout_h2r_vs_h2o_v1 | 10 | 0.3 | 1.0 | 0.7 | 0.3 | 1.0 | 0.7 |
| h2s_h2r_vs_h2j | results/tool_probe_replay_live_comparisons/20260512T_h2s_fresh_composed_holdout_h2r_vs_h2j_v1 | 10 | 0.1 | 1.0 | 0.9 | 0.1 | 1.0 | 0.9 |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| h2s_h2j_target_normalization | h2s_clean_route_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2s_h2j_target_normalization | h2s_contextual_alias_decoy_overlap | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2j_target_normalization | h2s_negated_decoy_guard | 1 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2j_target_normalization | h2s_stale_surface_alias | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2j_target_normalization | h2s_surface_alias_same_value_decoy | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2j_target_normalization | h2s_value_bearing_stale_decoy | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2o_value_bearing_synthesis | h2s_clean_route_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2s_h2o_value_bearing_synthesis | h2s_contextual_alias_decoy_overlap | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2o_value_bearing_synthesis | h2s_negated_decoy_guard | 1 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2o_value_bearing_synthesis | h2s_stale_surface_alias | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2o_value_bearing_synthesis | h2s_surface_alias_same_value_decoy | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2o_value_bearing_synthesis | h2s_value_bearing_stale_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2s_h2p_contextual_surface_alias_routing | h2s_clean_route_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2s_h2p_contextual_surface_alias_routing | h2s_contextual_alias_decoy_overlap | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2p_contextual_surface_alias_routing | h2s_negated_decoy_guard | 1 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2p_contextual_surface_alias_routing | h2s_stale_surface_alias | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2p_contextual_surface_alias_routing | h2s_surface_alias_same_value_decoy | 2 | 0 | 0.0 | 0 | 0.0 |
| h2s_h2p_contextual_surface_alias_routing | h2s_value_bearing_stale_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2s_h2r_composed_route_gating | h2s_clean_route_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2s_h2r_composed_route_gating | h2s_contextual_alias_decoy_overlap | 2 | 2 | 1.0 | 2 | 1.0 |
| h2s_h2r_composed_route_gating | h2s_negated_decoy_guard | 1 | 1 | 1.0 | 1 | 1.0 |
| h2s_h2r_composed_route_gating | h2s_stale_surface_alias | 2 | 2 | 1.0 | 2 | 1.0 |
| h2s_h2r_composed_route_gating | h2s_surface_alias_same_value_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2s_h2r_composed_route_gating | h2s_value_bearing_stale_decoy | 2 | 2 | 1.0 | 2 | 1.0 |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2s_h2j_target_normalization | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | argument_mismatch | False | extract_layout | review tile | extract_layout | review note |
| h2s_h2j_target_normalization | h2s_signal_panel_green_tag_marker_decoys | h2s_surface_alias_same_value_decoy | argument_mismatch | False | extract_layout | signal panel | extract_layout | Green signal tag |
| h2s_h2j_target_normalization | h2s_severity_pill_critical_archived_badge_decoy | h2s_value_bearing_stale_decoy | argument_mismatch | False | extract_layout | severity pill Critical | extract_layout | severity pill |
| h2s_h2j_target_normalization | h2s_reviewer_field_malik_old_owner_decoy | h2s_value_bearing_stale_decoy | argument_mismatch | False | extract_layout | reviewer field Malik | extract_layout | reviewer field |
| h2s_h2j_target_normalization | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | timeout exception notice | extract_layout | timeout banner |
| h2s_h2j_target_normalization | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | delivery field | extract_layout | paused toggle |
| h2s_h2j_target_normalization | h2s_result_panel_waiting_stale_selection_hint | h2s_stale_surface_alias | wrong_tool | False | extract_layout | result panel | refine_selection |  |
| h2s_h2j_target_normalization | h2s_status_tile_ready_stale_selection_hint | h2s_stale_surface_alias | wrong_tool | False | extract_layout | status tile | refine_selection |  |
| h2s_h2j_target_normalization | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | argument_mismatch | False | extract_layout | approval panel | extract_layout | approval note |
| h2s_h2o_value_bearing_synthesis | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | argument_mismatch | False | extract_layout | review tile | extract_layout | review note |
| h2s_h2o_value_bearing_synthesis | h2s_signal_panel_green_tag_marker_decoys | h2s_surface_alias_same_value_decoy | argument_mismatch | False | extract_layout | signal panel | extract_layout | Green signal tag |
| h2s_h2o_value_bearing_synthesis | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | timeout exception notice | extract_layout | timeout banner |
| h2s_h2o_value_bearing_synthesis | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | delivery field | extract_layout | paused toggle |
| h2s_h2o_value_bearing_synthesis | h2s_result_panel_waiting_stale_selection_hint | h2s_stale_surface_alias | wrong_tool | False | extract_layout | result panel | refine_selection |  |
| h2s_h2o_value_bearing_synthesis | h2s_status_tile_ready_stale_selection_hint | h2s_stale_surface_alias | wrong_tool | False | extract_layout | status tile | refine_selection |  |
| h2s_h2o_value_bearing_synthesis | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | argument_mismatch | False | extract_layout | approval panel | extract_layout | approval note |
| h2s_h2p_contextual_surface_alias_routing | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | argument_mismatch | False | extract_layout | review tile | extract_layout | review note |
| h2s_h2p_contextual_surface_alias_routing | h2s_signal_panel_green_tag_marker_decoys | h2s_surface_alias_same_value_decoy | argument_mismatch | False | extract_layout | signal panel | extract_layout | Green signal tag |
| h2s_h2p_contextual_surface_alias_routing | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | timeout exception notice | extract_layout | timeout banner |
| h2s_h2p_contextual_surface_alias_routing | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | argument_mismatch | False | extract_layout | delivery field | extract_layout | paused toggle |
| h2s_h2p_contextual_surface_alias_routing | h2s_result_panel_waiting_stale_selection_hint | h2s_stale_surface_alias | wrong_tool | False | extract_layout | result panel | refine_selection |  |
| h2s_h2p_contextual_surface_alias_routing | h2s_status_tile_ready_stale_selection_hint | h2s_stale_surface_alias | wrong_tool | False | extract_layout | status tile | refine_selection |  |
| h2s_h2p_contextual_surface_alias_routing | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | argument_mismatch | False | extract_layout | approval panel | extract_layout | approval note |

## Controller Intervention Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label | value_bearing_label | display_value | surface_label | requested_label | requested_region_id | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2s_h2j_target_normalization | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"Waiting review chip"} | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review note"} | review note |  |  |  |  |  |  |
| h2s_h2j_target_normalization | h2s_severity_pill_critical_archived_badge_decoy | h2s_value_bearing_stale_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"Critical severity pill"} | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"severity pill"} | severity pill |  |  |  |  |  |  |
| h2s_h2j_target_normalization | h2s_reviewer_field_malik_old_owner_decoy | h2s_value_bearing_stale_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"Malik reviewer field"} | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"reviewer field"} | reviewer field |  |  |  |  |  |  |
| h2s_h2j_target_normalization | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"archived runbook card"} | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout banner"} | timeout banner |  |  |  |  |  |  |
| h2s_h2j_target_normalization | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"delivery field"} | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"paused toggle"} | paused toggle |  |  |  |  |  |  |
| h2s_h2j_target_normalization | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"panel Pending"} | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval note"} | approval note |  |  |  |  |  |  |
| h2s_h2o_value_bearing_synthesis | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"Waiting review chip"} | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review note"} | review note |  |  |  |  |  |  |
| h2s_h2o_value_bearing_synthesis | h2s_severity_pill_critical_archived_badge_decoy | h2s_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"Critical severity pill"} | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"severity pill Critical"} | severity pill | severity pill Critical |  |  |  |  | value_bearing_label_recoverable |
| h2s_h2o_value_bearing_synthesis | h2s_reviewer_field_malik_old_owner_decoy | h2s_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"Malik reviewer field"} | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"reviewer field Malik"} | reviewer field | reviewer field Malik |  |  |  |  | value_bearing_label_recoverable |
| h2s_h2o_value_bearing_synthesis | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"archived runbook card"} | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout banner"} | timeout banner |  |  |  |  |  |  |
| h2s_h2o_value_bearing_synthesis | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"delivery field"} | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"paused toggle"} | paused toggle |  |  |  |  |  |  |
| h2s_h2o_value_bearing_synthesis | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"panel Pending"} | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval note"} | approval note |  |  |  |  |  |  |
| h2s_h2p_contextual_surface_alias_routing | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"Waiting review chip"} | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review note"} | review note |  |  |  |  |  |  |
| h2s_h2p_contextual_surface_alias_routing | h2s_severity_pill_critical_archived_badge_decoy | h2s_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"Critical severity pill"} | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"severity pill Critical"} | severity pill | severity pill Critical |  |  |  |  | value_bearing_label_recoverable |
| h2s_h2p_contextual_surface_alias_routing | h2s_reviewer_field_malik_old_owner_decoy | h2s_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"Malik reviewer field"} | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"reviewer field Malik"} | reviewer field | reviewer field Malik |  |  |  |  | value_bearing_label_recoverable |
| h2s_h2p_contextual_surface_alias_routing | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"archived runbook card"} | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout banner"} | timeout banner |  |  |  |  |  |  |
| h2s_h2p_contextual_surface_alias_routing | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"delivery field"} | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"paused toggle"} | paused toggle |  |  |  |  |  |  |
| h2s_h2p_contextual_surface_alias_routing | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"panel Pending"} | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval note"} | approval note |  |  |  |  |  |  |
| h2s_h2r_composed_route_gating | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review note"} | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review tile"} |  |  |  |  | review tile | h2s-review-tile-waiting-14002 | requested_surface_over_deprioritized_decoy |
| h2s_h2r_composed_route_gating | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"Waiting review chip"} | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review note"} | review note |  |  |  |  |  |  |
| h2s_h2r_composed_route_gating | h2s_signal_panel_green_tag_marker_decoys | h2s_surface_alias_same_value_decoy | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-signal-panel-green","target_query":"Green signal tag"} | extract_layout | {"image_id":"img-h2s-signal-panel-green","target_query":"signal panel"} |  |  |  |  | signal panel | h2s-signal-panel-green-14012 | requested_surface_over_deprioritized_decoy |
| h2s_h2r_composed_route_gating | h2s_severity_pill_critical_archived_badge_decoy | h2s_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"Critical severity pill"} | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"severity pill Critical"} | severity pill | severity pill Critical |  |  |  |  | value_bearing_label_recoverable |
| h2s_h2r_composed_route_gating | h2s_reviewer_field_malik_old_owner_decoy | h2s_value_bearing_stale_decoy | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"Malik reviewer field"} | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"reviewer field Malik"} | reviewer field | reviewer field Malik |  |  |  |  | value_bearing_label_recoverable |
| h2s_h2r_composed_route_gating | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout banner"} | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout exception notice"} |  |  |  |  | timeout exception notice | h2s-timeout-exception-notice-14042 | requested_surface_over_deprioritized_decoy |
| h2s_h2r_composed_route_gating | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"archived runbook card"} | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout banner"} | timeout banner |  |  |  |  |  |  |
| h2s_h2r_composed_route_gating | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"paused toggle"} | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"delivery field"} |  |  |  |  | delivery field | h2s-delivery-field-14052 | requested_surface_over_deprioritized_decoy |
| h2s_h2r_composed_route_gating | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"delivery field"} | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"paused toggle"} | paused toggle |  |  |  |  |  |  |
| h2s_h2r_composed_route_gating | h2s_result_panel_waiting_stale_selection_hint | h2s_stale_surface_alias | visual_composed_route_gating | refine_selection | {"filter_query":"Waiting","selection_id":"sel-archive-result-chip"} | extract_layout | {"image_id":"img-h2s-result-panel-stale-selection","target_query":"result panel"} |  |  |  |  | result panel | h2s-current-result-panel-14062 | stale_selection_to_requested_surface |
| h2s_h2r_composed_route_gating | h2s_status_tile_ready_stale_selection_hint | h2s_stale_surface_alias | visual_composed_route_gating | refine_selection | {"filter_query":"Ready","selection_id":"sel-old-status-tag"} | extract_layout | {"image_id":"img-h2s-status-tile-stale-selection","target_query":"status tile"} |  |  |  |  | status tile | h2s-current-status-tile-14072 | stale_selection_to_requested_surface |
| h2s_h2r_composed_route_gating | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval note"} | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval panel"} |  |  |  |  | approval panel | h2s-approval-panel-pending-14082 | requested_surface_over_deprioritized_decoy |
| h2s_h2r_composed_route_gating | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"panel Pending"} | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval note"} | approval note |  |  |  |  |  |  |

## Findings

| finding_id | finding |
| --- | --- |
| h2s_fresh_holdout_confirms_h2r_transfer | H2r reaches 10/10 strict and 10/10 executor-equivalent on fresh H2s, while H2p reaches 3/10, H2o reaches 3/10, and H2j reaches 1/10. |
| h2s_composed_route_gate_is_causal | H2r improves over H2p by 0.7 exact-rate and executor-equivalence rate, and over H2j by 0.9 exact-rate, on an unseen composed packet. |
| h2s_h2r_mechanism_is_mixed_not_single_helper | H2r uses a mixed controller path on H2s: 7 composed route gates, 2 value-bearing syntheses, and 4 target normalizations. |
| h2s_clean_control_does_not_need_visual_helper | The clean status-badge control remains exact without H2r-specific metadata, with 0 recorded helper rows for that control. |
| h2s_next_requires_h2t_or_packaged_transfer | H2r leaves 0 non-exact rows on H2s. The next research move is no longer patching H2r on this slice; it is either a harder H2t holdout or a packaged workflow transfer that preserves the same composed-route pressure. |
