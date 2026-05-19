# H2y Scaled CLI Semantic Pressure Synthesis

Generated: `2026-05-19T19:30:30.408671+00:00`

## Summary

H2y scales the H2x packaged/CLI semantic-pressure gate to sixteen cases across quoted stale negation, stale selection negation, instructional negation, and genuine displayed negated values. It preserves the replay-live attribution path and runs matched no-fallback controls.

H2u reaches `4 / 16` strict and `5 / 16` executor-equivalent. H2w reaches `12 / 16` on both metrics, a `0.5` exact-rate gain and `0.4375` executor-equivalence gain.

The no-fallback rows tie their full-controller rows, so fallback is still not the causal helper. The important new result is that H2w is no longer saturated: all stale-selection negation rows remain unresolved, and one value-before-component row collapses to a short component query.

![H2y scaled CLI semantic pressure gate](figures/h2y_scaled_cli_semantic_pressure_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2y_h2u_negation_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard | results/tool_probe_replay_live/20260519T_h2y_scaled_cli_semantic_pressure_h2u_execute_v1 | 16 | 4 | 0.25 | 5 | 0.3125 |
| h2y_h2u_no_controller_fallback | mlx_gemma4_e2b_reasoner_only_h2u_no_controller_fallback | results/tool_probe_replay_live/20260519T_h2y_scaled_cli_semantic_pressure_h2u_no_fallback_execute_v1 | 16 | 4 | 0.25 | 5 | 0.3125 |
| h2y_h2w_semantic_target_preservation | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard_visual_semantic_target_preservation | results/tool_probe_replay_live/20260519T_h2y_scaled_cli_semantic_pressure_h2w_execute_v1 | 16 | 12 | 0.75 | 12 | 0.75 |
| h2y_h2w_no_controller_fallback | mlx_gemma4_e2b_reasoner_only_h2w_no_controller_fallback | results/tool_probe_replay_live/20260519T_h2y_scaled_cli_semantic_pressure_h2w_no_fallback_execute_v1 | 16 | 12 | 0.75 | 12 | 0.75 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2y_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260519T_h2y_scaled_cli_semantic_pressure_h2w_vs_h2u_v1 | 16 | 0.25 | 0.75 | 0.5 | 0.3125 | 0.75 | 0.4375 |
| h2y_h2u_no_fallback_vs_h2u | results/tool_probe_replay_live_comparisons/20260519T_h2y_scaled_cli_semantic_pressure_h2u_no_fallback_vs_h2u_v1 | 16 | 0.25 | 0.25 | 0.0 | 0.3125 | 0.3125 | 0.0 |
| h2y_h2w_no_fallback_vs_h2w | results/tool_probe_replay_live_comparisons/20260519T_h2y_scaled_cli_semantic_pressure_h2w_no_fallback_vs_h2w_v1 | 16 | 0.75 | 0.75 | 0.0 | 0.75 | 0.75 | 0.0 |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| h2y_h2u_negation_guard | h2y_genuine_negated_target_value | 7 | 0 | 0.0 | 1 | 0.14285714285714285 |
| h2y_h2u_negation_guard | h2y_instructional_negation_context | 3 | 3 | 1.0 | 3 | 1.0 |
| h2y_h2u_negation_guard | h2y_quoted_stale_negation_context | 3 | 1 | 0.3333333333333333 | 1 | 0.3333333333333333 |
| h2y_h2u_negation_guard | h2y_stale_selection_negation_context | 3 | 0 | 0.0 | 0 | 0.0 |
| h2y_h2u_no_controller_fallback | h2y_genuine_negated_target_value | 7 | 0 | 0.0 | 1 | 0.14285714285714285 |
| h2y_h2u_no_controller_fallback | h2y_instructional_negation_context | 3 | 3 | 1.0 | 3 | 1.0 |
| h2y_h2u_no_controller_fallback | h2y_quoted_stale_negation_context | 3 | 1 | 0.3333333333333333 | 1 | 0.3333333333333333 |
| h2y_h2u_no_controller_fallback | h2y_stale_selection_negation_context | 3 | 0 | 0.0 | 0 | 0.0 |
| h2y_h2w_semantic_target_preservation | h2y_genuine_negated_target_value | 7 | 6 | 0.8571428571428571 | 6 | 0.8571428571428571 |
| h2y_h2w_semantic_target_preservation | h2y_instructional_negation_context | 3 | 3 | 1.0 | 3 | 1.0 |
| h2y_h2w_semantic_target_preservation | h2y_quoted_stale_negation_context | 3 | 3 | 1.0 | 3 | 1.0 |
| h2y_h2w_semantic_target_preservation | h2y_stale_selection_negation_context | 3 | 0 | 0.0 | 0 | 0.0 |
| h2y_h2w_no_controller_fallback | h2y_genuine_negated_target_value | 7 | 6 | 0.8571428571428571 | 6 | 0.8571428571428571 |
| h2y_h2w_no_controller_fallback | h2y_instructional_negation_context | 3 | 3 | 1.0 | 3 | 1.0 |
| h2y_h2w_no_controller_fallback | h2y_quoted_stale_negation_context | 3 | 3 | 1.0 | 3 | 1.0 |
| h2y_h2w_no_controller_fallback | h2y_stale_selection_negation_context | 3 | 0 | 0.0 | 0 | 0.0 |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2y_h2u_negation_guard | h2y_action_banner_quoted_not_banner_note | h2y_quoted_stale_negation_context | argument_mismatch | False | extract_layout | action banner | extract_layout | prior note |
| h2y_h2u_negation_guard | h2y_resolution_marker_old_quote_not_marker | h2y_quoted_stale_negation_context | argument_mismatch | False | extract_layout | resolution marker | extract_layout | audit memo |
| h2y_h2u_negation_guard | h2y_escalation_lane_stale_selection_not_lane | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | escalation lane | refine_selection |  |
| h2y_h2u_negation_guard | h2y_exception_panel_stale_selection_not_panel | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | exception panel | refine_selection |  |
| h2y_h2u_negation_guard | h2y_approval_field_stale_selection_not_field | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | approval field | refine_selection |  |
| h2y_h2u_negation_guard | h2y_not_replied_status_pill_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | status pill not replied | extract_layout | not replied |
| h2y_h2u_negation_guard | h2y_not_sent_delivery_tag_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | delivery tag not sent | extract_layout | not sent |
| h2y_h2u_negation_guard | h2y_not_required_approval_marker_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | approval marker not required | extract_layout | Not required |
| h2y_h2u_negation_guard | h2y_not_active_alert_banner_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | alert banner not active | extract_layout | alert |
| h2y_h2u_negation_guard | h2y_not_escalated_risk_chip_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | risk chip not escalated | extract_layout | risk chip |
| h2y_h2u_negation_guard | h2y_not_available_owner_field_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | owner field not available | extract_layout | not available |
| h2y_h2u_negation_guard | h2y_not_started_phase_tile_value_before_component | h2y_genuine_negated_target_value | executable_paraphrase | True | extract_layout | phase tile not started | extract_layout | phase tile Not started |
| h2y_h2u_no_controller_fallback | h2y_action_banner_quoted_not_banner_note | h2y_quoted_stale_negation_context | argument_mismatch | False | extract_layout | action banner | extract_layout | prior note |
| h2y_h2u_no_controller_fallback | h2y_resolution_marker_old_quote_not_marker | h2y_quoted_stale_negation_context | argument_mismatch | False | extract_layout | resolution marker | extract_layout | audit memo |
| h2y_h2u_no_controller_fallback | h2y_escalation_lane_stale_selection_not_lane | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | escalation lane | refine_selection |  |
| h2y_h2u_no_controller_fallback | h2y_exception_panel_stale_selection_not_panel | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | exception panel | refine_selection |  |
| h2y_h2u_no_controller_fallback | h2y_approval_field_stale_selection_not_field | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | approval field | refine_selection |  |
| h2y_h2u_no_controller_fallback | h2y_not_replied_status_pill_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | status pill not replied | extract_layout | not replied |
| h2y_h2u_no_controller_fallback | h2y_not_sent_delivery_tag_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | delivery tag not sent | extract_layout | not sent |
| h2y_h2u_no_controller_fallback | h2y_not_required_approval_marker_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | approval marker not required | extract_layout | Not required |
| h2y_h2u_no_controller_fallback | h2y_not_active_alert_banner_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | alert banner not active | extract_layout | alert |
| h2y_h2u_no_controller_fallback | h2y_not_escalated_risk_chip_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | risk chip not escalated | extract_layout | risk chip |
| h2y_h2u_no_controller_fallback | h2y_not_available_owner_field_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | owner field not available | extract_layout | not available |
| h2y_h2u_no_controller_fallback | h2y_not_started_phase_tile_value_before_component | h2y_genuine_negated_target_value | executable_paraphrase | True | extract_layout | phase tile not started | extract_layout | phase tile Not started |
| h2y_h2w_semantic_target_preservation | h2y_escalation_lane_stale_selection_not_lane | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | escalation lane | refine_selection |  |
| h2y_h2w_semantic_target_preservation | h2y_exception_panel_stale_selection_not_panel | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | exception panel | refine_selection |  |
| h2y_h2w_semantic_target_preservation | h2y_approval_field_stale_selection_not_field | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | approval field | refine_selection |  |
| h2y_h2w_semantic_target_preservation | h2y_not_active_alert_banner_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | alert banner not active | extract_layout | alert |
| h2y_h2w_no_controller_fallback | h2y_escalation_lane_stale_selection_not_lane | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | escalation lane | refine_selection |  |
| h2y_h2w_no_controller_fallback | h2y_exception_panel_stale_selection_not_panel | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | exception panel | refine_selection |  |
| h2y_h2w_no_controller_fallback | h2y_approval_field_stale_selection_not_field | h2y_stale_selection_negation_context | wrong_tool | False | extract_layout | approval field | refine_selection |  |
| h2y_h2w_no_controller_fallback | h2y_not_active_alert_banner_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | False | extract_layout | alert banner not active | extract_layout | alert |

## Controller Intervention Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | preserved_target_query | requested_label | requested_region_id | prompt_state_label | blocked_label | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2y_h2u_negation_guard | h2y_action_banner_quoted_not_banner_note | h2y_quoted_stale_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-action-banner-quote","target_query":"action banner"} | extract_layout | {"image_id":"img-h2y-action-banner-quote","target_query":"prior note"} |  |  |  | prior note |  |  |
| h2y_h2u_negation_guard | h2y_resolution_marker_old_quote_not_marker | h2y_quoted_stale_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-resolution-marker-quote","target_query":"resolution marker"} | extract_layout | {"image_id":"img-h2y-resolution-marker-quote","target_query":"audit memo"} |  |  |  | audit memo |  |  |
| h2y_h2u_negation_guard | h2y_vendor_chip_do_not_use_table | h2y_instructional_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-vendor-chip-table","target_query":"vendor chip Beta"} | extract_layout | {"image_id":"img-h2y-vendor-chip-table","target_query":"vendor chip"} |  |  |  | vendor chip |  |  |
| h2y_h2u_no_controller_fallback | h2y_action_banner_quoted_not_banner_note | h2y_quoted_stale_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-action-banner-quote","target_query":"action banner"} | extract_layout | {"image_id":"img-h2y-action-banner-quote","target_query":"prior note"} |  |  |  | prior note |  |  |
| h2y_h2u_no_controller_fallback | h2y_resolution_marker_old_quote_not_marker | h2y_quoted_stale_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-resolution-marker-quote","target_query":"resolution marker"} | extract_layout | {"image_id":"img-h2y-resolution-marker-quote","target_query":"audit memo"} |  |  |  | audit memo |  |  |
| h2y_h2u_no_controller_fallback | h2y_vendor_chip_do_not_use_table | h2y_instructional_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-vendor-chip-table","target_query":"vendor chip Beta"} | extract_layout | {"image_id":"img-h2y-vendor-chip-table","target_query":"vendor chip"} |  |  |  | vendor chip |  |  |
| h2y_h2w_semantic_target_preservation | h2y_action_banner_quoted_not_banner_note | h2y_quoted_stale_negation_context | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2y-action-banner-quote","target_query":"action banner"} | extract_layout | {"image_id":"img-h2y-action-banner-quote","target_query":"action banner"} | action banner |  |  | action banner | prior note | semantic_label_preserved_over_stale_context |
| h2y_h2w_semantic_target_preservation | h2y_resolution_marker_old_quote_not_marker | h2y_quoted_stale_negation_context | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2y-resolution-marker-quote","target_query":"resolution marker"} | extract_layout | {"image_id":"img-h2y-resolution-marker-quote","target_query":"resolution marker"} | resolution marker |  |  | resolution marker | audit memo | semantic_label_preserved_over_stale_context |
| h2y_h2w_semantic_target_preservation | h2y_vendor_chip_do_not_use_table | h2y_instructional_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-vendor-chip-table","target_query":"vendor chip Beta"} | extract_layout | {"image_id":"img-h2y-vendor-chip-table","target_query":"vendor chip"} |  |  |  | vendor chip |  |  |
| h2y_h2w_semantic_target_preservation | h2y_not_replied_status_pill_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-replied-status-pill","target_query":"not replied"} | extract_layout | {"image_id":"img-h2y-not-replied-status-pill","target_query":"status pill not replied"} |  |  |  | status pill not replied |  |  |
| h2y_h2w_semantic_target_preservation | h2y_not_sent_delivery_tag_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-sent-delivery-tag","target_query":"not sent"} | extract_layout | {"image_id":"img-h2y-not-sent-delivery-tag","target_query":"delivery tag not sent"} |  |  |  | delivery tag not sent |  |  |
| h2y_h2w_semantic_target_preservation | h2y_not_required_approval_marker_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-required-approval-marker","target_query":"Not required"} | extract_layout | {"image_id":"img-h2y-not-required-approval-marker","target_query":"approval marker not required"} |  |  |  | approval marker not required |  |  |
| h2y_h2w_semantic_target_preservation | h2y_not_escalated_risk_chip_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-escalated-risk-chip","target_query":"risk chip"} | extract_layout | {"image_id":"img-h2y-not-escalated-risk-chip","target_query":"risk chip not escalated"} |  |  |  | risk chip not escalated |  |  |
| h2y_h2w_semantic_target_preservation | h2y_not_available_owner_field_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-available-owner-field","target_query":"not available"} | extract_layout | {"image_id":"img-h2y-not-available-owner-field","target_query":"owner field not available"} |  |  |  | owner field not available |  |  |
| h2y_h2w_semantic_target_preservation | h2y_not_started_phase_tile_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-started-phase-tile","target_query":"phase tile Not started"} | extract_layout | {"image_id":"img-h2y-not-started-phase-tile","target_query":"phase tile not started"} |  |  |  | phase tile not started |  |  |
| h2y_h2w_no_controller_fallback | h2y_action_banner_quoted_not_banner_note | h2y_quoted_stale_negation_context | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2y-action-banner-quote","target_query":"action banner"} | extract_layout | {"image_id":"img-h2y-action-banner-quote","target_query":"action banner"} | action banner |  |  | action banner | prior note | semantic_label_preserved_over_stale_context |
| h2y_h2w_no_controller_fallback | h2y_resolution_marker_old_quote_not_marker | h2y_quoted_stale_negation_context | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2y-resolution-marker-quote","target_query":"resolution marker"} | extract_layout | {"image_id":"img-h2y-resolution-marker-quote","target_query":"resolution marker"} | resolution marker |  |  | resolution marker | audit memo | semantic_label_preserved_over_stale_context |
| h2y_h2w_no_controller_fallback | h2y_vendor_chip_do_not_use_table | h2y_instructional_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-vendor-chip-table","target_query":"vendor chip Beta"} | extract_layout | {"image_id":"img-h2y-vendor-chip-table","target_query":"vendor chip"} |  |  |  | vendor chip |  |  |
| h2y_h2w_no_controller_fallback | h2y_not_replied_status_pill_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-replied-status-pill","target_query":"not replied"} | extract_layout | {"image_id":"img-h2y-not-replied-status-pill","target_query":"status pill not replied"} |  |  |  | status pill not replied |  |  |
| h2y_h2w_no_controller_fallback | h2y_not_sent_delivery_tag_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-sent-delivery-tag","target_query":"not sent"} | extract_layout | {"image_id":"img-h2y-not-sent-delivery-tag","target_query":"delivery tag not sent"} |  |  |  | delivery tag not sent |  |  |
| h2y_h2w_no_controller_fallback | h2y_not_required_approval_marker_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-required-approval-marker","target_query":"Not required"} | extract_layout | {"image_id":"img-h2y-not-required-approval-marker","target_query":"approval marker not required"} |  |  |  | approval marker not required |  |  |
| h2y_h2w_no_controller_fallback | h2y_not_escalated_risk_chip_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-escalated-risk-chip","target_query":"risk chip"} | extract_layout | {"image_id":"img-h2y-not-escalated-risk-chip","target_query":"risk chip not escalated"} |  |  |  | risk chip not escalated |  |  |
| h2y_h2w_no_controller_fallback | h2y_not_available_owner_field_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-available-owner-field","target_query":"not available"} | extract_layout | {"image_id":"img-h2y-not-available-owner-field","target_query":"owner field not available"} |  |  |  | owner field not available |  |  |
| h2y_h2w_no_controller_fallback | h2y_not_started_phase_tile_value_before_component | h2y_genuine_negated_target_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h2y-not-started-phase-tile","target_query":"phase tile Not started"} | extract_layout | {"image_id":"img-h2y-not-started-phase-tile","target_query":"phase tile not started"} |  |  |  | phase tile not started |  |  |

## Fixed Case Rows

| comparison_label | case_id | family | baseline_failure_mode | candidate_failure_mode | baseline_executor_equivalence_match | candidate_executor_equivalence_match |
| --- | --- | --- | --- | --- | --- | --- |
| h2y_h2w_vs_h2u | h2y_action_banner_quoted_not_banner_note | h2y_quoted_stale_negation_context | argument_mismatch | exact | False | True |
| h2y_h2w_vs_h2u | h2y_not_available_owner_field_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | exact | False | True |
| h2y_h2w_vs_h2u | h2y_not_escalated_risk_chip_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | exact | False | True |
| h2y_h2w_vs_h2u | h2y_not_replied_status_pill_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | exact | False | True |
| h2y_h2w_vs_h2u | h2y_not_required_approval_marker_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | exact | False | True |
| h2y_h2w_vs_h2u | h2y_not_sent_delivery_tag_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | exact | False | True |
| h2y_h2w_vs_h2u | h2y_not_started_phase_tile_value_before_component | h2y_genuine_negated_target_value | executable_paraphrase | exact | True | True |
| h2y_h2w_vs_h2u | h2y_resolution_marker_old_quote_not_marker | h2y_quoted_stale_negation_context | argument_mismatch | exact | False | True |

## Unresolved H2w Boundary Rows

| profile_label | case_id | family | failure_mode | expected_calls | actual_calls |
| --- | --- | --- | --- | --- | --- |
| h2y_h2w_semantic_target_preservation | h2y_escalation_lane_stale_selection_not_lane | h2y_stale_selection_negation_context | wrong_tool | [{"arguments": {"image_id": "img-h2y-escalation-lane-stale-selection", "target_query": "escalation lane"}, "name": "extract_layout"}] | [{"arguments": {"filter_query": "not the escalation lane", "selection_id": "sel-h2y-old-note"}, "name": "refine_selection"}] |
| h2y_h2w_semantic_target_preservation | h2y_exception_panel_stale_selection_not_panel | h2y_stale_selection_negation_context | wrong_tool | [{"arguments": {"image_id": "img-h2y-exception-panel-stale-selection", "target_query": "exception panel"}, "name": "extract_layout"}] | [{"arguments": {"filter_query": "not the exception panel", "selection_id": "sel-h2y-exception-note"}, "name": "refine_selection"}] |
| h2y_h2w_semantic_target_preservation | h2y_approval_field_stale_selection_not_field | h2y_stale_selection_negation_context | wrong_tool | [{"arguments": {"image_id": "img-h2y-approval-field-stale-selection", "target_query": "approval field"}, "name": "extract_layout"}] | [{"arguments": {"filter_query": "approval field", "selection_id": "sel-h2y-approval-memo"}, "name": "refine_selection"}] |
| h2y_h2w_semantic_target_preservation | h2y_not_active_alert_banner_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | [{"arguments": {"image_id": "img-h2y-not-active-alert-banner", "target_query": "alert banner not active"}, "name": "extract_layout"}] | [{"arguments": {"image_id": "img-h2y-not-active-alert-banner", "target_query": "alert"}, "name": "extract_layout"}] |
| h2y_h2w_no_controller_fallback | h2y_escalation_lane_stale_selection_not_lane | h2y_stale_selection_negation_context | wrong_tool | [{"arguments": {"image_id": "img-h2y-escalation-lane-stale-selection", "target_query": "escalation lane"}, "name": "extract_layout"}] | [{"arguments": {"filter_query": "not the escalation lane", "selection_id": "sel-h2y-old-note"}, "name": "refine_selection"}] |
| h2y_h2w_no_controller_fallback | h2y_exception_panel_stale_selection_not_panel | h2y_stale_selection_negation_context | wrong_tool | [{"arguments": {"image_id": "img-h2y-exception-panel-stale-selection", "target_query": "exception panel"}, "name": "extract_layout"}] | [{"arguments": {"filter_query": "not the exception panel", "selection_id": "sel-h2y-exception-note"}, "name": "refine_selection"}] |
| h2y_h2w_no_controller_fallback | h2y_approval_field_stale_selection_not_field | h2y_stale_selection_negation_context | wrong_tool | [{"arguments": {"image_id": "img-h2y-approval-field-stale-selection", "target_query": "approval field"}, "name": "extract_layout"}] | [{"arguments": {"filter_query": "approval field", "selection_id": "sel-h2y-approval-memo"}, "name": "refine_selection"}] |
| h2y_h2w_no_controller_fallback | h2y_not_active_alert_banner_value_before_component | h2y_genuine_negated_target_value | argument_mismatch | [{"arguments": {"image_id": "img-h2y-not-active-alert-banner", "target_query": "alert banner not active"}, "name": "extract_layout"}] | [{"arguments": {"image_id": "img-h2y-not-active-alert-banner", "target_query": "alert"}, "name": "extract_layout"}] |

## Findings

| finding_id | finding |
| --- | --- |
| h2y_scales_h2x_pressure_and_breaks_h2w_saturation | H2y expands H2x to 16 cases; H2u reaches 4/16 strict and 5/16 executor-equivalent, while H2w reaches 12/16 on both metrics. |
| semantic_preservation_remains_causal_but_partial | H2w fixes 8 H2u strict misses, gaining 0.5 exact rate and 0.4375 executor-equivalence rate, but leaves 3 stale-selection negation rows plus one value-before-component row unresolved. |
| fallback_remains_non_causal_on_h2y | No-fallback controls tie their full rows: H2u fallback delta is 0.0 exact/0.0 executor-equivalent, and H2w fallback delta is 0.0 exact/0.0 executor-equivalent. |
| h2y_mechanism_mix_and_boundary | H2w records 2 semantic-preservation interventions and 7 target-query normalizations. Its exact family profile is h2y_genuine_negated_target_value 6/7, h2y_instructional_negation_context 3/3, h2y_quoted_stale_negation_context 3/3, h2y_stale_selection_negation_context 0/3. |
| next_helper_target_is_stale_selection_negation_and_short_component_value | The unresolved H2w rows show stale selection IDs passing through as refine_selection calls and the `not active alert banner` value collapsing to the short component query `alert`; the next helper ablation should target those two mechanisms, not fallback. |
