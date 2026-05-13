# H2t Overreach Independence Synthesis

Generated: `2026-05-13T00:11:50.486112+00:00`

## Summary

H2t is the first post-H2s holdout designed to break top-line saturation by separating helpful target normalization from overreach. It keeps low-score/value exactness pressure, but adds negation-scope rows where a note or caption names a decoy component that should not become the target.

H2r reaches `8 / 10` strict and `8 / 10` executor-equivalent. H2p, H2o, and H2j tie that score, while H2e reaches `6 / 10` strict but `9 / 10` executor-equivalent. The H2e/H2r split is the important result: H2r preserves more literal exactness on low-score/value cases, but H2e avoids the negation-scope controller rewrite.

There are `2` H2r rows where raw MLX Gemma emitted the expected target and the controller rewrote it to a prompt-state label. H2u should therefore patch the controller, not the model prompt: normalization needs a negation-aware guard.

![H2t overreach independence gate](figures/h2t_overreach_independence_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2e_execute_v1 | 10 | 6 | 0.6 | 9 | 0.9 |
| h2t_h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2j_execute_v1 | 10 | 8 | 0.8 | 8 | 0.8 |
| h2t_h2o_value_bearing_synthesis | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis | results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2o_execute_v1 | 10 | 8 | 0.8 | 8 | 0.8 |
| h2t_h2p_contextual_surface_alias_routing | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing | results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2p_execute_v1 | 10 | 8 | 0.8 | 8 | 0.8 |
| h2t_h2r_composed_route_gating | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating | results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2r_execute_v1 | 10 | 8 | 0.8 | 8 | 0.8 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2r_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2e_v1 | 10 | 0.6 | 0.8 | 0.20000000000000007 | 0.9 | 0.8 | -0.09999999999999998 |
| h2t_h2r_vs_h2j | results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2j_v1 | 10 | 0.8 | 0.8 | 0.0 | 0.8 | 0.8 | 0.0 |
| h2t_h2r_vs_h2o | results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2o_v1 | 10 | 0.8 | 0.8 | 0.0 | 0.8 | 0.8 | 0.0 |
| h2t_h2r_vs_h2p | results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2p_v1 | 10 | 0.8 | 0.8 | 0.0 | 0.8 | 0.8 | 0.0 |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| h2t_h2e_route_arbitration | h2t_clean_route_control | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2e_route_arbitration | h2t_current_selection_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2e_route_arbitration | h2t_low_score_surface_request | 3 | 0 | 0.0 | 2 | 0.6666666666666666 |
| h2t_h2e_route_arbitration | h2t_negation_scope_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2e_route_arbitration | h2t_value_is_target_guard | 1 | 0 | 0.0 | 1 | 1.0 |
| h2t_h2j_target_query_normalization | h2t_clean_route_control | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2j_target_query_normalization | h2t_current_selection_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2j_target_query_normalization | h2t_low_score_surface_request | 3 | 3 | 1.0 | 3 | 1.0 |
| h2t_h2j_target_query_normalization | h2t_negation_scope_guard | 2 | 0 | 0.0 | 0 | 0.0 |
| h2t_h2j_target_query_normalization | h2t_value_is_target_guard | 1 | 1 | 1.0 | 1 | 1.0 |
| h2t_h2o_value_bearing_synthesis | h2t_clean_route_control | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2o_value_bearing_synthesis | h2t_current_selection_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2o_value_bearing_synthesis | h2t_low_score_surface_request | 3 | 3 | 1.0 | 3 | 1.0 |
| h2t_h2o_value_bearing_synthesis | h2t_negation_scope_guard | 2 | 0 | 0.0 | 0 | 0.0 |
| h2t_h2o_value_bearing_synthesis | h2t_value_is_target_guard | 1 | 1 | 1.0 | 1 | 1.0 |
| h2t_h2p_contextual_surface_alias_routing | h2t_clean_route_control | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2p_contextual_surface_alias_routing | h2t_current_selection_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2p_contextual_surface_alias_routing | h2t_low_score_surface_request | 3 | 3 | 1.0 | 3 | 1.0 |
| h2t_h2p_contextual_surface_alias_routing | h2t_negation_scope_guard | 2 | 0 | 0.0 | 0 | 0.0 |
| h2t_h2p_contextual_surface_alias_routing | h2t_value_is_target_guard | 1 | 1 | 1.0 | 1 | 1.0 |
| h2t_h2r_composed_route_gating | h2t_clean_route_control | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2r_composed_route_gating | h2t_current_selection_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2r_composed_route_gating | h2t_low_score_surface_request | 3 | 3 | 1.0 | 3 | 1.0 |
| h2t_h2r_composed_route_gating | h2t_negation_scope_guard | 2 | 0 | 0.0 | 0 | 0.0 |
| h2t_h2r_composed_route_gating | h2t_value_is_target_guard | 1 | 1 | 1.0 | 1 | 1.0 |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2e_route_arbitration | h2t_result_drawer_low_score_badge_decoy | h2t_low_score_surface_request | argument_mismatch | False | extract_layout | result drawer | extract_layout | result badge c08 |
| h2t_h2e_route_arbitration | h2t_risk_lane_high_chip_decoy | h2t_low_score_surface_request | executable_paraphrase | True | extract_layout | risk lane | extract_layout | risk lane High |
| h2t_h2e_route_arbitration | h2t_stage_column_review_tag_decoy | h2t_low_score_surface_request | executable_paraphrase | True | extract_layout | stage column | extract_layout | stage column Review |
| h2t_h2e_route_arbitration | h2t_escalated_value_not_badge_component | h2t_value_is_target_guard | executable_paraphrase | True | extract_layout | Escalated value cell | extract_layout | Escalated value |
| h2t_h2j_target_query_normalization | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | metric panel | extract_layout | training note |
| h2t_h2j_target_query_normalization | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | summary tile | extract_layout | caption |
| h2t_h2o_value_bearing_synthesis | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | metric panel | extract_layout | training note |
| h2t_h2o_value_bearing_synthesis | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | summary tile | extract_layout | caption |
| h2t_h2p_contextual_surface_alias_routing | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | metric panel | extract_layout | training note |
| h2t_h2p_contextual_surface_alias_routing | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | summary tile | extract_layout | caption |
| h2t_h2r_composed_route_gating | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | metric panel | extract_layout | training note |
| h2t_h2r_composed_route_gating | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | summary tile | extract_layout | caption |

## Bad Normalization Rows

| profile_label | case_id | family | expected_tool | expected_target_query | raw_target_query | actual_target_query | prompt_state_label | from_arguments | to_arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2j_target_query_normalization | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | extract_layout | metric panel | metric panel | training note | training note | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"training note"} |
| h2t_h2j_target_query_normalization | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | extract_layout | summary tile | summary tile | caption | caption | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"caption"} |
| h2t_h2o_value_bearing_synthesis | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | extract_layout | metric panel | metric panel | training note | training note | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"training note"} |
| h2t_h2o_value_bearing_synthesis | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | extract_layout | summary tile | summary tile | caption | caption | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"caption"} |
| h2t_h2p_contextual_surface_alias_routing | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | extract_layout | metric panel | metric panel | training note | training note | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"training note"} |
| h2t_h2p_contextual_surface_alias_routing | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | extract_layout | summary tile | summary tile | caption | caption | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"caption"} |
| h2t_h2r_composed_route_gating | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | extract_layout | metric panel | metric panel | training note | training note | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"training note"} |
| h2t_h2r_composed_route_gating | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | extract_layout | summary tile | summary tile | caption | caption | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"caption"} |

## Controller Intervention Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label | value_bearing_label | display_value | surface_label | requested_label | requested_region_id | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2j_target_query_normalization | h2t_result_drawer_low_score_badge_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result badge c08"} | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result drawer"} | result drawer |  |  |  |  |  |  |
| h2t_h2j_target_query_normalization | h2t_risk_lane_high_chip_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane High"} | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane"} | risk lane |  |  |  |  |  |  |
| h2t_h2j_target_query_normalization | h2t_stage_column_review_tag_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column Review"} | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column"} | stage column |  |  |  |  |  |  |
| h2t_h2j_target_query_normalization | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"training note"} | training note |  |  |  |  |  |  |
| h2t_h2j_target_query_normalization | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"caption"} | caption |  |  |  |  |  |  |
| h2t_h2j_target_query_normalization | h2t_escalated_value_not_badge_component | h2t_value_is_target_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value"} | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value cell"} | Escalated value cell |  |  |  |  |  |  |
| h2t_h2o_value_bearing_synthesis | h2t_result_drawer_low_score_badge_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result badge c08"} | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result drawer"} | result drawer |  |  |  |  |  |  |
| h2t_h2o_value_bearing_synthesis | h2t_risk_lane_high_chip_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane High"} | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane"} | risk lane |  |  |  |  |  |  |
| h2t_h2o_value_bearing_synthesis | h2t_stage_column_review_tag_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column Review"} | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column"} | stage column |  |  |  |  |  |  |
| h2t_h2o_value_bearing_synthesis | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"training note"} | training note |  |  |  |  |  |  |
| h2t_h2o_value_bearing_synthesis | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"caption"} | caption |  |  |  |  |  |  |
| h2t_h2o_value_bearing_synthesis | h2t_escalated_value_not_badge_component | h2t_value_is_target_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value"} | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value cell"} | Escalated value cell |  |  |  |  |  |  |
| h2t_h2p_contextual_surface_alias_routing | h2t_result_drawer_low_score_badge_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result badge c08"} | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result drawer"} | result drawer |  |  |  |  |  |  |
| h2t_h2p_contextual_surface_alias_routing | h2t_risk_lane_high_chip_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane High"} | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane"} | risk lane |  |  |  |  |  |  |
| h2t_h2p_contextual_surface_alias_routing | h2t_stage_column_review_tag_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column Review"} | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column"} | stage column |  |  |  |  |  |  |
| h2t_h2p_contextual_surface_alias_routing | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"training note"} | training note |  |  |  |  |  |  |
| h2t_h2p_contextual_surface_alias_routing | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"caption"} | caption |  |  |  |  |  |  |
| h2t_h2p_contextual_surface_alias_routing | h2t_escalated_value_not_badge_component | h2t_value_is_target_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value"} | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value cell"} | Escalated value cell |  |  |  |  |  |  |
| h2t_h2r_composed_route_gating | h2t_result_drawer_low_score_badge_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result badge c08"} | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result drawer"} | result drawer |  |  |  |  |  |  |
| h2t_h2r_composed_route_gating | h2t_risk_lane_high_chip_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane High"} | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane"} | risk lane |  |  |  |  |  |  |
| h2t_h2r_composed_route_gating | h2t_stage_column_review_tag_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column Review"} | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column"} | stage column |  |  |  |  |  |  |
| h2t_h2r_composed_route_gating | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"training note"} | training note |  |  |  |  |  |  |
| h2t_h2r_composed_route_gating | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"caption"} | caption |  |  |  |  |  |  |
| h2t_h2r_composed_route_gating | h2t_escalated_value_not_badge_component | h2t_value_is_target_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value"} | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value cell"} | Escalated value cell |  |  |  |  |  |  |

## Findings

| finding_id | finding |
| --- | --- |
| h2t_breaks_h2r_topline_saturation | H2r reaches 8/10 strict and 8/10 executor-equivalent on H2t; H2p, H2o, and H2j also reach 8/10, 8/10, and 8/10. |
| h2t_exposes_h2e_tradeoff | H2e reaches 6/10 strict and 9/10 executor-equivalent. H2r gains 0.20000000000000007 exact-rate versus H2e but loses -0.09999999999999998 executor-equivalence-rate. |
| h2t_later_helpers_do_not_add_signal | H2r ties H2p on H2t with delta 0.0 exact-rate; the overreach signal is shared by target-query normalization and the later synthesis/routing stack. |
| h2t_bad_normalization_is_controller_induced | H2r records 2 rows where the raw model emitted the expected target but controller normalization rewrote it to a prompt-state label. The H2r non-exact rows are 2 negation-scope cases. |
| h2t_h2e_preserves_negation_scope | H2e has 0 negation-scope misses while H2r has 2; this isolates the regression to the normalization helper rather than Gemma's raw local MLX call on those rows. |
| h2t_next_requires_h2u | H2t should promote an H2u intervention: target-query normalization must be negation-aware and must not rewrite an exact current-surface label to a note/caption label introduced only as context. H2r used 6 target normalizations on H2t. |
