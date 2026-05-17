# H2w Transfer Backtest Synthesis

Generated: `2026-05-17T15:42:28.661931+00:00`

## Summary

H2w was introduced to repair H2v semantic target preservation: cases where the controller had to distinguish stale or quoted negation context from genuine negated target values. This backtest asks the next causal question: did that more permissive semantic preservation control regress older route, stale-selection, target-normalization, component-value, and negation-scope packets?

The answer on this battery is no. H2w reaches `109 / 109` strict and `109 / 109` executor-equivalent across H2s/H2t/H2q/H2m/H2k/H2l/H2f/H2b/H1x/H1y/H1o/H1p. Excluding H2q, the subtotal is `101 / 101`.

H2w ties H2u on every transfer/back-compat comparison: aggregate exact-rate delta `0.0` and executor-equivalence-rate delta `0.0`. Against H2r, the only positive delta is the inherited H2t negation-scope repair (`0.19999999999999996` exact-rate).

Operationally, this is also a runtime-posture result: one four-way parallel MLX attempt hit a Metal GPU timeout, while the sequential rerun completed cleanly. Treat local MLX transfer backtests as low-concurrency workloads unless the runtime is explicitly hardened for parallel replay.

![H2w transfer backtest gate](figures/h2w_transfer_backtest_gate.svg)

## Packet Pair Rows

| slice | case_count | h2r_exact_success_count | h2u_exact_success_count | h2w_exact_success_count | h2r_executor_success_count | h2u_executor_success_count | h2w_executor_success_count | h2w_delta_exact_vs_h2u | h2w_delta_executor_vs_h2u | h2w_delta_exact_vs_h2r | h2w_delta_executor_vs_h2r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2t | 10 | 8 | 10 | 10 | 8 | 10 | 10 | 0.0 | 0.0 | 0.19999999999999996 | 0.19999999999999996 |
| h2s | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 0.0 | 0.0 | 0.0 | 0.0 |
| h2q | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| h2m | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| h2k | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| h2l | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| h2f | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 0.0 | 0.0 | 0.0 | 0.0 |
| h2b | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 0.0 | 0.0 | 0.0 |
| h1x | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| h1y | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 0.0 | 0.0 | 0.0 | 0.0 |
| h1o | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 0.0 | 0.0 | 0.0 | 0.0 |
| h1p | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 0.0 | 0.0 | 0.0 | 0.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h2t_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2s_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h2s_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2q_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h2q_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2m_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h2m_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2k_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h2k_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2l_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h2l_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2f_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h2f_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2b_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h2b_v1 | 5 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1x_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h1x_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1y_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h1y_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1o_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h1o_v1 | 12 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1p_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2u_on_h1p_v1 | 12 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2t_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2t_v1 | 10 | 0.8 | 1.0 | 0.19999999999999996 | 0.8 | 1.0 | 0.19999999999999996 |
| h2s_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2s_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2q_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2q_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2m_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2m_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2k_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2k_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2l_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2l_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2f_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2f_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2b_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h2b_v1 | 5 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1x_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h1x_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1y_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h1y_v1 | 10 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1o_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h1o_v1 | 12 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1p_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260517T_h2w_semantic_target_preservation_vs_h2r_on_h1p_v1 | 12 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |

## Family Rows

| profile_label | slice | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2w_semantic_target_preservation | h2t | h2t_clean_route_control | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2w_semantic_target_preservation | h2t | h2t_current_selection_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2w_semantic_target_preservation | h2t | h2t_low_score_surface_request | 3 | 3 | 1.0 | 3 | 1.0 |
| h2t_h2w_semantic_target_preservation | h2t | h2t_negation_scope_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2t_h2w_semantic_target_preservation | h2t | h2t_value_is_target_guard | 1 | 1 | 1.0 | 1 | 1.0 |
| h2s_h2w_semantic_target_preservation | h2s | h2s_clean_route_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2s_h2w_semantic_target_preservation | h2s | h2s_contextual_alias_decoy_overlap | 2 | 2 | 1.0 | 2 | 1.0 |
| h2s_h2w_semantic_target_preservation | h2s | h2s_negated_decoy_guard | 1 | 1 | 1.0 | 1 | 1.0 |
| h2s_h2w_semantic_target_preservation | h2s | h2s_stale_surface_alias | 2 | 2 | 1.0 | 2 | 1.0 |
| h2s_h2w_semantic_target_preservation | h2s | h2s_surface_alias_same_value_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2s_h2w_semantic_target_preservation | h2s | h2s_value_bearing_stale_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2w_semantic_target_preservation | h2q | h2q_contextual_alias_decoy_overlap | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2w_semantic_target_preservation | h2q | h2q_stale_surface_alias | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2w_semantic_target_preservation | h2q | h2q_surface_alias_value_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2q_h2w_semantic_target_preservation | h2q | h2q_value_bearing_stale_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2m_h2w_semantic_target_preservation | h2m | h2m_contextual_alias_is_target | 2 | 2 | 1.0 | 2 | 1.0 |
| h2m_h2w_semantic_target_preservation | h2m | h2m_h2k_regression_guard_less_direct | 2 | 2 | 1.0 | 2 | 1.0 |
| h2m_h2w_semantic_target_preservation | h2m | h2m_less_direct_value_bearing_target | 4 | 4 | 1.0 | 4 | 1.0 |
| h2k_h2w_semantic_target_preservation | h2k | h2k_before_reading_decoy | 2 | 2 | 1.0 | 2 | 1.0 |
| h2k_h2w_semantic_target_preservation | h2k | h2k_code_label_overlap | 2 | 2 | 1.0 | 2 | 1.0 |
| h2k_h2w_semantic_target_preservation | h2k | h2k_negated_same_component_decoy | 3 | 3 | 1.0 | 3 | 1.0 |
| h2k_h2w_semantic_target_preservation | h2k | h2k_transfer_regression_guard | 1 | 1 | 1.0 | 1 | 1.0 |
| h2l_h2w_semantic_target_preservation | h2l | h2l_alias_is_target | 2 | 2 | 1.0 | 2 | 1.0 |
| h2l_h2w_semantic_target_preservation | h2l | h2l_h2k_regression_guard | 2 | 2 | 1.0 | 2 | 1.0 |
| h2l_h2w_semantic_target_preservation | h2l | h2l_value_bearing_target | 4 | 4 | 1.0 | 4 | 1.0 |
| h2f_h2w_semantic_target_preservation | h2f | h2f_activation_panel_notice | 2 | 2 | 1.0 | 2 | 1.0 |
| h2f_h2w_semantic_target_preservation | h2f | h2f_route_code_label | 2 | 2 | 1.0 | 2 | 1.0 |
| h2f_h2w_semantic_target_preservation | h2f | h2f_route_component_class_transfer | 2 | 2 | 1.0 | 2 | 1.0 |
| h2f_h2w_semantic_target_preservation | h2f | h2f_route_nonstandard_class | 2 | 2 | 1.0 | 2 | 1.0 |
| h2f_h2w_semantic_target_preservation | h2f | h2f_route_stale_field | 2 | 2 | 1.0 | 2 | 1.0 |
| h2b_h2w_semantic_target_preservation | h2b | h1o_code_negation_preservation | 2 | 2 | 1.0 | 2 | 1.0 |
| h2b_h2w_semantic_target_preservation | h2b | h1p_component_value_compact | 1 | 1 | 1.0 | 1 | 1.0 |
| h2b_h2w_semantic_target_preservation | h2b | h1p_component_value_surface | 1 | 1 | 1.0 | 1 | 1.0 |
| h2b_h2w_semantic_target_preservation | h2b | visual_argument_transfer_component_value_pill | 1 | 1 | 1.0 | 1 | 1.0 |
| h1x_h2w_semantic_target_preservation | h1x | h1x_oblique_activation_no_call | 2 | 2 | 1.0 | 2 | 1.0 |
| h1x_h2w_semantic_target_preservation | h1x | h1x_oblique_nonstandard_class | 2 | 2 | 1.0 | 2 | 1.0 |
| h1x_h2w_semantic_target_preservation | h1x | h1x_oblique_stale_field | 2 | 2 | 1.0 | 2 | 1.0 |
| h1x_h2w_semantic_target_preservation | h1x | h1x_oblique_surface_value | 2 | 2 | 1.0 | 2 | 1.0 |
| h1y_h2w_semantic_target_preservation | h1y | h1y_activation_no_call | 1 | 1 | 1.0 | 1 | 1.0 |
| h1y_h2w_semantic_target_preservation | h1y | h1y_preserve_surface_value | 2 | 2 | 1.0 | 2 | 1.0 |
| h1y_h2w_semantic_target_preservation | h1y | h1y_route_code_label | 2 | 2 | 1.0 | 2 | 1.0 |
| h1y_h2w_semantic_target_preservation | h1y | h1y_route_nonstandard_class | 2 | 2 | 1.0 | 2 | 1.0 |
| h1y_h2w_semantic_target_preservation | h1y | h1y_route_stale_field | 3 | 3 | 1.0 | 3 | 1.0 |
| h1o_h2w_semantic_target_preservation | h1o | h1o_activation_no_call | 4 | 4 | 1.0 | 4 | 1.0 |
| h1o_h2w_semantic_target_preservation | h1o | h1o_code_negation_preservation | 4 | 4 | 1.0 | 4 | 1.0 |
| h1o_h2w_semantic_target_preservation | h1o | h1o_component_value_boundary | 4 | 4 | 1.0 | 4 | 1.0 |
| h1p_h2w_semantic_target_preservation | h1p | h1p_component_value_compact | 4 | 4 | 1.0 | 4 | 1.0 |
| h1p_h2w_semantic_target_preservation | h1p | h1p_component_value_stale_selection | 4 | 4 | 1.0 | 4 | 1.0 |
| h1p_h2w_semantic_target_preservation | h1p | h1p_component_value_surface | 4 | 4 | 1.0 | 4 | 1.0 |

## Controller Intervention Rows

| profile_label | slice | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | preserved_target_query | blocked_label | prompt_state_label | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2w_semantic_target_preservation | h2t | h2t_result_drawer_low_score_badge_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result badge c08"} | extract_layout | {"image_id":"img-h2t-result-drawer-blocked","target_query":"result drawer"} |  | result drawer | result drawer |  |
| h2t_h2w_semantic_target_preservation | h2t | h2t_risk_lane_high_chip_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane High"} | extract_layout | {"image_id":"img-h2t-risk-lane-high","target_query":"risk lane"} |  | risk lane | risk lane |  |
| h2t_h2w_semantic_target_preservation | h2t | h2t_stage_column_review_tag_decoy | h2t_low_score_surface_request | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column Review"} | extract_layout | {"image_id":"img-h2t-stage-column-review","target_query":"stage column"} |  | stage column | stage column |  |
| h2t_h2w_semantic_target_preservation | h2t | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} | metric panel | training note | metric panel | semantic_label_preserved_over_stale_context |
| h2t_h2w_semantic_target_preservation | h2t | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | visual_composed_route_gating_blocked | extract_layout | {"image_id":"img-h2t-metric-panel-negation-scope","target_query":"metric panel"} |  | {} | metric panel | training note |  | negation_scope_exact_layout_label |
| h2t_h2w_semantic_target_preservation | h2t | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} | summary tile | caption | summary tile | semantic_label_preserved_over_stale_context |
| h2t_h2w_semantic_target_preservation | h2t | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | visual_composed_route_gating_blocked | extract_layout | {"image_id":"img-h2t-summary-tile-negation-scope","target_query":"summary tile"} |  | {} | summary tile | caption |  | negation_scope_exact_layout_label |
| h2t_h2w_semantic_target_preservation | h2t | h2t_escalated_value_not_badge_component | h2t_value_is_target_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value"} | extract_layout | {"image_id":"img-h2t-escalated-value","target_query":"Escalated value cell"} |  | Escalated value cell | Escalated value cell |  |
| h2s_h2w_semantic_target_preservation | h2s | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"Waiting review chip"} | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review note"} |  | review note | review note |  |
| h2s_h2w_semantic_target_preservation | h2s | h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review note"} | extract_layout | {"image_id":"img-h2s-review-tile-waiting","target_query":"review tile"} |  |  |  | requested_surface_over_deprioritized_decoy |
| h2s_h2w_semantic_target_preservation | h2s | h2s_signal_panel_green_tag_marker_decoys | h2s_surface_alias_same_value_decoy | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-signal-panel-green","target_query":"Green signal tag"} | extract_layout | {"image_id":"img-h2s-signal-panel-green","target_query":"signal panel"} |  |  |  | requested_surface_over_deprioritized_decoy |
| h2s_h2w_semantic_target_preservation | h2s | h2s_severity_pill_critical_archived_badge_decoy | h2s_value_bearing_stale_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"Critical severity pill"} | extract_layout | {"image_id":"img-h2s-severity-pill-critical","target_query":"severity pill Critical"} |  | severity pill Critical | severity pill Critical |  |
| h2s_h2w_semantic_target_preservation | h2s | h2s_reviewer_field_malik_old_owner_decoy | h2s_value_bearing_stale_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"Malik reviewer field"} | extract_layout | {"image_id":"img-h2s-reviewer-field-malik","target_query":"reviewer field Malik"} |  | reviewer field Malik | reviewer field Malik |  |
| h2s_h2w_semantic_target_preservation | h2s | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"archived runbook card"} | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout banner"} |  | timeout banner | timeout banner |  |
| h2s_h2w_semantic_target_preservation | h2s | h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout banner"} | extract_layout | {"image_id":"img-h2s-timeout-exception-notice","target_query":"timeout exception notice"} |  |  |  | requested_surface_over_deprioritized_decoy |
| h2s_h2w_semantic_target_preservation | h2s | h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"delivery field"} | extract_layout | {"image_id":"img-h2s-delivery-field-paused","target_query":"delivery field"} | delivery field | paused toggle | delivery field | semantic_label_preserved_over_stale_context |
| h2s_h2w_semantic_target_preservation | h2s | h2s_result_panel_waiting_stale_selection_hint | h2s_stale_surface_alias | visual_composed_route_gating | refine_selection | {"filter_query":"Waiting","selection_id":"sel-archive-result-chip"} | extract_layout | {"image_id":"img-h2s-result-panel-stale-selection","target_query":"result panel"} |  |  |  | stale_selection_to_requested_surface |
| h2s_h2w_semantic_target_preservation | h2s | h2s_status_tile_ready_stale_selection_hint | h2s_stale_surface_alias | visual_composed_route_gating | refine_selection | {"filter_query":"Ready","selection_id":"sel-old-status-tag"} | extract_layout | {"image_id":"img-h2s-status-tile-stale-selection","target_query":"status tile"} |  |  |  | stale_selection_to_requested_surface |
| h2s_h2w_semantic_target_preservation | h2s | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"panel Pending"} | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval note"} |  | approval note | approval note |  |
| h2s_h2w_semantic_target_preservation | h2s | h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | visual_composed_route_gating | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval note"} | extract_layout | {"image_id":"img-h2s-approval-panel-pending","target_query":"approval panel"} |  |  |  | requested_surface_over_deprioritized_decoy |
| h2q_h2w_semantic_target_preservation | h2q | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"Blocked badge"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} |  | result comment | result comment |  |
| h2q_h2w_semantic_target_preservation | h2q | h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | visual_composed_route_gating | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result comment"} | extract_layout | {"image_id":"img-h2q-result-tile-blocked","target_query":"result tile"} |  |  |  | requested_surface_over_deprioritized_decoy |
| h2q_h2w_semantic_target_preservation | h2q | h2q_state_panel_closed_value_tag_decoy | h2q_surface_alias_value_decoy | visual_contextual_surface_alias_routing | extract_layout | {"image_id":"img-h2q-state-panel-closed","target_query":"Closed"} | extract_layout | {"image_id":"img-h2q-state-panel-closed","target_query":"state panel"} |  |  |  | contextual_surface_alias_recoverable |
| h2q_h2w_semantic_target_preservation | h2q | h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"Critical priority badge"} | extract_layout | {"image_id":"img-h2q-priority-badge-critical","target_query":"priority badge Critical"} |  | priority badge Critical | priority badge Critical |  |
| h2q_h2w_semantic_target_preservation | h2q | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field Amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field Amina"} | owner field Amina | owner field | owner field Amina | semantic_label_preserved_over_stale_context |
| h2q_h2w_semantic_target_preservation | h2q | h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"amina"} | extract_layout | {"image_id":"img-h2q-owner-field-amina","target_query":"owner field Amina"} |  |  |  |  |
| h2q_h2w_semantic_target_preservation | h2q | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_target_query_normalization | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"archived exception panel"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} |  | error banner | error banner |  |
| h2q_h2w_semantic_target_preservation | h2q | h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | visual_composed_route_gating | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error banner"} | extract_layout | {"image_id":"img-h2q-error-notice","target_query":"error notice"} |  |  |  | requested_surface_over_deprioritized_decoy |
| h2q_h2w_semantic_target_preservation | h2q | h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | extract_layout | {"image_id":"img-h2q-mode-field","target_query":"mode field"} | mode field | mode switch | mode field | semantic_label_preserved_over_stale_context |
| h2q_h2w_semantic_target_preservation | h2q | h2q_result_tile_stale_selection_hint | h2q_stale_surface_alias | visual_composed_route_gating | refine_selection | {"filter_query":"Blocked","selection_id":"sel-archived-result-badge"} | extract_layout | {"image_id":"img-h2q-result-tile-stale-selection","target_query":"result tile"} |  |  |  | stale_selection_to_requested_surface |
| h2q_h2w_semantic_target_preservation | h2q | h2q_state_panel_stale_selection_hint | h2q_stale_surface_alias | visual_composed_route_gating | refine_selection | {"filter_query":"Closed","selection_id":"sel-archived-state-tag"} | extract_layout | {"image_id":"img-h2q-state-panel-stale-selection","target_query":"state panel"} |  |  |  | stale_selection_to_requested_surface |
| h2m_h2w_semantic_target_preservation | h2m | h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | visual_value_bearing_target_query_synthesis | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result chip"} | extract_layout | {"image_id":"img-h2m-result-badge-blocked","target_query":"result badge Blocked"} |  | result badge | result badge | value_bearing_label_recoverable |
| h2m_h2w_semantic_target_preservation | h2m | h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"Closed state tag"} | extract_layout | {"image_id":"img-h2m-state-tag-closed","target_query":"state tag Closed"} |  | state tag Closed | state tag Closed |  |
| h2m_h2w_semantic_target_preservation | h2m | h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-mode-toggle-manual","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2m-mode-toggle-manual","target_query":"mode toggle Manual"} |  | mode toggle Manual | mode toggle Manual |  |
| h2m_h2w_semantic_target_preservation | h2m | h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge critical"} | extract_layout | {"image_id":"img-h2m-priority-badge-critical","target_query":"priority badge Critical"} |  | priority badge Critical | priority badge Critical |  |
| h2m_h2w_semantic_target_preservation | h2m | h2m_error_notice_contextual_alias | h2m_contextual_alias_is_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"archive panel"} | extract_layout | {"image_id":"img-h2m-error-notice","target_query":"error notice"} |  | error notice | error notice |  |
| h2m_h2w_semantic_target_preservation | h2m | h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | visual_contextual_surface_alias_routing | extract_layout | {"image_id":"img-h2m-result-tile","target_query":"Blocked"} | extract_layout | {"image_id":"img-h2m-result-tile","target_query":"result tile"} |  |  |  | contextual_surface_alias_recoverable |
| h2m_h2w_semantic_target_preservation | h2m | h2m_mode_field_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | visual_target_query_normalization | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode switch"} | extract_layout | {"image_id":"img-h2m-mode-field-short","target_query":"mode field"} |  | mode field | mode field |  |
| h2k_h2w_semantic_target_preservation | h2k | h2k_mode_toggle_negated_consent_toggle_decoy | h2k_negated_same_component_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-mode-toggle","target_query":"mode toggle Manual"} | extract_layout | {"image_id":"img-h2k-mode-toggle","target_query":"mode toggle"} |  | mode toggle | mode toggle |  |
| h2k_h2w_semantic_target_preservation | h2k | h2k_result_badge_negated_result_tile_decoy | h2k_negated_same_component_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-result-badge","target_query":"result badge Blocked"} | extract_layout | {"image_id":"img-h2k-result-badge","target_query":"result badge"} |  | result badge | result badge |  |
| h2k_h2w_semantic_target_preservation | h2k | h2k_error_banner_archived_error_notice_decoy | h2k_transfer_regression_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-error-banner","target_query":"error notice"} | extract_layout | {"image_id":"img-h2k-error-banner","target_query":"error banner"} |  | error banner | error banner |  |
| h2k_h2w_semantic_target_preservation | h2k | h2k_state_tag_before_reading_state_marker_decoy | h2k_before_reading_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-state-tag","target_query":"state tag Closed"} | extract_layout | {"image_id":"img-h2k-state-tag","target_query":"state tag"} |  | state tag | state tag |  |
| h2k_h2w_semantic_target_preservation | h2k | h2k_mode_field_before_reading_mode_switch_decoy | h2k_before_reading_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-mode-field","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2k-mode-field","target_query":"mode field"} |  | mode field | mode field |  |
| h2k_h2w_semantic_target_preservation | h2k | h2k_alert_t47_archived_alert_s92_decoy | h2k_code_label_overlap | visual_composed_route_gating_blocked | extract_layout | {"image_id":"img-h2k-alert-t47","target_query":"alert t47"} |  | {} | alert t47 | alert s92 |  | negation_scope_exact_layout_label |
| h2l_h2w_semantic_target_preservation | h2l | h2l_status_badge_short_label_regression_guard | h2l_h2k_regression_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2l-status-badge-short","target_query":"critical chip"} | extract_layout | {"image_id":"img-h2l-status-badge-short","target_query":"status badge"} |  | status badge | status badge |  |
| h2f_h2w_semantic_target_preservation | h2f | h2f_result_tile_comment_value_decoy | h2f_route_component_class_transfer | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-result-tile","target_query":"Blocked"} | extract_layout | {"image_id":"img-h2f-result-tile","target_query":"result tile"} |  | result tile | result tile |  |
| h2f_h2w_semantic_target_preservation | h2f | h2f_resolution_badge_log_result_decoy | h2f_route_component_class_transfer | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-resolution-badge","target_query":"Deferred"} | extract_layout | {"image_id":"img-h2f-resolution-badge","target_query":"resolution badge"} |  | resolution badge | resolution badge |  |
| h2f_h2w_semantic_target_preservation | h2f | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-state-marker","target_query":"lifecycle state marker"} | extract_layout | {"image_id":"img-h2f-state-marker","target_query":"state marker"} |  | state marker | state marker |  |
| h2f_h2w_semantic_target_preservation | h2f | h2f_mode_switch_note_value_decoy | h2f_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-mode-switch","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2f-mode-switch","target_query":"mode switch"} |  | mode switch | mode switch |  |
| h2f_h2w_semantic_target_preservation | h2f | h2f_owner_field_previous_memo_decoy | h2f_route_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"owner field","selection_id":"sel-h2f-owner-memo"} | extract_layout | {"image_id":"img-h2f-owner-field","target_query":"owner field"} |  |  |  |  |
| h2b_h2w_semantic_target_preservation | h2b | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | visual_stale_selection_gate | refine_selection | {"filter_query":"approved","selection_id":null} | extract_layout | {"image_id":"img-component-result-pill","target_query":"result pill"} |  |  |  |  |
| h1x_h2w_semantic_target_preservation | h1x | h1x_responsible_party_field_old_owner_memo_decoy | h1x_oblique_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"responsible-party entry showing Iris","selection_id":"sel-owner-memo"} | extract_layout | {"image_id":"img-h1x-owner-field","target_query":"owner field"} |  |  |  |  |
| h1x_h2w_semantic_target_preservation | h1x | h1x_workstream_owner_field_previous_summary_decoy | h1x_oblique_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"owner","selection_id":"sel-workstream-summary"} | extract_layout | {"image_id":"img-h1x-workstream-owner","target_query":"owner field"} |  |  |  |  |
| h1y_h2w_semantic_target_preservation | h1y | h1y_responsible_party_field_old_owner_memo_decoy | h1y_route_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"sel-h1y-owner-memo"} | extract_layout | {"image_id":"img-h1y-owner-field","target_query":"owner field"} |  |  |  |  |
| h1y_h2w_semantic_target_preservation | h1y | h1y_escalation_contact_field_saved_summary_decoy | h1y_route_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"latest","selection_id":"sel-h1y-contact-summary"} | extract_layout | {"image_id":"img-h1y-escalation-contact","target_query":"owner field"} |  |  |  |  |
| h1y_h2w_semantic_target_preservation | h1y | h1y_lifecycle_state_tag_audit_value_decoy | h1y_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h1y-state-tag","target_query":"lifecycle state tag"} | extract_layout | {"image_id":"img-h1y-state-tag","target_query":"state tag"} |  | state tag | state tag |  |
| h1y_h2w_semantic_target_preservation | h1y | h1y_alert_s92_negated_toggle_decoy | h1y_route_code_label | visual_composed_route_gating_blocked | extract_layout | {"image_id":"img-h1y-alert-s92","target_query":"alert s92"} |  | {} | alert s92 | override toggle |  | negation_scope_exact_layout_label |
| h1y_h2w_semantic_target_preservation | h1y | h1y_status_pill_summary_value_holdout | h1y_preserve_surface_value | visual_target_query_normalization | extract_layout | {"image_id":"img-h1y-status-pill","target_query":"Pending"} | extract_layout | {"image_id":"img-h1y-status-pill","target_query":"status pill"} |  | status pill | status pill |  |
| h1o_h2w_semantic_target_preservation | h1o | h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | visual_target_query_normalization | extract_layout | {"image_id":"img-h1o-activation-warning-tile","target_query":"overdue"} | extract_layout | {"image_id":"img-h1o-activation-warning-tile","target_query":"warning tile"} |  | warning tile | warning tile |  |

## Fixed Case Rows

| comparison_label | case_id | family | delta_exact_match | baseline_failure_mode | candidate_failure_mode | baseline_executor_equivalence_match | candidate_executor_equivalence_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2w_vs_h2r | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | 1 | argument_mismatch | exact | False | True |
| h2t_h2w_vs_h2r | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | 1 | argument_mismatch | exact | False | True |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2t_h2r_composed_route_gating | h2t_metric_panel_negation_scope_note | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | metric panel | extract_layout | training note |
| h2t_h2r_composed_route_gating | h2t_summary_tile_negation_scope_caption | h2t_negation_scope_guard | argument_mismatch | False | extract_layout | summary tile | extract_layout | caption |

## Findings

| finding_id | finding |
| --- | --- |
| h2w_transfer_backtest_is_clean | H2w preserves 109/109 strict exactness and 109/109 executor equivalence across the 12-packet transfer/backward compatibility battery. |
| h2w_ties_current_h2u_incumbent | Against H2u, H2w has zero exact-rate and executor-equivalence-rate delta on every transfer packet (aggregate delta exact 0.0); there are 0 strict regressions. |
| h2w_keeps_h2t_repair_vs_h2r | Against H2r, H2w only changes the H2t slice: delta exact 0.19999999999999996 and executor-equivalence 0.19999999999999996. The fixed rows are h2t_metric_panel_negation_scope_note, h2t_summary_tile_negation_scope_caption. |
| h2w_controller_activity_does_not_imply_transfer_cost | The transfer runs record controller activity (5 semantic-preservation, 30 target-normalization, 7 stale-selection, and 4 blocked composed-route rows) while still leaving 0 non-exact H2w rows. |
| h2w_runtime_posture_needs_low_concurrency | The evidence also separates model/control quality from local runtime posture: a four-way parallel MLX replay attempt hit a Metal GPU timeout, while the sequential rerun completed cleanly. Future local MLX backtests should default to sequential or very-low-concurrency execution. |
