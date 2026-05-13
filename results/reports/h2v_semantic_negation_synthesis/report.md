# H2v Semantic Negation Synthesis

Generated: `2026-05-13T01:40:12.030472+00:00`

## Summary

H2v is the first fresh semantic-negation holdout after H2u closed the same-family transfer backtest. It separates quoted negation, instructional negation, stale example captions, genuine negated targets, and a clean control. The result breaks the apparent H2u saturation: H2u is better than H2r/H2j, but only by one case.

H2j and H2r each reach `3 / 10` strict and `4 / 10` executor-equivalent. H2u reaches `4 / 10` strict and `5 / 10` executor-equivalent, a `0.10` exact-rate improvement versus H2r.

The failure split matters more than the top-line: H2u solves the two instructional-negation rows and the clean control, fixes one quoted-negation row, but fails both stale-example rows and all three genuine-negated-target rows under strict exactness.

![H2v semantic negation gate](figures/h2v_semantic_negation_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2v_h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2j_execute_v1 | 10 | 3 | 0.3 | 4 | 0.4 |
| h2v_h2r_composed_route_gating | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating | results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2r_execute_v1 | 10 | 3 | 0.3 | 4 | 0.4 |
| h2v_h2u_negation_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard | results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2u_execute_v1 | 10 | 4 | 0.4 | 5 | 0.5 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2v_h2u_vs_h2r | results/tool_probe_replay_live_comparisons/20260513T_h2v_semantic_negation_h2u_vs_h2r_v1 | 10 | 0.3 | 0.4 | 0.10000000000000003 | 0.4 | 0.5 | 0.09999999999999998 |
| h2v_h2u_vs_h2j | results/tool_probe_replay_live_comparisons/20260513T_h2v_semantic_negation_h2u_vs_h2j_v1 | 10 | 0.3 | 0.4 | 0.10000000000000003 | 0.4 | 0.5 | 0.09999999999999998 |
| h2v_h2r_vs_h2j | results/tool_probe_replay_live_comparisons/20260513T_h2v_semantic_negation_h2r_vs_h2j_v1 | 10 | 0.3 | 0.3 | 0.0 | 0.4 | 0.4 | 0.0 |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| h2v_h2j_target_query_normalization | h2v_clean_negation_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2v_h2j_target_query_normalization | h2v_genuine_negated_target | 3 | 0 | 0.0 | 1 | 0.3333333333333333 |
| h2v_h2j_target_query_normalization | h2v_instructional_negation_context | 2 | 2 | 1.0 | 2 | 1.0 |
| h2v_h2j_target_query_normalization | h2v_quoted_negation_context | 2 | 0 | 0.0 | 0 | 0.0 |
| h2v_h2j_target_query_normalization | h2v_stale_example_negation_context | 2 | 0 | 0.0 | 0 | 0.0 |
| h2v_h2r_composed_route_gating | h2v_clean_negation_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2v_h2r_composed_route_gating | h2v_genuine_negated_target | 3 | 0 | 0.0 | 1 | 0.3333333333333333 |
| h2v_h2r_composed_route_gating | h2v_instructional_negation_context | 2 | 2 | 1.0 | 2 | 1.0 |
| h2v_h2r_composed_route_gating | h2v_quoted_negation_context | 2 | 0 | 0.0 | 0 | 0.0 |
| h2v_h2r_composed_route_gating | h2v_stale_example_negation_context | 2 | 0 | 0.0 | 0 | 0.0 |
| h2v_h2u_negation_guard | h2v_clean_negation_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2v_h2u_negation_guard | h2v_genuine_negated_target | 3 | 0 | 0.0 | 1 | 0.3333333333333333 |
| h2v_h2u_negation_guard | h2v_instructional_negation_context | 2 | 2 | 1.0 | 2 | 1.0 |
| h2v_h2u_negation_guard | h2v_quoted_negation_context | 2 | 1 | 0.5 | 1 | 0.5 |
| h2v_h2u_negation_guard | h2v_stale_example_negation_context | 2 | 0 | 0.0 | 0 | 0.0 |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2v_h2j_target_query_normalization | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | argument_mismatch | False | extract_layout | metric panel | extract_layout | audit note |
| h2v_h2j_target_query_normalization | h2v_summary_tile_quoted_not_label_caption | h2v_quoted_negation_context | no_tool_call | False | extract_layout | summary tile |  |  |
| h2v_h2j_target_query_normalization | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | argument_mismatch | False | extract_layout | review tile | extract_layout | stale caption |
| h2v_h2j_target_query_normalization | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | argument_mismatch | False | extract_layout | risk lane | extract_layout | example note |
| h2v_h2j_target_query_normalization | h2v_not_ready_badge_genuine_value | h2v_genuine_negated_target | argument_mismatch | False | extract_layout | status badge Not ready | extract_layout | Not ready |
| h2v_h2j_target_query_normalization | h2v_not_applicable_chip_genuine_value | h2v_genuine_negated_target | argument_mismatch | False | extract_layout | reason chip Not applicable | extract_layout | Not applicable |
| h2v_h2j_target_query_normalization | h2v_not_approved_toggle_genuine_value | h2v_genuine_negated_target | executable_paraphrase | True | extract_layout | approval toggle Not approved | extract_layout | Not approved toggle |
| h2v_h2r_composed_route_gating | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | argument_mismatch | False | extract_layout | metric panel | extract_layout | audit note |
| h2v_h2r_composed_route_gating | h2v_summary_tile_quoted_not_label_caption | h2v_quoted_negation_context | no_tool_call | False | extract_layout | summary tile |  |  |
| h2v_h2r_composed_route_gating | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | argument_mismatch | False | extract_layout | review tile | extract_layout | stale caption |
| h2v_h2r_composed_route_gating | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | argument_mismatch | False | extract_layout | risk lane | extract_layout | example note |
| h2v_h2r_composed_route_gating | h2v_not_ready_badge_genuine_value | h2v_genuine_negated_target | argument_mismatch | False | extract_layout | status badge Not ready | extract_layout | Not ready |
| h2v_h2r_composed_route_gating | h2v_not_applicable_chip_genuine_value | h2v_genuine_negated_target | argument_mismatch | False | extract_layout | reason chip Not applicable | extract_layout | Not applicable |
| h2v_h2r_composed_route_gating | h2v_not_approved_toggle_genuine_value | h2v_genuine_negated_target | executable_paraphrase | True | extract_layout | approval toggle Not approved | extract_layout | Not approved toggle |
| h2v_h2u_negation_guard | h2v_summary_tile_quoted_not_label_caption | h2v_quoted_negation_context | no_tool_call | False | extract_layout | summary tile |  |  |
| h2v_h2u_negation_guard | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | argument_mismatch | False | extract_layout | review tile | extract_layout | stale caption |
| h2v_h2u_negation_guard | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | argument_mismatch | False | extract_layout | risk lane | extract_layout | example note |
| h2v_h2u_negation_guard | h2v_not_ready_badge_genuine_value | h2v_genuine_negated_target | argument_mismatch | False | extract_layout | status badge Not ready | extract_layout | Not ready |
| h2v_h2u_negation_guard | h2v_not_applicable_chip_genuine_value | h2v_genuine_negated_target | argument_mismatch | False | extract_layout | reason chip Not applicable | extract_layout | Not applicable |
| h2v_h2u_negation_guard | h2v_not_approved_toggle_genuine_value | h2v_genuine_negated_target | executable_paraphrase | True | extract_layout | approval toggle Not approved | extract_layout | Not approved toggle |

## Controller Intervention Rows

| profile_label | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | preserved_target_query | blocked_label | prompt_state_label | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2v_h2j_target_query_normalization | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"metric panel"} | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"audit note"} |  | audit note | audit note |  |
| h2v_h2j_target_query_normalization | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2v-review-tile-stale-caption","target_query":"review tile"} | extract_layout | {"image_id":"img-h2v-review-tile-stale-caption","target_query":"stale caption"} |  | stale caption | stale caption |  |
| h2v_h2j_target_query_normalization | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | visual_stale_selection_gate | refine_selection | {"filter_query":"High","selection_id":"img-h2v-risk-lane-stale-example"} | extract_layout | {"image_id":"img-h2v-risk-lane-stale-example","target_query":"example note"} |  |  |  |  |
| h2v_h2r_composed_route_gating | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"metric panel"} | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"audit note"} |  | audit note | audit note |  |
| h2v_h2r_composed_route_gating | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2v-review-tile-stale-caption","target_query":"review tile"} | extract_layout | {"image_id":"img-h2v-review-tile-stale-caption","target_query":"stale caption"} |  | stale caption | stale caption |  |
| h2v_h2r_composed_route_gating | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | visual_stale_selection_gate | refine_selection | {"filter_query":"High","selection_id":"img-h2v-risk-lane-stale-example"} | extract_layout | {"image_id":"img-h2v-risk-lane-stale-example","target_query":"example note"} |  |  |  |  |
| h2v_h2u_negation_guard | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | visual_target_query_normalization_blocked | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"metric panel"} |  | {} | metric panel | audit note | audit note | negation_scope_exact_layout_label |
| h2v_h2u_negation_guard | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | visual_composed_route_gating_blocked | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"metric panel"} |  | {} | metric panel | audit note |  | negation_scope_exact_layout_label |
| h2v_h2u_negation_guard | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | visual_target_query_normalization | extract_layout | {"image_id":"img-h2v-review-tile-stale-caption","target_query":"review tile"} | extract_layout | {"image_id":"img-h2v-review-tile-stale-caption","target_query":"stale caption"} |  | stale caption | stale caption |  |
| h2v_h2u_negation_guard | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | visual_stale_selection_gate | refine_selection | {"filter_query":"High","selection_id":"img-h2v-risk-lane-stale-example"} | extract_layout | {"image_id":"img-h2v-risk-lane-stale-example","target_query":"example note"} |  |  |  |  |

## Fixed Case Rows

| comparison_label | case_id | family | baseline_failure_mode | candidate_failure_mode | baseline_executor_equivalence_match | candidate_executor_equivalence_match |
| --- | --- | --- | --- | --- | --- | --- |
| h2v_h2u_vs_h2r | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | argument_mismatch | exact | False | True |
| h2v_h2u_vs_h2j | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | argument_mismatch | exact | False | True |

## Findings

| finding_id | finding |
| --- | --- |
| h2v_breaks_h2u_transfer_saturation | H2v breaks the prior H2u same-family transfer saturation: H2u reaches 4/10 strict and 5/10 executor-equivalent after H2u had preserved 99/99 strict/executor-equivalent on the earlier transfer set. |
| h2u_negation_guard_help_is_real_but_small | H2u improves over H2r by 0.10 strict and 0.10 executor-equivalence rate, and over H2j by 0.10 strict and 0.10 executor-equivalence rate. |
| h2r_and_h2j_tie_on_h2v | H2r ties H2j on H2v at 3/10 strict and 4/10 executor-equivalent, with 0.00 exact-rate delta. Composed route gating alone does not solve this semantic negation split. |
| h2v_family_split_identifies_next_repair | H2u solves instructional negation at 2/2 and the clean control, but reaches only 1/2 on quoted negation, 0/2 on stale examples, and 0/3 strict plus 1/3 executor-equivalent on genuine negated targets. |
| h2u_fixed_case_is_one_quoted_context_row | The only strict H2u gain over both H2r and H2j is h2v_metric_panel_quoted_not_label_note; the remaining 6 H2u non-exact rows are not repaired by the current negation guard. |
| next_h2w_should_preserve_semantic_targets | The next candidate should distinguish negated context that must be ignored from negated values that are themselves the target, while also treating stale examples as old context even when the word not appears near a tempting label. |
