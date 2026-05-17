# H2w Semantic Target Preservation Synthesis

Generated: `2026-05-17T15:42:28.037213+00:00`

## Summary

H2w is the direct repair candidate for H2v. It adds semantic target preservation on top of H2u, separating stale/quoted negation context from genuine negated target values and adding a bounded no-call visual fallback when the requested visual target is unambiguous.

H2w reaches `10 / 10` strict and `10 / 10` executor-equivalent, versus H2u's `4 / 10` strict. The exact-rate gain over H2u is `0.6`.

Mechanistically, H2w does not simply suppress the word `not`. It preserves current requested labels when negation belongs to stale context, canonicalizes value-before-surface phrases such as `Not ready status badge` to layout labels such as `status badge Not ready`. The control also has a bounded no-call visual fallback, but the final H2v H2w packet did not need to exercise it.

The separate H2w transfer backtest now preserves `109 / 109` strict and executor-equivalent rows across the current transfer/back-compat battery. This H2v-local report should therefore be read together with `../h2w_transfer_backtest_synthesis/report.md`: transfer is clean, while packaged workflow semantic pressure remains unproven.

![H2w semantic target preservation gate](figures/h2w_semantic_target_preservation_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2v_h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2j_execute_v1 | 10 | 3 | 0.3 | 4 | 0.4 |
| h2v_h2r_composed_route_gating | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating | results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2r_execute_v1 | 10 | 3 | 0.3 | 4 | 0.4 |
| h2v_h2u_negation_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard | results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2u_execute_v1 | 10 | 4 | 0.4 | 5 | 0.5 |
| h2v_h2w_semantic_target_preservation | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard_visual_semantic_target_preservation | results/tool_probe_replay_live/20260513T_h2v_semantic_negation_h2w_execute_v1 | 10 | 10 | 1.0 | 10 | 1.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2v_h2w_vs_h2u | results/tool_probe_replay_live_comparisons/20260513T_h2v_semantic_negation_h2w_vs_h2u_v1 | 10 | 0.4 | 1.0 | 0.6 | 0.5 | 1.0 | 0.5 |
| h2v_h2w_vs_h2r | results/tool_probe_replay_live_comparisons/20260513T_h2v_semantic_negation_h2w_vs_h2r_v1 | 10 | 0.3 | 1.0 | 0.7 | 0.4 | 1.0 | 0.6 |
| h2v_h2w_vs_h2j | results/tool_probe_replay_live_comparisons/20260513T_h2v_semantic_negation_h2w_vs_h2j_v1 | 10 | 0.3 | 1.0 | 0.7 | 0.4 | 1.0 | 0.6 |

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
| h2v_h2w_semantic_target_preservation | h2v_clean_negation_control | 1 | 1 | 1.0 | 1 | 1.0 |
| h2v_h2w_semantic_target_preservation | h2v_genuine_negated_target | 3 | 3 | 1.0 | 3 | 1.0 |
| h2v_h2w_semantic_target_preservation | h2v_instructional_negation_context | 2 | 2 | 1.0 | 2 | 1.0 |
| h2v_h2w_semantic_target_preservation | h2v_quoted_negation_context | 2 | 2 | 1.0 | 2 | 1.0 |
| h2v_h2w_semantic_target_preservation | h2v_stale_example_negation_context | 2 | 2 | 1.0 | 2 | 1.0 |

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
| h2v_h2w_semantic_target_preservation | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"metric panel"} | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"metric panel"} | metric panel | audit note | metric panel | semantic_label_preserved_over_stale_context |
| h2v_h2w_semantic_target_preservation | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | visual_composed_route_gating_blocked | extract_layout | {"image_id":"img-h2v-metric-panel-quoted-note","target_query":"metric panel"} |  | {} | metric panel | audit note |  | negation_scope_exact_layout_label |
| h2v_h2w_semantic_target_preservation | h2v_summary_tile_quoted_not_label_caption | h2v_quoted_negation_context | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2v-summary-tile-quoted-caption","target_query":"summary tile"} | extract_layout | {"image_id":"img-h2v-summary-tile-quoted-caption","target_query":"summary tile"} | summary tile | caption | summary tile | semantic_label_preserved_over_stale_context |
| h2v_h2w_semantic_target_preservation | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2v-review-tile-stale-caption","target_query":"review tile"} | extract_layout | {"image_id":"img-h2v-review-tile-stale-caption","target_query":"review tile"} | review tile | stale caption | review tile | semantic_label_preserved_over_stale_context |
| h2v_h2w_semantic_target_preservation | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | visual_semantic_target_preservation | extract_layout | {"image_id":"img-h2v-risk-lane-stale-example","target_query":"risk lane"} | extract_layout | {"image_id":"img-h2v-risk-lane-stale-example","target_query":"risk lane"} | risk lane | example note | risk lane | semantic_label_preserved_over_stale_context |
| h2v_h2w_semantic_target_preservation | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | visual_stale_selection_gate | refine_selection | {"filter_query":"High","selection_id":"img-h2v-risk-lane-stale-example"} | extract_layout | {"image_id":"img-h2v-risk-lane-stale-example","target_query":"risk lane"} |  |  |  |  |
| h2v_h2w_semantic_target_preservation | h2v_not_ready_badge_genuine_value | h2v_genuine_negated_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2v-not-ready-badge","target_query":"Not ready"} | extract_layout | {"image_id":"img-h2v-not-ready-badge","target_query":"status badge Not ready"} |  | status badge Not ready | status badge Not ready |  |
| h2v_h2w_semantic_target_preservation | h2v_not_applicable_chip_genuine_value | h2v_genuine_negated_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2v-not-applicable-chip","target_query":"Not applicable"} | extract_layout | {"image_id":"img-h2v-not-applicable-chip","target_query":"reason chip Not applicable"} |  | reason chip Not applicable | reason chip Not applicable |  |
| h2v_h2w_semantic_target_preservation | h2v_not_approved_toggle_genuine_value | h2v_genuine_negated_target | visual_target_query_normalization | extract_layout | {"image_id":"img-h2v-not-approved-toggle","target_query":"Not approved toggle"} | extract_layout | {"image_id":"img-h2v-not-approved-toggle","target_query":"approval toggle Not approved"} |  | approval toggle Not approved | approval toggle Not approved |  |

## Fixed Case Rows

| comparison_label | case_id | family | baseline_failure_mode | candidate_failure_mode | baseline_executor_equivalence_match | candidate_executor_equivalence_match |
| --- | --- | --- | --- | --- | --- | --- |
| h2v_h2w_vs_h2u | h2v_not_applicable_chip_genuine_value | h2v_genuine_negated_target | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2u | h2v_not_approved_toggle_genuine_value | h2v_genuine_negated_target | executable_paraphrase | exact | True | True |
| h2v_h2w_vs_h2u | h2v_not_ready_badge_genuine_value | h2v_genuine_negated_target | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2u | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2u | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2u | h2v_summary_tile_quoted_not_label_caption | h2v_quoted_negation_context | no_tool_call | exact | False | True |
| h2v_h2w_vs_h2r | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2r | h2v_not_applicable_chip_genuine_value | h2v_genuine_negated_target | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2r | h2v_not_approved_toggle_genuine_value | h2v_genuine_negated_target | executable_paraphrase | exact | True | True |
| h2v_h2w_vs_h2r | h2v_not_ready_badge_genuine_value | h2v_genuine_negated_target | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2r | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2r | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2r | h2v_summary_tile_quoted_not_label_caption | h2v_quoted_negation_context | no_tool_call | exact | False | True |
| h2v_h2w_vs_h2j | h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2j | h2v_not_applicable_chip_genuine_value | h2v_genuine_negated_target | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2j | h2v_not_approved_toggle_genuine_value | h2v_genuine_negated_target | executable_paraphrase | exact | True | True |
| h2v_h2w_vs_h2j | h2v_not_ready_badge_genuine_value | h2v_genuine_negated_target | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2j | h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2j | h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | argument_mismatch | exact | False | True |
| h2v_h2w_vs_h2j | h2v_summary_tile_quoted_not_label_caption | h2v_quoted_negation_context | no_tool_call | exact | False | True |

## Findings

| finding_id | finding |
| --- | --- |
| h2w_repairs_h2v_strict_and_executor | H2w repairs H2v from H2u's 4/10 strict and 5/10 executor-equivalent to 10/10 strict and executor-equivalent. |
| h2w_gain_is_causal_on_six_h2u_misses | H2w fixes 6 strict H2u misses with a 0.6 exact-rate gain and 0.5 executor-equivalence gain. |
| h2w_mechanism_splits_three_error_types | The H2w run records 4 semantic-preservation interventions, 3 component-qualified value canonicalizations, 1 stale-selection repair, and 1 negation-aware composed-route block. |
| h2w_family_saturation_is_local_not_global | H2w reaches exactness across all H2v families (h2v_clean_negation_control 1/1, h2v_genuine_negated_target 3/3, h2v_instructional_negation_context 2/2, h2v_quoted_negation_context 2/2, h2v_stale_example_negation_context 2/2). The separate H2w transfer backtest is now clean, so the remaining promotion gap is packaged-workflow or harder CLI-live semantic pressure rather than same-family replay transfer. |
| h2w_next_requires_packaged_semantic_pressure | The next step is packaged-workflow or harder CLI-live pressure, not another replay transfer pass: H2w includes a bounded no-call visual fallback and a more permissive semantic label selector that should be tested where workflow scaffolding cannot resolve the ambiguity upstream. |
