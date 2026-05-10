# H2d Transfer Tradeoff Synthesis

Generated: `2026-05-10T20:41:22.262681+00:00`

## Summary

H2d repairs the H2c transfer failure on H1x, but it is not a clean global replacement. The class-preserving route restores `8 / 8` exactness on H1x and fixes the `result chip` class-swap, while giving back one strict H2b exact row. The tradeoff is publishable because the lost H2b row remains executor-equivalent, whereas H2c's H1x miss broke executor-equivalence.

## Packet Rows

| packet_label | profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2b_residual_fit | h2a_stale_selection_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_h2a_execute_v1 | 5 | 0 | 0.0 | 3 | 0.6 |
| h2b_residual_fit | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_residual_guard_execute_v1 | 5 | 4 | 0.8 | 4 | 0.8 |
| h2b_residual_fit | h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 5 | 1.0 | 5 | 1.0 |
| h2b_residual_fit | h2d_class_preserving_route | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_class_preserving_residual_route_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h2b_execute_v1 | 5 | 4 | 0.8 | 5 | 1.0 |
| h1x_transfer | h2a_stale_selection_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x_transfer | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_residual_guard_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x_transfer | h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h1x_execute_v1 | 8 | 7 | 0.875 | 7 | 0.875 |
| h1x_transfer | h2d_class_preserving_route | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_class_preserving_residual_route_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2b_h2d_vs_h2c | results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_h2c_on_h2b_v1 | 5 | 1.0 | 0.8 | -0.19999999999999996 | 1.0 | 1.0 | 0.0 |
| h2b_h2d_vs_v12 | results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_component_residual_guard_on_h2b_v1 | 5 | 0.8 | 0.8 | 0.0 | 0.8 | 1.0 | 0.19999999999999996 |
| h2b_h2d_vs_h2a | results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_h2a_on_h2b_v1 | 5 | 0.0 | 0.8 | 0.8 | 0.6 | 1.0 | 0.4 |
| h1x_h2d_vs_h2c | results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_h2c_on_h1x_v1 | 8 | 0.875 | 1.0 | 0.125 | 0.875 | 1.0 | 0.125 |
| h1x_h2d_vs_h2a | results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_h2a_on_h1x_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1x_h2d_vs_v12 | results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_component_residual_guard_on_h1x_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1x_h2d_vs_v11 | results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_component_label_guard_on_h1x_v1 | 8 | 0.875 | 1.0 | 0.125 | 0.875 | 1.0 | 0.125 |

## H2d H2b Non-Exact Rows

| packet_label | profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_arguments | actual_tool | actual_arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2b_residual_fit | h2d_class_preserving_route | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | True | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "badge c08"} | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "escalated badge c08"} |

## H2c H1x Non-Exact Rows

| packet_label | profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_arguments | actual_tool | actual_arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1x_transfer | h2c_scoped_residual_gate | h1x_resolution_chip_comment_result_decoy | h1x_oblique_surface_value | argument_mismatch | False | extract_layout | {"image_id": "img-h1x-resolution-chip", "target_query": "result chip"} | extract_layout | {"image_id": "img-h1x-resolution-chip", "target_query": "result pill"} |

## Findings

| finding_id | finding |
| --- | --- |
| h2d_repairs_h2c_transfer_regression | H2d is 8/8 on H1x versus H2c at 7/8, a delta of 0.125 exact and executor-equivalence rate. |
| h2d_pays_local_h2b_exactness_cost | H2d is 4/5 on H2b versus H2c at 5/5, a delta of -0.19999999999999996 exact rate while preserving 5/5 executor-equivalence. |
| h2d_h2b_miss_is_executor_equivalent_over_specific_query | The H2d H2b exact miss is h1o_code_badge_c08_note_decoy: expected {"image_id": "img-h1o-code-badge-c08", "target_query": "badge c08"}, but produced {"image_id": "img-h1o-code-badge-c08", "target_query": "escalated badge c08"}; the executor still selected the same region. |
| h2c_h1x_miss_is_not_executor_equivalent | The H2c H1x miss is h1x_resolution_chip_comment_result_decoy: expected {"image_id": "img-h1x-resolution-chip", "target_query": "result chip"}, but produced {"image_id": "img-h1x-resolution-chip", "target_query": "result pill"}; this broke executor-equivalence. |
| next_slice | Build H2e as route arbitration, not a larger generic prompt: preserve H2c's H2b exactness for compact code/value residuals while using H2d's class-preserving rule for held-out component-class transfer. |
