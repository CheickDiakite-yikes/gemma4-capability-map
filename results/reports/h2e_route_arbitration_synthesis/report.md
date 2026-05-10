# H2e Route Arbitration Synthesis

Generated: `2026-05-10T20:49:48.512146+00:00`

## Summary

H2e reconciles the H2c/H2d tradeoff. H2c saturated H2b but lost held-out H1x transfer; H2d fixed transfer but gave back one local H2b exact row. H2e reaches `5 / 5` exact on H2b and `8 / 8` exact on H1x, with executor-equivalence saturated on both packets. This should not be promoted as a global default yet; it should seed a fresh H2f holdout gate.

![H2e route arbitration gate](figures/h2e_route_arbitration_gate.svg)

## Packet Rows

| packet_label | profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2b_residual_fit | h2a_stale_selection_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_h2a_execute_v1 | 5 | 0 | 0.0 | 3 | 0.6 |
| h2b_residual_fit | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_residual_guard_execute_v1 | 5 | 4 | 0.8 | 4 | 0.8 |
| h2b_residual_fit | h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 5 | 1.0 | 5 | 1.0 |
| h2b_residual_fit | h2d_class_preserving_route | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_class_preserving_residual_route_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h2b_execute_v1 | 5 | 4 | 0.8 | 5 | 1.0 |
| h2b_residual_fit | h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h2b_execute_v1 | 5 | 5 | 1.0 | 5 | 1.0 |
| h1x_transfer | h2a_stale_selection_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x_transfer | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_residual_guard_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x_transfer | h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h1x_execute_v1 | 8 | 7 | 0.875 | 7 | 0.875 |
| h1x_transfer | h2d_class_preserving_route | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_class_preserving_residual_route_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x_transfer | h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2b_h2e_vs_h2c | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_h2c_on_h2b_v1 | 5 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2b_h2e_vs_h2d | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_h2d_on_h2b_v1 | 5 | 0.8 | 1.0 | 0.19999999999999996 | 1.0 | 1.0 | 0.0 |
| h2b_h2e_vs_h2a | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_h2a_on_h2b_v1 | 5 | 0.0 | 1.0 | 1.0 | 0.6 | 1.0 | 0.4 |
| h2b_h2e_vs_v12 | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_component_residual_guard_on_h2b_v1 | 5 | 0.8 | 1.0 | 0.19999999999999996 | 0.8 | 1.0 | 0.19999999999999996 |
| h1x_h2e_vs_h2c | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_h2c_on_h1x_v1 | 8 | 0.875 | 1.0 | 0.125 | 0.875 | 1.0 | 0.125 |
| h1x_h2e_vs_h2d | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_h2d_on_h1x_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1x_h2e_vs_h2a | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_h2a_on_h1x_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1x_h2e_vs_v12 | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_component_residual_guard_on_h1x_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1x_h2e_vs_v11 | results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_component_label_guard_on_h1x_v1 | 8 | 0.875 | 1.0 | 0.125 | 0.875 | 1.0 | 0.125 |

## H2e Non-Exact Rows

_None._

## Counterfactual Miss Rows

| packet_label | profile_label | packet_dir | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_arguments | actual_tool | actual_arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| counterfactual | h2c_h1x_and_h2d_h2b | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h1x_execute_v1 | h1x_resolution_chip_comment_result_decoy | h1x_oblique_surface_value | argument_mismatch | False | extract_layout | {"image_id": "img-h1x-resolution-chip", "target_query": "result chip"} | extract_layout | {"image_id": "img-h1x-resolution-chip", "target_query": "result pill"} |
| counterfactual | h2c_h1x_and_h2d_h2b | results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h2b_execute_v1 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | True | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "badge c08"} | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "escalated badge c08"} |

## Findings

| finding_id | finding |
| --- | --- |
| h2e_saturates_both_h2b_and_h1x | H2e reaches 5/5 exact on H2b and 8/8 exact on H1x, with executor-equivalence also saturated. |
| h2e_reconciles_h2c_h2d_tradeoff | H2c is 5/5 then 7/8; H2d is 4/5 then 8/8; H2e preserves the max of both at 5/5 and 8/8. |
| h2e_transfer_gain_is_specific | H2e improves over H2c on H1x by 0.125 exact and executor-equivalence rate, and improves over H2d on H2b by 0.19999999999999996 strict exact rate. |
| counterfactual_misses_are_covered | The counterfactual miss table has 2 rows: H2c's result-chip class swap and H2d's badge-code over-specific query. H2e has zero non-exact rows across the two packets. |
| next_slice | Promote H2e only to a fresh H2f holdout gate. The current result is strong mechanism evidence, but the next proof must use newly authored route-arbitration cases rather than H2b/H1x rows. |
