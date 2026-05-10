# H2c Transfer Probe Synthesis

Generated: `2026-05-10T20:33:02.979828+00:00`

## Summary

H2c saturates H2b locally but fails the first held-out H1x transfer check. The failure is precise: it turns the expected `result chip` into `result pill`, indicating class-swap overfit from the H2b fit row. The next candidate should preserve the exact component class named in the prompt.

## Packet Rows

| packet_label | profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2b_residual_fit | h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 5 | 1.0 | 5 | 1.0 |
| h1x_transfer | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_label_guard_execute_v1 | 8 | 7 | 0.875 | 7 | 0.875 |
| h1x_transfer | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_residual_guard_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x_transfer | h2a_stale_selection_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x_transfer | h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h1x_execute_v1 | 8 | 7 | 0.875 | 7 | 0.875 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1x_h2c_vs_v11 | results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_component_label_guard_on_h1x_v1 | 8 | 0.875 | 0.875 | 0.0 | 0.875 | 0.875 | 0.0 |
| h1x_h2c_vs_v12 | results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_component_residual_guard_on_h1x_v1 | 8 | 1.0 | 0.875 | -0.125 | 1.0 | 0.875 | -0.125 |
| h1x_h2c_vs_h2a | results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_h2a_on_h1x_v1 | 8 | 1.0 | 0.875 | -0.125 | 1.0 | 0.875 | -0.125 |
| h1x_h2c_vs_no_directive | results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_no_directive_on_h1x_v1 | 8 | 0.25 | 0.875 | 0.625 | 0.25 | 0.875 | 0.625 |

## H2c H1x Non-Exact Rows

| packet_label | profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_arguments | actual_tool | actual_arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1x_transfer | h2c_scoped_residual_gate | h1x_resolution_chip_comment_result_decoy | h1x_oblique_surface_value | argument_mismatch | False | extract_layout | {"image_id": "img-h1x-resolution-chip", "target_query": "result chip"} | extract_layout | {"image_id": "img-h1x-resolution-chip", "target_query": "result pill"} |

## Findings

| finding_id | finding |
| --- | --- |
| h2c_local_fit_does_not_transfer_cleanly_to_h1x | H2c is 5/5 on H2b but only 7/8 on H1x, while H2a is 8/8 and v12 is 8/8. |
| h2c_regression_is_component_class_swap | The H1x miss is h1x_resolution_chip_comment_result_decoy: expected {"image_id": "img-h1x-resolution-chip", "target_query": "result chip"}, but H2c produced {"image_id": "img-h1x-resolution-chip", "target_query": "result pill"}. This is a class-swap overfit from `result pill` to `result chip`. |
| next_slice | Build H2d as a class-preserving scoped residual route: keep exact role-plus-component phrases from the prompt, but never substitute a component class learned from the H2b fit packet. |
