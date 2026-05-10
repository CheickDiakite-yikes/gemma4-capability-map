# H1r Component Residual Synthesis

Generated: `2026-05-10T17:43:28.308703+00:00`

## Summary

H1r isolates the H1q residual families: stale-selection fields, nonstandard component classes such as tag/toggle, and code-label exactness. No-directive collapses to `0 / 6` exact and `1 / 6` executor-equivalent; v11 is strong at `5 / 6`; v12 saturates the packet at `6 / 6`.

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | no_tool_call_count | argument_mismatch_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_no_directive_execute_v1 | 6 | 0 | 0.0 | 1 | 0.16666666666666666 | 3 | 2 | 1 |
| component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_label_guard_execute_v1 | 6 | 5 | 0.8333333333333334 | 5 | 0.8333333333333334 | 0 | 1 | 0 |
| component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_residual_guard_execute_v1 | 6 | 6 | 1.0 | 6 | 1.0 | 0 | 0 | 0 |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| no_directive | h1r_code_label_exactness | 2 | 0 | 0.0 | 1 | 0.5 |
| no_directive | h1r_nonstandard_component_class | 2 | 0 | 0.0 | 0 | 0.0 |
| no_directive | h1r_stale_selection_component_label | 2 | 0 | 0.0 | 0 | 0.0 |
| component_label_guard_v11 | h1r_code_label_exactness | 2 | 1 | 0.5 | 1 | 0.5 |
| component_label_guard_v11 | h1r_nonstandard_component_class | 2 | 2 | 1.0 | 2 | 1.0 |
| component_label_guard_v11 | h1r_stale_selection_component_label | 2 | 2 | 1.0 | 2 | 1.0 |
| component_residual_guard_v12 | h1r_code_label_exactness | 2 | 2 | 1.0 | 2 | 1.0 |
| component_residual_guard_v12 | h1r_nonstandard_component_class | 2 | 2 | 1.0 | 2 | 1.0 |
| component_residual_guard_v12 | h1r_stale_selection_component_label | 2 | 2 | 1.0 | 2 | 1.0 |

## Comparison Rows

| comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tool_probe_replay_live_comparisons/20260510T_h1r_component_label_guard_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | 6 | 0.0 | 0.8333333333333334 | 0.8333333333333334 | 0.16666666666666666 | 0.8333333333333334 | 0.6666666666666667 |
| results/tool_probe_replay_live_comparisons/20260510T_h1r_component_residual_guard_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | 6 | 0.0 | 1.0 | 1.0 | 0.16666666666666666 | 1.0 | 0.8333333333333334 |
| results/tool_probe_replay_live_comparisons/20260510T_h1r_component_residual_guard_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | 6 | 0.8333333333333334 | 1.0 | 0.16666666666666663 | 0.8333333333333334 | 1.0 | 0.16666666666666663 |

## Findings

| finding_id | finding |
| --- | --- |
| h1r_breaks_no_directive | No-directive reaches 0/6 exact and 1/6 executor-equivalent, so H1r is a useful residual discriminator. |
| v11_is_strong_incumbent | Component-label guard v11 reaches 5/6 exact and 5/6 executor-equivalent, leaving only the alert-s92 code-label miss. |
| v12_saturates_h1r | Component-residual guard v12 reaches 6/6 exact and 6/6 executor-equivalent, improving over v11 by 0.167 exact-rate. |

## Interpretation

This is positive residual evidence for v12, but not yet a global promotion. The next test should transfer v12 back across H1n/H1o/H1p and verify that the extra residual wording does not reintroduce the H1p executor-equivalence loss or the H1n broad-prose regressions.
