# H1t Conditional Residual-Route Synthesis

Generated: `2026-05-10T18:07:36.287975+00:00`

## Summary

H1t tests whether conditional v12-style residual wording can preserve H1r while avoiding the H1s transfer regressions. The answer is no for this profile: v13 fails the H1r early-stop gate at `3 / 6`, so broader H1n/H1o/H1p transfer was intentionally skipped.

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | no_tool_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_no_directive_execute_v1 | `6` | `0` | `0.00000` | `1` | `0.16667` | `2` | `0` | `3` |
| component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_label_guard_execute_v1 | `6` | `5` | `0.83333` | `5` | `0.83333` | `1` | `0` | `0` |
| component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_residual_guard_execute_v1 | `6` | `6` | `1.00000` | `6` | `1.00000` | `0` | `0` | `0` |
| conditional_residual_route_v13 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route | results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1 | `6` | `3` | `0.50000` | `3` | `0.50000` | `3` | `0` | `0` |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| no_directive | h1r_code_label_exactness | `2` | `0` | `0.00000` | `1` | `0.50000` |
| no_directive | h1r_nonstandard_component_class | `2` | `0` | `0.00000` | `0` | `0.00000` |
| no_directive | h1r_stale_selection_component_label | `2` | `0` | `0.00000` | `0` | `0.00000` |
| component_label_guard_v11 | h1r_code_label_exactness | `2` | `1` | `0.50000` | `1` | `0.50000` |
| component_label_guard_v11 | h1r_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_label_guard_v11 | h1r_stale_selection_component_label | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1r_code_label_exactness | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1r_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1r_stale_selection_component_label | `2` | `2` | `1.00000` | `2` | `1.00000` |
| conditional_residual_route_v13 | h1r_code_label_exactness | `2` | `1` | `0.50000` | `1` | `0.50000` |
| conditional_residual_route_v13 | h1r_nonstandard_component_class | `2` | `0` | `0.00000` | `0` | `0.00000` |
| conditional_residual_route_v13 | h1r_stale_selection_component_label | `2` | `2` | `1.00000` | `2` | `1.00000` |

## Comparison Rows

| comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tool_probe_replay_live_comparisons/20260510T_h1t_conditional_residual_route_h1r_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route | `6` | `0.00000` | `0.50000` | `0.50000` | `0.16667` | `0.50000` | `0.33333` |
| results/tool_probe_replay_live_comparisons/20260510T_h1t_conditional_residual_route_h1r_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route | `6` | `0.83333` | `0.50000` | `-0.33333` | `0.83333` | `0.50000` | `-0.33333` |
| results/tool_probe_replay_live_comparisons/20260510T_h1t_conditional_residual_route_h1r_vs_component_residual_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route | `6` | `1.00000` | `0.50000` | `-0.50000` | `1.00000` | `0.50000` | `-0.50000` |

## v13 Non-Exact Cases

| case_id | family | failure_mode | executor_equivalence_match | output_dir |
| --- | --- | --- | --- | --- |
| h1r_state_tag_log_value_decoy | h1r_nonstandard_component_class | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1/runs/h1r_state_tag_log_value_decoy |
| h1r_mode_toggle_note_value_decoy | h1r_nonstandard_component_class | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1/runs/h1r_mode_toggle_note_value_decoy |
| h1r_alert_s92_toggle_negation_decoy | h1r_code_label_exactness | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1/runs/h1r_alert_s92_toggle_negation_decoy |

## Findings

| finding_id | finding |
| --- | --- |
| v13_fails_h1r_gate | Conditional route v13 reaches only 3/6 exact and 3/6 executor-equivalent on H1r. |
| v13_below_v11_and_v12 | v13 is below v11 (5/6) and v12 (6/6); delta versus v11 is -0.333 exact-rate and delta versus v12 is -0.500. |
| failure_pattern | v13 failures are h1r_state_tag_log_value_decoy:argument_mismatch, h1r_mode_toggle_note_value_decoy:argument_mismatch, h1r_alert_s92_toggle_negation_decoy:argument_mismatch. |
| early_stop_decision | Stop before H1n/H1o/H1p transfer. A conditional route that cannot preserve the H1r local win is not a credible promotion candidate. |
