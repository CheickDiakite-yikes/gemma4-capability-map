# H1v Code-Label Exact Transfer Synthesis

Generated: `2026-05-10T18:25:59.016733+00:00`

## Summary

H1v rejects v15 as a global promotion. The code-label exact guard saturated H1r locally, but transfers to only `25 / 32` strict exact and `25 / 32` executor-equivalent successes across H1n/H1o/H1p.

## Packet Rows

| packet_label | profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | no_tool_call_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1n_component_value | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1n_component_value_no_directive_execute_v1 | `8` | `5` | `0.62500` | `6` | `0.75000` | `0` | `0` | `2` | `1` |
| h1n_component_value | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1 | `8` | `6` | `0.75000` | `7` | `0.87500` | `0` | `1` | `0` | `1` |
| h1n_component_value | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1 | `8` | `5` | `0.62500` | `5` | `0.62500` | `3` | `0` | `0` | `0` |
| h1n_component_value | code_label_exact_guard_v15 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1 | `8` | `5` | `0.62500` | `5` | `0.62500` | `2` | `1` | `0` | `0` |
| h1o_control_factorial | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_directive_execute_v1 | `12` | `5` | `0.41667` | `6` | `0.50000` | `3` | `0` | `3` | `1` |
| h1o_control_factorial | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1 | `12` | `10` | `0.83333` | `12` | `1.00000` | `0` | `0` | `0` | `2` |
| h1o_control_factorial | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1 | `12` | `11` | `0.91667` | `11` | `0.91667` | `1` | `0` | `0` | `0` |
| h1o_control_factorial | code_label_exact_guard_v15 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1o_control_factorial_execute_v1 | `12` | `11` | `0.91667` | `11` | `0.91667` | `1` | `0` | `0` | `0` |
| h1p_component_value | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1p_component_value_no_directive_execute_v1 | `12` | `0` | `0.00000` | `0` | `0.00000` | `8` | `0` | `4` | `0` |
| h1p_component_value | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1 | `12` | `10` | `0.83333` | `10` | `0.83333` | `2` | `0` | `0` | `0` |
| h1p_component_value | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1 | `12` | `11` | `0.91667` | `11` | `0.91667` | `0` | `1` | `0` | `0` |
| h1p_component_value | code_label_exact_guard_v15 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1 | `12` | `9` | `0.75000` | `9` | `0.75000` | `3` | `0` | `0` | `0` |

## Aggregate Rows

| profile_label | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | no_tool_call_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code_label_exact_guard_v15 | `32` | `25` | `0.78125` | `25` | `0.78125` | `6` | `1` | `0` | `0` |
| component_label_guard_v11 | `32` | `26` | `0.81250` | `29` | `0.90625` | `2` | `1` | `0` | `3` |
| component_residual_guard_v12 | `32` | `27` | `0.84375` | `27` | `0.84375` | `4` | `1` | `0` | `0` |
| no_directive | `32` | `10` | `0.31250` | `12` | `0.37500` | `11` | `0` | `9` | `2` |

## Family Rows

| packet_label | profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h1n_component_value | no_directive | visual_argument_transfer_component_value_nonpill | `3` | `1` | `0.33333` | `2` | `0.66667` |
| h1n_component_value | no_directive | visual_argument_transfer_component_value_pill | `3` | `3` | `1.00000` | `3` | `1.00000` |
| h1n_component_value | no_directive | visual_tool_routing_component_value | `2` | `1` | `0.50000` | `1` | `0.50000` |
| h1n_component_value | component_label_guard_v11 | visual_argument_transfer_component_value_nonpill | `3` | `3` | `1.00000` | `3` | `1.00000` |
| h1n_component_value | component_label_guard_v11 | visual_argument_transfer_component_value_pill | `3` | `2` | `0.66667` | `3` | `1.00000` |
| h1n_component_value | component_label_guard_v11 | visual_tool_routing_component_value | `2` | `1` | `0.50000` | `1` | `0.50000` |
| h1n_component_value | component_residual_guard_v12 | visual_argument_transfer_component_value_nonpill | `3` | `2` | `0.66667` | `2` | `0.66667` |
| h1n_component_value | component_residual_guard_v12 | visual_argument_transfer_component_value_pill | `3` | `1` | `0.33333` | `1` | `0.33333` |
| h1n_component_value | component_residual_guard_v12 | visual_tool_routing_component_value | `2` | `2` | `1.00000` | `2` | `1.00000` |
| h1n_component_value | code_label_exact_guard_v15 | visual_argument_transfer_component_value_nonpill | `3` | `2` | `0.66667` | `2` | `0.66667` |
| h1n_component_value | code_label_exact_guard_v15 | visual_argument_transfer_component_value_pill | `3` | `2` | `0.66667` | `2` | `0.66667` |
| h1n_component_value | code_label_exact_guard_v15 | visual_tool_routing_component_value | `2` | `1` | `0.50000` | `1` | `0.50000` |
| h1o_control_factorial | no_directive | h1o_activation_no_call | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1o_control_factorial | no_directive | h1o_code_negation_preservation | `4` | `1` | `0.25000` | `2` | `0.50000` |
| h1o_control_factorial | no_directive | h1o_component_value_boundary | `4` | `0` | `0.00000` | `0` | `0.00000` |
| h1o_control_factorial | component_label_guard_v11 | h1o_activation_no_call | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1o_control_factorial | component_label_guard_v11 | h1o_code_negation_preservation | `4` | `2` | `0.50000` | `4` | `1.00000` |
| h1o_control_factorial | component_label_guard_v11 | h1o_component_value_boundary | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1o_control_factorial | component_residual_guard_v12 | h1o_activation_no_call | `4` | `3` | `0.75000` | `3` | `0.75000` |
| h1o_control_factorial | component_residual_guard_v12 | h1o_code_negation_preservation | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1o_control_factorial | component_residual_guard_v12 | h1o_component_value_boundary | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1o_control_factorial | code_label_exact_guard_v15 | h1o_activation_no_call | `4` | `3` | `0.75000` | `3` | `0.75000` |
| h1o_control_factorial | code_label_exact_guard_v15 | h1o_code_negation_preservation | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1o_control_factorial | code_label_exact_guard_v15 | h1o_component_value_boundary | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1p_component_value | no_directive | h1p_component_value_compact | `4` | `0` | `0.00000` | `0` | `0.00000` |
| h1p_component_value | no_directive | h1p_component_value_stale_selection | `4` | `0` | `0.00000` | `0` | `0.00000` |
| h1p_component_value | no_directive | h1p_component_value_surface | `4` | `0` | `0.00000` | `0` | `0.00000` |
| h1p_component_value | component_label_guard_v11 | h1p_component_value_compact | `4` | `3` | `0.75000` | `3` | `0.75000` |
| h1p_component_value | component_label_guard_v11 | h1p_component_value_stale_selection | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1p_component_value | component_label_guard_v11 | h1p_component_value_surface | `4` | `3` | `0.75000` | `3` | `0.75000` |
| h1p_component_value | component_residual_guard_v12 | h1p_component_value_compact | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1p_component_value | component_residual_guard_v12 | h1p_component_value_stale_selection | `4` | `3` | `0.75000` | `3` | `0.75000` |
| h1p_component_value | component_residual_guard_v12 | h1p_component_value_surface | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1p_component_value | code_label_exact_guard_v15 | h1p_component_value_compact | `4` | `3` | `0.75000` | `3` | `0.75000` |
| h1p_component_value | code_label_exact_guard_v15 | h1p_component_value_stale_selection | `4` | `4` | `1.00000` | `4` | `1.00000` |
| h1p_component_value | code_label_exact_guard_v15 | h1p_component_value_surface | `4` | `2` | `0.50000` | `2` | `0.50000` |

## Comparison Rows

| comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tool_probe_replay_live_comparisons/20260510T_h1v_code_label_exact_guard_h1n_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `8` | `0.75000` | `0.62500` | `-0.12500` | `0.87500` | `0.62500` | `-0.25000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1v_code_label_exact_guard_h1n_vs_component_residual_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `8` | `0.62500` | `0.62500` | `0.00000` | `0.62500` | `0.62500` | `0.00000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1v_code_label_exact_guard_h1o_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `12` | `0.83333` | `0.91667` | `0.08333` | `1.00000` | `0.91667` | `-0.08333` |
| results/tool_probe_replay_live_comparisons/20260510T_h1v_code_label_exact_guard_h1o_vs_component_residual_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `12` | `0.91667` | `0.91667` | `0.00000` | `0.91667` | `0.91667` | `0.00000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1v_code_label_exact_guard_h1p_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `12` | `0.83333` | `0.75000` | `-0.08333` | `0.83333` | `0.75000` | `-0.08333` |
| results/tool_probe_replay_live_comparisons/20260510T_h1v_code_label_exact_guard_h1p_vs_component_residual_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `12` | `0.91667` | `0.75000` | `-0.16667` | `0.91667` | `0.75000` | `-0.16667` |

## V15 Non-Exact Rows

| packet_label | case_id | family | failure_mode | executor_equivalence_match | output_dir |
| --- | --- | --- | --- | --- | --- |
| h1n_component_value | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1/runs/component_value_state_pill_note_decoy |
| h1n_component_value | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1/runs/component_value_status_badge_email_decoy |
| h1n_component_value | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | wrong_tool | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1/runs/component_value_owner_field_stale_selection_decoy |
| h1o_control_factorial | h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1o_control_factorial_execute_v1/runs/h1o_activation_warning_tile_no_call_decoy |
| h1p_component_value | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1/runs/h1p_compact_state_tag_log_value_decoy |
| h1p_component_value | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1/runs/h1p_surface_mode_toggle_note_value_decoy |
| h1p_component_value | h1p_surface_result_badge_comment_value_decoy | h1p_component_value_surface | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1/runs/h1p_surface_result_badge_comment_value_decoy |

## Findings

| finding_id | finding |
| --- | --- |
| v15_not_global_promotion | V15 reaches 25/32 exact and 25/32 executor-equivalent transfer successes, below v11's 29/32 executor-equivalent and v12's 27/32 exact totals. |
| h1n_negative_transfer_persists | V15 ties v12 on H1n but remains below v11, with -0.125 exact-rate and -0.250 executor-rate deltas versus v11. |
| h1o_code_gain_has_executor_cost | V15 improves H1o strict exactness versus v11 by 0.083, but loses executor-equivalence by -0.083. |
| h1p_component_value_regression | V15 loses the H1p component-value holdout against v12 by -0.167 exact-rate and -0.167 executor-rate. |
| next_slice | Keep v11 as the transfer-stable default. Treat v15 as a local code-label repair and design the next slice around the remaining v15 failures: component_value_state_pill_note_decoy, component_value_status_badge_email_decoy, component_value_owner_field_stale_selection_decoy, h1o_activation_warning_tile_no_call_decoy, h1p_compact_state_tag_log_value_decoy, h1p_surface_mode_toggle_note_value_decoy, h1p_surface_result_badge_comment_value_decoy. |
