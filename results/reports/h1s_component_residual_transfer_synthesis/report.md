# H1s Component-Residual Transfer Synthesis

Generated: `2026-05-10T17:54:44.790608+00:00`

## Summary

H1s tests whether the H1r v12 residual prompt should become the global visual-role catalog default. The answer is no for now: v12 is a useful targeted patch, but it trades executor robustness for strict exactness on the broader H1n/H1o/H1p transfer surface.

## Transfer Aggregate

| profile_label | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | no_tool_call_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| component_label_guard_v11 | `32` | `26` | `0.81250` | `29` | `0.90625` | `2` | `1` | `0` | `3` |
| component_residual_guard_v12 | `32` | `27` | `0.84375` | `27` | `0.84375` | `4` | `1` | `0` | `0` |
| no_directive | `32` | `10` | `0.31250` | `12` | `0.37500` | `11` | `0` | `9` | `2` |

## Packet Rows

| packet_label | profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | no_tool_call_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1r_component_residual | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_no_directive_execute_v1 | `6` | `0` | `0.00000` | `1` | `0.16667` | `2` | `0` | `3` | `1` |
| h1r_component_residual | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_label_guard_execute_v1 | `6` | `5` | `0.83333` | `5` | `0.83333` | `1` | `0` | `0` | `0` |
| h1r_component_residual | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_residual_guard_execute_v1 | `6` | `6` | `1.00000` | `6` | `1.00000` | `0` | `0` | `0` | `0` |
| h1n_component_value | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1n_component_value_no_directive_execute_v1 | `8` | `5` | `0.62500` | `6` | `0.75000` | `0` | `0` | `2` | `1` |
| h1n_component_value | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1 | `8` | `6` | `0.75000` | `7` | `0.87500` | `0` | `1` | `0` | `1` |
| h1n_component_value | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1 | `8` | `5` | `0.62500` | `5` | `0.62500` | `3` | `0` | `0` | `0` |
| h1o_control_factorial | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_directive_execute_v1 | `12` | `5` | `0.41667` | `6` | `0.50000` | `3` | `0` | `3` | `1` |
| h1o_control_factorial | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1 | `12` | `10` | `0.83333` | `12` | `1.00000` | `0` | `0` | `0` | `2` |
| h1o_control_factorial | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1 | `12` | `11` | `0.91667` | `11` | `0.91667` | `1` | `0` | `0` | `0` |
| h1p_component_value | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1p_component_value_no_directive_execute_v1 | `12` | `0` | `0.00000` | `0` | `0.00000` | `8` | `0` | `4` | `0` |
| h1p_component_value | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1 | `12` | `10` | `0.83333` | `10` | `0.83333` | `2` | `0` | `0` | `0` |
| h1p_component_value | component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1 | `12` | `11` | `0.91667` | `11` | `0.91667` | `0` | `1` | `0` | `0` |

## Pairwise Comparisons

| comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1n_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `8` | `0.75000` | `0.62500` | `-0.12500` | `0.87500` | `0.62500` | `-0.25000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1n_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `8` | `0.62500` | `0.62500` | `0.00000` | `0.75000` | `0.62500` | `-0.12500` |
| results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1o_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `12` | `0.83333` | `0.91667` | `0.08333` | `1.00000` | `0.91667` | `-0.08333` |
| results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1o_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `12` | `0.41667` | `0.91667` | `0.50000` | `0.50000` | `0.91667` | `0.41667` |
| results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1p_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `12` | `0.83333` | `0.91667` | `0.08333` | `0.83333` | `0.91667` | `0.08333` |
| results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1p_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `12` | `0.00000` | `0.91667` | `0.91667` | `0.00000` | `0.91667` | `0.91667` |

## v12 Non-Exact Cases

| packet_label | case_id | family | failure_mode | executor_equivalence_match | output_dir |
| --- | --- | --- | --- | --- | --- |
| h1n_component_value | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1/runs/component_value_state_pill_note_decoy |
| h1n_component_value | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1/runs/component_value_status_badge_email_decoy |
| h1n_component_value | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1/runs/component_value_result_pill_log_decoy |
| h1o_control_factorial | h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1/runs/h1o_activation_warning_tile_no_call_decoy |
| h1p_component_value | h1p_stale_phase_tile_archive_decoy | h1p_component_value_stale_selection | wrong_tool | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1/runs/h1p_stale_phase_tile_archive_decoy |

## Findings

| finding_id | finding |
| --- | --- |
| v12_solves_local_h1r_residual | Component-residual guard v12 saturates the residual H1r slice at 6/6 exact and 6/6 executor-equivalent. |
| v12_transfers_strict_but_not_executor | Across H1n/H1o/H1p, v12 improves strict exactness from 26/32 to 27/32, but lowers executor-equivalence from 29/32 to 27/32. |
| negative_h1n_transfer | H1n is the clearest negative-transfer warning: v12 delta versus v11 is -0.125 exact-rate and -0.250 executor-rate. |
| h1o_strict_executor_split | On H1o, v12 improves strict exactness but loses executor-equivalence versus v11: delta exact-rate 0.083; delta executor-rate -0.083. |
| h1p_transfer_is_real_but_partial | On H1p, v12 improves both exact and executor-equivalence rates versus v11 by 0.083, but still leaves one wrong-tool stale-selection miss. |
| remaining_v12_failures | Remaining non-executor v12 failures are h1n_component_value:component_value_state_pill_note_decoy:argument_mismatch, h1n_component_value:component_value_status_badge_email_decoy:argument_mismatch, h1n_component_value:component_value_result_pill_log_decoy:argument_mismatch, h1o_control_factorial:h1o_activation_warning_tile_no_call_decoy:argument_mismatch, h1p_component_value:h1p_stale_phase_tile_archive_decoy:wrong_tool. |
| promotion_decision | Do not promote v12 as the global visual-role catalog default yet. Treat it as a targeted residual patch or conditional route while v11 remains the more executor-robust general transfer profile. |

## Interpretation

This is the strongest evidence so far that prompt-contract improvements need transfer gates, not just local residual wins. v12 should feed the next conditional-routing or prompt-factorial slice, while v11 remains the safer general-purpose component-label guard until the H1n executor regressions are removed.
