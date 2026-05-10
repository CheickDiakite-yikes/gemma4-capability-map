# H1q Component-Label Guard Transfer Synthesis

Generated: `2026-05-10T17:24:54.482531+00:00`

## Aggregate Profile Summary

| profile_label | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| argument_hints_v2 | `32` | `21` | `0.65625` | `23` | `0.71875` | `8` | `0` | `2` |
| component_label_guard_v11 | `32` | `26` | `0.81250` | `29` | `0.90625` | `2` | `1` | `3` |
| component_value_guard_v9 | `32` | `23` | `0.71875` | `25` | `0.78125` | `6` | `1` | `2` |
| hybrid_label_guard_v8 | `32` | `23` | `0.71875` | `27` | `0.84375` | `5` | `0` | `4` |
| no_call_control_rescue_v10 | `32` | `20` | `0.62500` | `22` | `0.68750` | `9` | `0` | `2` |
| no_directive | `32` | `10` | `0.31250` | `12` | `0.37500` | `11` | `0` | `2` |

## Packet Summary

| packet_label | profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1n_component_value | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1n_component_value_no_directive_execute_v1 | `8` | `5` | `0.62500` | `6` | `0.75000` | `0` | `0` | `1` |
| h1n_component_value | argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | results/tool_probe_replay_live/20260510T_h1n_component_value_argument_hints_execute_v1 | `8` | `6` | `0.75000` | `7` | `0.87500` | `1` | `0` | `1` |
| h1n_component_value | hybrid_label_guard_v8 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard | results/tool_probe_replay_live/20260510T_h1n_component_value_hybrid_label_guard_execute_v1 | `8` | `6` | `0.75000` | `7` | `0.87500` | `1` | `0` | `1` |
| h1n_component_value | component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | results/tool_probe_replay_live/20260510T_h1n_component_value_component_value_guard_execute_v1 | `8` | `4` | `0.50000` | `4` | `0.50000` | `4` | `0` | `0` |
| h1n_component_value | no_call_control_rescue_v10 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue | results/tool_probe_replay_live/20260510T_h1n_component_value_no_call_control_rescue_execute_v1 | `8` | `7` | `0.87500` | `8` | `1.00000` | `0` | `0` | `1` |
| h1n_component_value | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1 | `8` | `6` | `0.75000` | `7` | `0.87500` | `0` | `1` | `1` |
| h1o_control_factorial | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_directive_execute_v1 | `12` | `5` | `0.41667` | `6` | `0.50000` | `3` | `0` | `1` |
| h1o_control_factorial | argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | results/tool_probe_replay_live/20260510T_h1o_control_factorial_argument_hints_execute_v1 | `12` | `9` | `0.75000` | `10` | `0.83333` | `1` | `0` | `1` |
| h1o_control_factorial | hybrid_label_guard_v8 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard | results/tool_probe_replay_live/20260510T_h1o_control_factorial_hybrid_label_guard_execute_v1 | `12` | `8` | `0.66667` | `10` | `0.83333` | `2` | `0` | `2` |
| h1o_control_factorial | component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | results/tool_probe_replay_live/20260510T_h1o_control_factorial_component_value_guard_execute_v1 | `12` | `9` | `0.75000` | `10` | `0.83333` | `2` | `0` | `1` |
| h1o_control_factorial | no_call_control_rescue_v10 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue | results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_call_control_rescue_execute_v1 | `12` | `7` | `0.58333` | `8` | `0.66667` | `3` | `0` | `1` |
| h1o_control_factorial | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1 | `12` | `10` | `0.83333` | `12` | `1.00000` | `0` | `0` | `2` |
| h1p_component_value | no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1p_component_value_no_directive_execute_v1 | `12` | `0` | `0.00000` | `0` | `0.00000` | `8` | `0` | `0` |
| h1p_component_value | argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | results/tool_probe_replay_live/20260510T_h1p_component_value_argument_hints_execute_v1 | `12` | `6` | `0.50000` | `6` | `0.50000` | `6` | `0` | `0` |
| h1p_component_value | hybrid_label_guard_v8 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard | results/tool_probe_replay_live/20260510T_h1p_component_value_hybrid_label_guard_execute_v1 | `12` | `9` | `0.75000` | `10` | `0.83333` | `2` | `0` | `1` |
| h1p_component_value | component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | results/tool_probe_replay_live/20260510T_h1p_component_value_component_value_guard_execute_v1 | `12` | `10` | `0.83333` | `11` | `0.91667` | `0` | `1` | `1` |
| h1p_component_value | no_call_control_rescue_v10 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue | results/tool_probe_replay_live/20260510T_h1p_component_value_no_call_control_rescue_execute_v1 | `12` | `6` | `0.50000` | `6` | `0.50000` | `6` | `0` | `0` |
| h1p_component_value | component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1 | `12` | `10` | `0.83333` | `10` | `0.83333` | `2` | `0` | `0` |

## v11 Non-Exact Cases

| packet_label | case_id | family | failure_mode | executor_equivalence_match | output_dir |
| --- | --- | --- | --- | --- | --- |
| h1n_component_value | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | `true` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1/runs/component_value_result_pill_log_decoy |
| h1n_component_value | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | wrong_tool | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1/runs/component_value_owner_field_stale_selection_decoy |
| h1o_control_factorial | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | `true` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1/runs/h1o_code_alert_s92_negated_toggle_decoy |
| h1o_control_factorial | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | `true` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1/runs/h1o_code_badge_c08_note_decoy |
| h1p_component_value | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1/runs/h1p_compact_state_tag_log_value_decoy |
| h1p_component_value | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1/runs/h1p_surface_mode_toggle_note_value_decoy |

## Findings

| finding_id | finding |
| --- | --- |
| aggregate_strict_upper_bound | Aggregate strict upper bound across H1n/H1o/H1p is component_label_guard_v11 at 26/32. |
| aggregate_executor_upper_bound | Aggregate executor-equivalence upper bound is component_label_guard_v11 at 29/32. |
| v11_repairs_v9_h1n_regressions | v11 repairs the broad v9 regression on H1n component-value: 6/8 exact and 7/8 executor-equivalent versus v9 at 4/8 exact and 4/8 executor-equivalent. |
| v11_sets_h1o_executor_ceiling | v11 sets the current H1o transfer ceiling: 10/12 exact and 12/12 executor-equivalent. |
| h1p_tradeoff_vs_v9 | On H1p, v11 ties v9 strict exactness but loses one executor-equivalent case: v11 is 10/12 exact and 10/12 executor-equivalent, while v9 is 10/12 exact and 11/12 executor-equivalent. |
| remaining_v11_failures | Remaining non-executor v11 failures are h1n_component_value:component_value_owner_field_stale_selection_decoy, h1p_component_value:h1p_compact_state_tag_log_value_decoy, h1p_component_value:h1p_surface_mode_toggle_note_value_decoy. |
| promotion_decision | Treat v11 as the current best transfer candidate (26/32 exact and 29/32 executor-equivalent versus v9 at 23/32 and 25/32), but do not make it the global default until the remaining owner-field, state-tag, and mode-toggle failures are isolated. |
