# H1w Residual-Overlap Synthesis

Generated: `2026-05-10T18:38:12.756732+00:00`

## Summary

H1w creates a fresh residual-overlap packet that breaks no-directive (`0 / 8`) while component-label guard v11 saturates (`8 / 8`). V12 and v15 repair most cases but regress on surface component-value rows.

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | wrong_tool_count | no_tool_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1 | `8` | `0` | `0.00000` | `0` | `0.00000` | `2` | `0` | `6` |
| component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_label_guard_execute_v1 | `8` | `8` | `1.00000` | `8` | `1.00000` | `0` | `0` | `0` |
| component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_residual_guard_execute_v1 | `8` | `7` | `0.87500` | `7` | `0.87500` | `1` | `0` | `0` |
| code_label_exact_guard_v15 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | results/tool_probe_replay_live/20260510T_h1w_residual_overlap_code_label_exact_guard_execute_v1 | `8` | `6` | `0.75000` | `6` | `0.75000` | `2` | `0` | `0` |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| no_directive | h1w_activation_no_call | `2` | `0` | `0.00000` | `0` | `0.00000` |
| no_directive | h1w_nonstandard_component_class | `2` | `0` | `0.00000` | `0` | `0.00000` |
| no_directive | h1w_stale_field_routing | `2` | `0` | `0.00000` | `0` | `0.00000` |
| no_directive | h1w_surface_component_value | `2` | `0` | `0.00000` | `0` | `0.00000` |
| component_label_guard_v11 | h1w_activation_no_call | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_label_guard_v11 | h1w_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_label_guard_v11 | h1w_stale_field_routing | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_label_guard_v11 | h1w_surface_component_value | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1w_activation_no_call | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1w_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1w_stale_field_routing | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1w_surface_component_value | `2` | `1` | `0.50000` | `1` | `0.50000` |
| code_label_exact_guard_v15 | h1w_activation_no_call | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1w_nonstandard_component_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1w_stale_field_routing | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1w_surface_component_value | `2` | `0` | `0.00000` | `0` | `0.00000` |

## Comparison Rows

| comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tool_probe_replay_live_comparisons/20260510T_h1w_component_label_guard_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | `8` | `0.00000` | `1.00000` | `1.00000` | `0.00000` | `1.00000` | `1.00000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1w_component_residual_guard_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `8` | `0.00000` | `0.87500` | `0.87500` | `0.00000` | `0.87500` | `0.87500` |
| results/tool_probe_replay_live_comparisons/20260510T_h1w_code_label_exact_guard_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `8` | `0.00000` | `0.75000` | `0.75000` | `0.00000` | `0.75000` | `0.75000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1w_component_residual_guard_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `8` | `1.00000` | `0.87500` | `-0.12500` | `1.00000` | `0.87500` | `-0.12500` |
| results/tool_probe_replay_live_comparisons/20260510T_h1w_code_label_exact_guard_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `8` | `1.00000` | `0.75000` | `-0.25000` | `1.00000` | `0.75000` | `-0.25000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1w_code_label_exact_guard_vs_component_residual_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `8` | `0.87500` | `0.75000` | `-0.12500` | `0.87500` | `0.75000` | `-0.12500` |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | output_dir |
| --- | --- | --- | --- | --- | --- |
| no_directive | h1w_owner_field_memo_stale_selection_decoy | h1w_stale_field_routing | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1/runs/h1w_owner_field_memo_stale_selection_decoy |
| no_directive | h1w_assignee_field_archive_summary_decoy | h1w_stale_field_routing | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1/runs/h1w_assignee_field_archive_summary_decoy |
| no_directive | h1w_state_tag_audit_log_value_decoy | h1w_nonstandard_component_class | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1/runs/h1w_state_tag_audit_log_value_decoy |
| no_directive | h1w_mode_toggle_settings_note_decoy | h1w_nonstandard_component_class | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1/runs/h1w_mode_toggle_settings_note_decoy |
| no_directive | h1w_result_badge_comment_value_decoy | h1w_surface_component_value | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1/runs/h1w_result_badge_comment_value_decoy |
| no_directive | h1w_status_pill_summary_value_decoy | h1w_surface_component_value | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1/runs/h1w_status_pill_summary_value_decoy |
| no_directive | h1w_warning_tile_no_call_note_decoy | h1w_activation_no_call | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1/runs/h1w_warning_tile_no_call_note_decoy |
| no_directive | h1w_error_banner_no_call_history_decoy | h1w_activation_no_call | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1/runs/h1w_error_banner_no_call_history_decoy |
| component_residual_guard_v12 | h1w_status_pill_summary_value_decoy | h1w_surface_component_value | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_residual_guard_execute_v1/runs/h1w_status_pill_summary_value_decoy |
| code_label_exact_guard_v15 | h1w_result_badge_comment_value_decoy | h1w_surface_component_value | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_code_label_exact_guard_execute_v1/runs/h1w_result_badge_comment_value_decoy |
| code_label_exact_guard_v15 | h1w_status_pill_summary_value_decoy | h1w_surface_component_value | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1w_residual_overlap_code_label_exact_guard_execute_v1/runs/h1w_status_pill_summary_value_decoy |

## Findings

| finding_id | finding |
| --- | --- |
| h1w_breaks_no_directive | No-directive reaches 0/8 exact and 0/8 executor-equivalent, so the residual-overlap packet is a real controller-dependence probe. |
| v11_saturates_h1w | Component-label guard v11 reaches 8/8 exact and 8/8 executor-equivalent, with a +1.000 exact-rate delta versus no-directive. |
| broader_residual_wording_regresses_surface_value | V12 reaches 7/8 and v15 reaches 6/8; the remaining non-exact rows are h1w_status_pill_summary_value_decoy, h1w_result_badge_comment_value_decoy, h1w_status_pill_summary_value_decoy. |
| v15_surface_value_weakness | V15's code-label exactness wording transfers cleanly on stale routing, nonstandard class, and activation/no-call rows, but is 0/2 on surface component-value. |
| next_slice | Keep v11 as the default. The next hard slice should break v11 more directly by combining component-label requests with oblique labels, old selections, and repeated values in one case. |
