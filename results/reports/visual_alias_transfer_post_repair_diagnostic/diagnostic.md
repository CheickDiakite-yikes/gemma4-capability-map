# Visual Live Alias Transfer Post Repair Diagnostic

Generated: `2026-05-10T01:52:05.133393+00:00`

## Findings

- `strict_upper_bound`: oblique_code_guard_v7 is the strict upper bound at 0.75.
- `executor_equivalence_set`: Executor-equivalent full-success rows: none.
- `executor_without_strict`: Rows with executor gain without strict gain: none.
- `regressions`: Regression cases: contracted:post_repair_node_k21_chart_decoy, oblique_code_hints_v6:post_repair_review_tile_table_decoy.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | mlx_gemma4_e2b_reasoner_only | 8 | 0.25 | 0.375 | 0.125 | 0.25 | 0.375 | 0.125 |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 8 | 0.25 | 0.625 | 0.375 | 0.25 | 0.625 | 0.375 |
| oblique_code_hints_v6 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints | 8 | 0.25 | 0.625 | 0.375 | 0.25 | 0.625 | 0.375 |
| oblique_code_guard_v7 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard | 8 | 0.25 | 0.75 | 0.5 | 0.25 | 0.75 | 0.5 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | post_repair_alert_c77_toggle_decoy | visual_tool_routing_transfer_post_repair | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| contracted | post_repair_badge_t64_notice_decoy | visual_argument_transfer_post_repair_code | argument_mismatch | exact | 1 | 1 | strict_gain |
| contracted | post_repair_chip_l90_person_decoy | visual_argument_transfer_post_repair_code | argument_mismatch | wrong_tool | 0 | 0 | unchanged |
| contracted | post_repair_field_b12_stale_selection_decoy | visual_tool_routing_transfer_post_repair | no_tool_call | exact | 1 | 1 | strict_gain |
| contracted | post_repair_node_k21_chart_decoy | visual_argument_transfer_post_repair_code | exact | wrong_tool | -1 | -1 | regression |
| contracted | post_repair_review_tile_table_decoy | visual_argument_transfer_post_repair_noncode | exact | exact | 0 | 0 | unchanged |
| contracted | post_repair_status_pill_note_decoy | visual_argument_transfer_post_repair_noncode | no_tool_call | wrong_tool | 0 | 0 | unchanged |
| contracted | post_repair_warning_toast_email_decoy | visual_argument_transfer_post_repair_noncode | no_tool_call | wrong_tool | 0 | 0 | unchanged |
| argument_hints_v2 | post_repair_alert_c77_toggle_decoy | visual_tool_routing_transfer_post_repair | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | post_repair_badge_t64_notice_decoy | visual_argument_transfer_post_repair_code | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | post_repair_chip_l90_person_decoy | visual_argument_transfer_post_repair_code | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | post_repair_field_b12_stale_selection_decoy | visual_tool_routing_transfer_post_repair | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | post_repair_node_k21_chart_decoy | visual_argument_transfer_post_repair_code | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | post_repair_review_tile_table_decoy | visual_argument_transfer_post_repair_noncode | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | post_repair_status_pill_note_decoy | visual_argument_transfer_post_repair_noncode | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | post_repair_warning_toast_email_decoy | visual_argument_transfer_post_repair_noncode | no_tool_call | exact | 1 | 1 | strict_gain |
| oblique_code_hints_v6 | post_repair_alert_c77_toggle_decoy | visual_tool_routing_transfer_post_repair | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_hints_v6 | post_repair_badge_t64_notice_decoy | visual_argument_transfer_post_repair_code | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_hints_v6 | post_repair_chip_l90_person_decoy | visual_argument_transfer_post_repair_code | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| oblique_code_hints_v6 | post_repair_field_b12_stale_selection_decoy | visual_tool_routing_transfer_post_repair | no_tool_call | exact | 1 | 1 | strict_gain |
| oblique_code_hints_v6 | post_repair_node_k21_chart_decoy | visual_argument_transfer_post_repair_code | exact | exact | 0 | 0 | unchanged |
| oblique_code_hints_v6 | post_repair_review_tile_table_decoy | visual_argument_transfer_post_repair_noncode | exact | no_tool_call | -1 | -1 | regression |
| oblique_code_hints_v6 | post_repair_status_pill_note_decoy | visual_argument_transfer_post_repair_noncode | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| oblique_code_hints_v6 | post_repair_warning_toast_email_decoy | visual_argument_transfer_post_repair_noncode | no_tool_call | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | post_repair_alert_c77_toggle_decoy | visual_tool_routing_transfer_post_repair | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | post_repair_badge_t64_notice_decoy | visual_argument_transfer_post_repair_code | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | post_repair_chip_l90_person_decoy | visual_argument_transfer_post_repair_code | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| oblique_code_guard_v7 | post_repair_field_b12_stale_selection_decoy | visual_tool_routing_transfer_post_repair | no_tool_call | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | post_repair_node_k21_chart_decoy | visual_argument_transfer_post_repair_code | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | post_repair_review_tile_table_decoy | visual_argument_transfer_post_repair_noncode | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | post_repair_status_pill_note_decoy | visual_argument_transfer_post_repair_noncode | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| oblique_code_guard_v7 | post_repair_warning_toast_email_decoy | visual_argument_transfer_post_repair_noncode | no_tool_call | exact | 1 | 1 | strict_gain |
