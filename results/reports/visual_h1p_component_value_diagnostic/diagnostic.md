# Visual Live H1P Component Value Diagnostic

Generated: `2026-05-10T17:05:27.814723+00:00`

## Findings

- `strict_upper_bound`: component_value_guard_v9 is the strict upper bound at 0.8333333333333334.
- `executor_equivalence_set`: Executor-equivalent full-success rows: none.
- `executor_without_strict`: Rows with executor gain without strict gain: component_value_guard_v9, hybrid_label_guard_v8.
- `regressions`: Regression cases: none.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 12 | 0.0 | 0.5 | 0.5 | 0.0 | 0.5 | 0.5 |
| hybrid_label_guard_v8 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard | 12 | 0.0 | 0.75 | 0.75 | 0.0 | 0.8333333333333334 | 0.8333333333333334 |
| no_call_control_rescue_v10 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue | 12 | 0.0 | 0.5 | 0.5 | 0.0 | 0.5 | 0.5 |
| component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | 12 | 0.0 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 0.9166666666666666 | 0.9166666666666666 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| argument_hints_v2 | h1p_compact_queue_badge_table_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1p_compact_stage_chip_email_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | h1p_compact_status_pill_summary_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | h1p_stale_phase_tile_archive_decoy | h1p_component_value_stale_selection | argument_mismatch | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1p_stale_priority_chip_old_selection_decoy | h1p_component_value_stale_selection | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1p_stale_risk_badge_old_selection_decoy | h1p_component_value_stale_selection | argument_mismatch | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1p_stale_severity_pill_previous_region_decoy | h1p_component_value_stale_selection | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1p_surface_lane_tile_board_value_decoy | h1p_component_value_surface | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | h1p_surface_owner_field_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1p_surface_result_badge_comment_value_decoy | h1p_component_value_surface | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1p_compact_queue_badge_table_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_compact_stage_chip_email_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1p_compact_status_pill_summary_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_stale_phase_tile_archive_decoy | h1p_component_value_stale_selection | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_stale_priority_chip_old_selection_decoy | h1p_component_value_stale_selection | no_tool_call | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_stale_risk_badge_old_selection_decoy | h1p_component_value_stale_selection | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_stale_severity_pill_previous_region_decoy | h1p_component_value_stale_selection | no_tool_call | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_surface_lane_tile_board_value_decoy | h1p_component_value_surface | no_tool_call | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1p_surface_owner_field_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1p_surface_result_badge_comment_value_decoy | h1p_component_value_surface | argument_mismatch | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| no_call_control_rescue_v10 | h1p_compact_queue_badge_table_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1p_compact_stage_chip_email_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1p_compact_status_pill_summary_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1p_stale_phase_tile_archive_decoy | h1p_component_value_stale_selection | argument_mismatch | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1p_stale_priority_chip_old_selection_decoy | h1p_component_value_stale_selection | no_tool_call | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1p_stale_risk_badge_old_selection_decoy | h1p_component_value_stale_selection | argument_mismatch | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1p_stale_severity_pill_previous_region_decoy | h1p_component_value_stale_selection | no_tool_call | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1p_surface_lane_tile_board_value_decoy | h1p_component_value_surface | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1p_surface_owner_field_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1p_surface_result_badge_comment_value_decoy | h1p_component_value_surface | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| component_value_guard_v9 | h1p_compact_queue_badge_table_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_compact_stage_chip_email_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_compact_status_pill_summary_value_decoy | h1p_component_value_compact | argument_mismatch | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_stale_phase_tile_archive_decoy | h1p_component_value_stale_selection | argument_mismatch | wrong_tool | 0 | 0 | unchanged |
| component_value_guard_v9 | h1p_stale_priority_chip_old_selection_decoy | h1p_component_value_stale_selection | no_tool_call | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_stale_risk_badge_old_selection_decoy | h1p_component_value_stale_selection | argument_mismatch | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_stale_severity_pill_previous_region_decoy | h1p_component_value_stale_selection | no_tool_call | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_surface_lane_tile_board_value_decoy | h1p_component_value_surface | no_tool_call | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| component_value_guard_v9 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | no_tool_call | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_surface_owner_field_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1p_surface_result_badge_comment_value_decoy | h1p_component_value_surface | argument_mismatch | exact | 1 | 1 | strict_gain |
