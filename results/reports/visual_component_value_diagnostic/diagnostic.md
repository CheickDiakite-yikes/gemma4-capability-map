# Visual Live Component Value Diagnostic

Generated: `2026-05-10T15:53:13.913211+00:00`

## Findings

- `strict_upper_bound`: argument_hints_v2 is the strict upper bound at 0.75.
- `executor_equivalence_set`: Executor-equivalent full-success rows: none.
- `executor_without_strict`: Rows with executor gain without strict gain: none.
- `regressions`: Regression cases: argument_hints_v2:component_value_phase_tile_ticket_decoy, component_value_guard_v9:component_value_priority_chip_table_decoy, component_value_guard_v9:component_value_result_pill_log_decoy, component_value_guard_v9:component_value_state_pill_note_decoy, contracted:component_value_phase_tile_ticket_decoy, contracted:component_value_result_pill_log_decoy, contracted:component_value_risk_badge_stale_selection_decoy, contracted:component_value_severity_pill_chart_decoy, contracted:component_value_state_pill_note_decoy, hybrid_label_guard_v8:component_value_priority_chip_table_decoy, hybrid_label_guard_v8:component_value_result_pill_log_decoy, oblique_code_guard_v7:component_value_phase_tile_ticket_decoy, oblique_code_guard_v7:component_value_priority_chip_table_decoy, oblique_code_hints_v6:component_value_phase_tile_ticket_decoy, oblique_code_hints_v6:component_value_priority_chip_table_decoy, oblique_code_hints_v6:component_value_risk_badge_stale_selection_decoy, oblique_code_hints_v6:component_value_state_pill_note_decoy, schema_field_hints_v4:component_value_phase_tile_ticket_decoy, schema_field_hints_v4:component_value_priority_chip_table_decoy, schema_field_hints_v4:component_value_result_pill_log_decoy, schema_field_hints_v4:component_value_severity_pill_chart_decoy.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | mlx_gemma4_e2b_reasoner_only | 8 | 0.625 | 0.125 | -0.5 | 0.75 | 0.125 | -0.625 |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 8 | 0.625 | 0.75 | 0.125 | 0.75 | 0.875 | 0.125 |
| hybrid_label_guard_v8 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard | 8 | 0.625 | 0.75 | 0.125 | 0.75 | 0.875 | 0.125 |
| component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | 8 | 0.625 | 0.5 | -0.125 | 0.75 | 0.5 | -0.25 |
| oblique_code_guard_v7 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard | 8 | 0.625 | 0.625 | 0.0 | 0.75 | 0.625 | -0.125 |
| oblique_code_hints_v6 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints | 8 | 0.625 | 0.25 | -0.375 | 0.75 | 0.25 | -0.5 |
| schema_field_hints_v4 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | 8 | 0.625 | 0.375 | -0.25 | 0.75 | 0.5 | -0.25 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| contracted | component_value_phase_tile_ticket_decoy | visual_argument_transfer_component_value_nonpill | exact | wrong_tool | -1 | -1 | regression |
| contracted | component_value_priority_chip_table_decoy | visual_argument_transfer_component_value_nonpill | executable_paraphrase | exact | 1 | 0 | strict_gain |
| contracted | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | exact | wrong_tool | -1 | -1 | regression |
| contracted | component_value_risk_badge_stale_selection_decoy | visual_tool_routing_component_value | exact | argument_mismatch | -1 | -1 | regression |
| contracted | component_value_severity_pill_chart_decoy | visual_argument_transfer_component_value_pill | exact | wrong_tool | -1 | -1 | regression |
| contracted | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | exact | wrong_tool | -1 | -1 | regression |
| contracted | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | no_tool_call | wrong_tool | 0 | 0 | unchanged |
| argument_hints_v2 | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | component_value_phase_tile_ticket_decoy | visual_argument_transfer_component_value_nonpill | exact | argument_mismatch | -1 | -1 | regression |
| argument_hints_v2 | component_value_priority_chip_table_decoy | visual_argument_transfer_component_value_nonpill | executable_paraphrase | executable_paraphrase | 0 | 0 | unchanged |
| argument_hints_v2 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | component_value_risk_badge_stale_selection_decoy | visual_tool_routing_component_value | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | component_value_severity_pill_chart_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | no_tool_call | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | no_tool_call | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | component_value_phase_tile_ticket_decoy | visual_argument_transfer_component_value_nonpill | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | component_value_priority_chip_table_decoy | visual_argument_transfer_component_value_nonpill | executable_paraphrase | argument_mismatch | 0 | -1 | regression |
| hybrid_label_guard_v8 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | exact | executable_paraphrase | -1 | 0 | regression |
| hybrid_label_guard_v8 | component_value_risk_badge_stale_selection_decoy | visual_tool_routing_component_value | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | component_value_severity_pill_chart_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | no_tool_call | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| component_value_guard_v9 | component_value_phase_tile_ticket_decoy | visual_argument_transfer_component_value_nonpill | exact | exact | 0 | 0 | unchanged |
| component_value_guard_v9 | component_value_priority_chip_table_decoy | visual_argument_transfer_component_value_nonpill | executable_paraphrase | argument_mismatch | 0 | -1 | regression |
| component_value_guard_v9 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | exact | argument_mismatch | -1 | -1 | regression |
| component_value_guard_v9 | component_value_risk_badge_stale_selection_decoy | visual_tool_routing_component_value | exact | exact | 0 | 0 | unchanged |
| component_value_guard_v9 | component_value_severity_pill_chart_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| component_value_guard_v9 | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | exact | argument_mismatch | -1 | -1 | regression |
| component_value_guard_v9 | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | no_tool_call | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | no_tool_call | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | component_value_phase_tile_ticket_decoy | visual_argument_transfer_component_value_nonpill | exact | argument_mismatch | -1 | -1 | regression |
| oblique_code_guard_v7 | component_value_priority_chip_table_decoy | visual_argument_transfer_component_value_nonpill | executable_paraphrase | argument_mismatch | 0 | -1 | regression |
| oblique_code_guard_v7 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | component_value_risk_badge_stale_selection_decoy | visual_tool_routing_component_value | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | component_value_severity_pill_chart_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| oblique_code_hints_v6 | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| oblique_code_hints_v6 | component_value_phase_tile_ticket_decoy | visual_argument_transfer_component_value_nonpill | exact | argument_mismatch | -1 | -1 | regression |
| oblique_code_hints_v6 | component_value_priority_chip_table_decoy | visual_argument_transfer_component_value_nonpill | executable_paraphrase | argument_mismatch | 0 | -1 | regression |
| oblique_code_hints_v6 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| oblique_code_hints_v6 | component_value_risk_badge_stale_selection_decoy | visual_tool_routing_component_value | exact | no_tool_call | -1 | -1 | regression |
| oblique_code_hints_v6 | component_value_severity_pill_chart_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| oblique_code_hints_v6 | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | exact | no_tool_call | -1 | -1 | regression |
| oblique_code_hints_v6 | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| schema_field_hints_v4 | component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_field_hints_v4 | component_value_phase_tile_ticket_decoy | visual_argument_transfer_component_value_nonpill | exact | argument_mismatch | -1 | -1 | regression |
| schema_field_hints_v4 | component_value_priority_chip_table_decoy | visual_argument_transfer_component_value_nonpill | executable_paraphrase | argument_mismatch | 0 | -1 | regression |
| schema_field_hints_v4 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | exact | executable_paraphrase | -1 | 0 | regression |
| schema_field_hints_v4 | component_value_risk_badge_stale_selection_decoy | visual_tool_routing_component_value | exact | exact | 0 | 0 | unchanged |
| schema_field_hints_v4 | component_value_severity_pill_chart_decoy | visual_argument_transfer_component_value_pill | exact | argument_mismatch | -1 | -1 | regression |
| schema_field_hints_v4 | component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | exact | exact | 0 | 0 | unchanged |
| schema_field_hints_v4 | component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
