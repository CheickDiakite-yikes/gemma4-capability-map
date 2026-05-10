# Visual Live Alias Transfer Oracle Diagnostic

Generated: `2026-05-10T00:23:54.123418+00:00`

## Findings

- `strict_upper_bound`: argument_hints_v2 is the strict upper bound at 0.8333333333333334.
- `executor_equivalence_set`: Executor-equivalent full-success rows: argument_hints_v2.
- `executor_without_strict`: Rows with executor gain without strict gain: argument_hints_v2.
- `regressions`: Regression cases: contracted:transfer_error_banner_note_decoy, contracted:transfer_queue_badge_person_decoy, schema_field_hints_v4:transfer_queue_badge_person_decoy, schema_literal_targets_v5:transfer_queue_badge_person_decoy.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | mlx_gemma4_e2b_reasoner_only | 6 | 0.3333333333333333 | 0.16666666666666666 | -0.16666666666666666 | 0.3333333333333333 | 0.16666666666666666 | -0.16666666666666666 |
| role_catalog_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | 6 | 0.3333333333333333 | 0.5 | 0.16666666666666669 | 0.3333333333333333 | 0.5 | 0.16666666666666669 |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 6 | 0.3333333333333333 | 0.8333333333333334 | 0.5 | 0.3333333333333333 | 1.0 | 0.6666666666666667 |
| schema_field_hints_v4 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | 6 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 0.3333333333333333 | 0.3333333333333333 | 0.0 |
| schema_literal_targets_v5 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets | 6 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | transfer_error_banner_note_decoy | visual_argument_transfer | exact | argument_mismatch | -1 | -1 | regression |
| contracted | transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | argument_mismatch | exact | 1 | 1 | strict_gain |
| contracted | transfer_queue_badge_person_decoy | visual_argument_transfer | exact | wrong_tool | -1 | -1 | regression |
| contracted | transfer_review_tile_notice_table_decoy | visual_argument_transfer | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| contracted | transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | argument_mismatch | wrong_tool | 0 | 0 | unchanged |
| contracted | transfer_status_pill_chart_decoy | visual_argument_transfer | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_error_banner_note_decoy | visual_argument_transfer | exact | exact | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | argument_mismatch | exact | 1 | 1 | strict_gain |
| role_catalog_v1 | transfer_queue_badge_person_decoy | visual_argument_transfer | exact | exact | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_review_tile_notice_table_decoy | visual_argument_transfer | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_status_pill_chart_decoy | visual_argument_transfer | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| argument_hints_v2 | transfer_error_banner_note_decoy | visual_argument_transfer | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | argument_mismatch | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_queue_badge_person_decoy | visual_argument_transfer | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | transfer_review_tile_notice_table_decoy | visual_argument_transfer | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | argument_mismatch | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_status_pill_chart_decoy | visual_argument_transfer | no_tool_call | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| schema_field_hints_v4 | transfer_error_banner_note_decoy | visual_argument_transfer | exact | exact | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | argument_mismatch | exact | 1 | 1 | strict_gain |
| schema_field_hints_v4 | transfer_queue_badge_person_decoy | visual_argument_transfer | exact | argument_mismatch | -1 | -1 | regression |
| schema_field_hints_v4 | transfer_review_tile_notice_table_decoy | visual_argument_transfer | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | argument_mismatch | no_tool_call | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_status_pill_chart_decoy | visual_argument_transfer | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_error_banner_note_decoy | visual_argument_transfer | exact | exact | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | argument_mismatch | exact | 1 | 1 | strict_gain |
| schema_literal_targets_v5 | transfer_queue_badge_person_decoy | visual_argument_transfer | exact | argument_mismatch | -1 | -1 | regression |
| schema_literal_targets_v5 | transfer_review_tile_notice_table_decoy | visual_argument_transfer | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | argument_mismatch | exact | 1 | 1 | strict_gain |
| schema_literal_targets_v5 | transfer_status_pill_chart_decoy | visual_argument_transfer | no_tool_call | exact | 1 | 1 | strict_gain |
