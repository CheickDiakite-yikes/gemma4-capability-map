# Visual Live Alias Transfer Repeat Diagnostic

Generated: `2026-05-10T00:51:27.981611+00:00`

## Findings

- `strict_upper_bound`: argument_hints_v2 is the strict upper bound at 0.8333333333333334.
- `executor_equivalence_set`: Executor-equivalent full-success rows: argument_hints_v2, schema_literal_targets_v5.
- `executor_without_strict`: Rows with executor gain without strict gain: argument_hints_v2, schema_literal_targets_v5.
- `regressions`: Regression cases: contracted:transfer_repeat_audit_card_email_decoy, contracted:transfer_repeat_consent_alert_toggle_decoy.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | mlx_gemma4_e2b_reasoner_only | 6 | 0.3333333333333333 | 0.0 | -0.3333333333333333 | 0.3333333333333333 | 0.0 | -0.3333333333333333 |
| role_catalog_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | 6 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 6 | 0.3333333333333333 | 0.8333333333333334 | 0.5 | 0.3333333333333333 | 1.0 | 0.6666666666666667 |
| schema_field_hints_v4 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | 6 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 |
| schema_literal_targets_v5 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets | 6 | 0.3333333333333333 | 0.8333333333333334 | 0.5 | 0.3333333333333333 | 1.0 | 0.6666666666666667 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | transfer_repeat_audit_card_email_decoy | visual_argument_transfer_repeat | exact | argument_mismatch | -1 | -1 | regression |
| contracted | transfer_repeat_consent_alert_toggle_decoy | visual_tool_routing_transfer_repeat | exact | wrong_tool | -1 | -1 | regression |
| contracted | transfer_repeat_latency_chip_person_decoy | visual_argument_transfer_repeat | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| contracted | transfer_repeat_missing_field_old_selection_decoy | visual_tool_routing_transfer_repeat | wrong_tool | argument_mismatch | 0 | 0 | unchanged |
| contracted | transfer_repeat_priority_tag_chart_decoy | visual_argument_transfer_repeat | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| contracted | transfer_repeat_warning_toast_note_decoy | visual_argument_transfer_repeat | no_tool_call | wrong_tool | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_repeat_audit_card_email_decoy | visual_argument_transfer_repeat | exact | exact | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_repeat_consent_alert_toggle_decoy | visual_tool_routing_transfer_repeat | exact | exact | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_repeat_latency_chip_person_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| role_catalog_v1 | transfer_repeat_missing_field_old_selection_decoy | visual_tool_routing_transfer_repeat | wrong_tool | wrong_tool | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_repeat_priority_tag_chart_decoy | visual_argument_transfer_repeat | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_repeat_warning_toast_note_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_repeat_audit_card_email_decoy | visual_argument_transfer_repeat | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | transfer_repeat_consent_alert_toggle_decoy | visual_tool_routing_transfer_repeat | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | transfer_repeat_latency_chip_person_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_repeat_missing_field_old_selection_decoy | visual_tool_routing_transfer_repeat | wrong_tool | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| argument_hints_v2 | transfer_repeat_priority_tag_chart_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_repeat_warning_toast_note_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_field_hints_v4 | transfer_repeat_audit_card_email_decoy | visual_argument_transfer_repeat | exact | exact | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_repeat_consent_alert_toggle_decoy | visual_tool_routing_transfer_repeat | exact | exact | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_repeat_latency_chip_person_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_field_hints_v4 | transfer_repeat_missing_field_old_selection_decoy | visual_tool_routing_transfer_repeat | wrong_tool | wrong_tool | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_repeat_priority_tag_chart_decoy | visual_argument_transfer_repeat | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_repeat_warning_toast_note_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_literal_targets_v5 | transfer_repeat_audit_card_email_decoy | visual_argument_transfer_repeat | exact | exact | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_repeat_consent_alert_toggle_decoy | visual_tool_routing_transfer_repeat | exact | exact | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_repeat_latency_chip_person_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_literal_targets_v5 | transfer_repeat_missing_field_old_selection_decoy | visual_tool_routing_transfer_repeat | wrong_tool | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| schema_literal_targets_v5 | transfer_repeat_priority_tag_chart_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_literal_targets_v5 | transfer_repeat_warning_toast_note_decoy | visual_argument_transfer_repeat | no_tool_call | exact | 1 | 1 | strict_gain |
