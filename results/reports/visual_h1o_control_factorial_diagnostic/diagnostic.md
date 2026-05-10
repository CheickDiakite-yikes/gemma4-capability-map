# Visual Live H1O Control Factorial Diagnostic

Generated: `2026-05-10T16:39:43.895297+00:00`

## Findings

- `strict_upper_bound`: argument_hints_v2, component_value_guard_v9 are the strict upper bound at 0.75.
- `executor_equivalence_set`: Executor-equivalent full-success rows: none.
- `executor_without_strict`: Rows with executor gain without strict gain: hybrid_label_guard_v8, oblique_code_guard_v7.
- `regressions`: Regression cases: no_call_control_rescue_v10:h1o_activation_error_banner_previous_region_decoy.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 12 | 0.4166666666666667 | 0.75 | 0.3333333333333333 | 0.5 | 0.8333333333333334 | 0.33333333333333337 |
| hybrid_label_guard_v8 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard | 12 | 0.4166666666666667 | 0.6666666666666666 | 0.24999999999999994 | 0.5 | 0.8333333333333334 | 0.33333333333333337 |
| no_call_control_rescue_v10 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue | 12 | 0.4166666666666667 | 0.5833333333333334 | 0.16666666666666669 | 0.5 | 0.6666666666666666 | 0.16666666666666663 |
| oblique_code_guard_v7 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard | 12 | 0.4166666666666667 | 0.6666666666666666 | 0.24999999999999994 | 0.5 | 0.75 | 0.25 |
| component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | 12 | 0.4166666666666667 | 0.75 | 0.3333333333333333 | 0.5 | 0.8333333333333334 | 0.33333333333333337 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| argument_hints_v2 | h1o_activation_error_banner_previous_region_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | h1o_activation_owner_field_stale_selection_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | h1o_activation_status_badge_email_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | 0 | 0 | unchanged |
| argument_hints_v2 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1o_code_field_u17_old_selection_decoy | h1o_code_negation_preservation | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1o_code_node_j44_table_decoy | h1o_code_negation_preservation | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | h1o_component_phase_tile_value_decoy | h1o_component_value_boundary | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | h1o_component_priority_chip_value_decoy | h1o_component_value_boundary | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1o_component_result_badge_value_decoy | h1o_component_value_boundary | argument_mismatch | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | h1o_component_state_pill_value_decoy | h1o_component_value_boundary | argument_mismatch | no_tool_call | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1o_activation_error_banner_previous_region_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1o_activation_owner_field_stale_selection_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1o_activation_status_badge_email_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | 1 | 0 | strict_gain |
| hybrid_label_guard_v8 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | no_tool_call | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| hybrid_label_guard_v8 | h1o_code_field_u17_old_selection_decoy | h1o_code_negation_preservation | no_tool_call | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1o_code_node_j44_table_decoy | h1o_code_negation_preservation | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1o_component_phase_tile_value_decoy | h1o_component_value_boundary | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | h1o_component_priority_chip_value_decoy | h1o_component_value_boundary | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1o_component_result_badge_value_decoy | h1o_component_value_boundary | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | h1o_component_state_pill_value_decoy | h1o_component_value_boundary | argument_mismatch | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| no_call_control_rescue_v10 | h1o_activation_error_banner_previous_region_decoy | h1o_activation_no_call | exact | no_tool_call | -1 | -1 | regression |
| no_call_control_rescue_v10 | h1o_activation_owner_field_stale_selection_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1o_activation_status_badge_email_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | no_tool_call | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1o_code_field_u17_old_selection_decoy | h1o_code_negation_preservation | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1o_code_node_j44_table_decoy | h1o_code_negation_preservation | exact | exact | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1o_component_phase_tile_value_decoy | h1o_component_value_boundary | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | h1o_component_priority_chip_value_decoy | h1o_component_value_boundary | no_tool_call | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1o_component_result_badge_value_decoy | h1o_component_value_boundary | argument_mismatch | exact | 1 | 1 | strict_gain |
| no_call_control_rescue_v10 | h1o_component_state_pill_value_decoy | h1o_component_value_boundary | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| oblique_code_guard_v7 | h1o_activation_error_banner_previous_region_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | h1o_activation_owner_field_stale_selection_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | h1o_activation_status_badge_email_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | 1 | 0 | strict_gain |
| oblique_code_guard_v7 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | no_tool_call | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| oblique_code_guard_v7 | h1o_code_field_u17_old_selection_decoy | h1o_code_negation_preservation | no_tool_call | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | h1o_code_node_j44_table_decoy | h1o_code_negation_preservation | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | h1o_component_phase_tile_value_decoy | h1o_component_value_boundary | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | h1o_component_priority_chip_value_decoy | h1o_component_value_boundary | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| oblique_code_guard_v7 | h1o_component_result_badge_value_decoy | h1o_component_value_boundary | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| oblique_code_guard_v7 | h1o_component_state_pill_value_decoy | h1o_component_value_boundary | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| component_value_guard_v9 | h1o_activation_error_banner_previous_region_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| component_value_guard_v9 | h1o_activation_owner_field_stale_selection_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| component_value_guard_v9 | h1o_activation_status_badge_email_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| component_value_guard_v9 | h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | exact | exact | 0 | 0 | unchanged |
| component_value_guard_v9 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | 0 | 0 | unchanged |
| component_value_guard_v9 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | no_tool_call | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1o_code_field_u17_old_selection_decoy | h1o_code_negation_preservation | no_tool_call | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1o_code_node_j44_table_decoy | h1o_code_negation_preservation | exact | exact | 0 | 0 | unchanged |
| component_value_guard_v9 | h1o_component_phase_tile_value_decoy | h1o_component_value_boundary | argument_mismatch | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1o_component_priority_chip_value_decoy | h1o_component_value_boundary | no_tool_call | exact | 1 | 1 | strict_gain |
| component_value_guard_v9 | h1o_component_result_badge_value_decoy | h1o_component_value_boundary | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| component_value_guard_v9 | h1o_component_state_pill_value_decoy | h1o_component_value_boundary | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
