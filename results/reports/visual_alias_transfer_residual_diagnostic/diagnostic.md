# Visual Live Alias Transfer Residual Diagnostic

Generated: `2026-05-10T16:13:08.847777+00:00`

## Findings

- `strict_upper_bound`: hybrid_label_guard_v8 is the strict upper bound at 0.875.
- `executor_equivalence_set`: Executor-equivalent full-success rows: none.
- `executor_without_strict`: Rows with executor gain without strict gain: argument_hints_v2, no_call_control_rescue_v10, oblique_code_guard_v7.
- `regressions`: Regression cases: contracted:residual_badge_q14_notice_decoy, contracted:residual_chip_n31_owner_note_decoy, contracted:residual_phase_pill_ticket_decoy.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | mlx_gemma4_e2b_reasoner_only | 8 | 0.5 | 0.25 | -0.25 | 0.5 | 0.25 | -0.25 |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 8 | 0.5 | 0.625 | 0.125 | 0.5 | 0.875 | 0.375 |
| oblique_code_hints_v6 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints | 8 | 0.5 | 0.75 | 0.25 | 0.5 | 0.75 | 0.25 |
| oblique_code_guard_v7 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard | 8 | 0.5 | 0.75 | 0.25 | 0.5 | 0.875 | 0.375 |
| hybrid_label_guard_v8 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard | 8 | 0.5 | 0.875 | 0.375 | 0.5 | 0.875 | 0.375 |
| no_call_control_rescue_v10 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue | 8 | 0.5 | 0.5 | 0.0 | 0.5 | 0.75 | 0.25 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | residual_alert_h73_toggle_decoy | visual_tool_routing_transfer_residual | argument_mismatch | wrong_tool | 0 | 0 | unchanged |
| contracted | residual_badge_q14_notice_decoy | visual_argument_transfer_residual_code | exact | wrong_tool | -1 | -1 | regression |
| contracted | residual_chip_n31_owner_note_decoy | visual_argument_transfer_residual_code | exact | wrong_tool | -1 | -1 | regression |
| contracted | residual_chip_v82_chart_decoy | visual_argument_transfer_residual_code | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| contracted | residual_field_m20_stale_selection_decoy | visual_tool_routing_transfer_residual | no_tool_call | exact | 1 | 1 | strict_gain |
| contracted | residual_notice_tile_email_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| contracted | residual_phase_pill_ticket_decoy | visual_argument_transfer_residual_noncode | exact | wrong_tool | -1 | -1 | regression |
| contracted | residual_state_pill_note_decoy | visual_argument_transfer_residual_noncode | no_tool_call | wrong_tool | 0 | 0 | unchanged |
| argument_hints_v2 | residual_alert_h73_toggle_decoy | visual_tool_routing_transfer_residual | argument_mismatch | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | residual_badge_q14_notice_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | residual_chip_n31_owner_note_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | residual_chip_v82_chart_decoy | visual_argument_transfer_residual_code | argument_mismatch | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| argument_hints_v2 | residual_field_m20_stale_selection_decoy | visual_tool_routing_transfer_residual | no_tool_call | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| argument_hints_v2 | residual_notice_tile_email_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | residual_phase_pill_ticket_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | residual_state_pill_note_decoy | visual_argument_transfer_residual_noncode | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| oblique_code_hints_v6 | residual_alert_h73_toggle_decoy | visual_tool_routing_transfer_residual | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_hints_v6 | residual_badge_q14_notice_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| oblique_code_hints_v6 | residual_chip_n31_owner_note_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| oblique_code_hints_v6 | residual_chip_v82_chart_decoy | visual_argument_transfer_residual_code | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_hints_v6 | residual_field_m20_stale_selection_decoy | visual_tool_routing_transfer_residual | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| oblique_code_hints_v6 | residual_notice_tile_email_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| oblique_code_hints_v6 | residual_phase_pill_ticket_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| oblique_code_hints_v6 | residual_state_pill_note_decoy | visual_argument_transfer_residual_noncode | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| oblique_code_guard_v7 | residual_alert_h73_toggle_decoy | visual_tool_routing_transfer_residual | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | residual_badge_q14_notice_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | residual_chip_n31_owner_note_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | residual_chip_v82_chart_decoy | visual_argument_transfer_residual_code | argument_mismatch | exact | 1 | 1 | strict_gain |
| oblique_code_guard_v7 | residual_field_m20_stale_selection_decoy | visual_tool_routing_transfer_residual | no_tool_call | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| oblique_code_guard_v7 | residual_notice_tile_email_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | residual_phase_pill_ticket_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| oblique_code_guard_v7 | residual_state_pill_note_decoy | visual_argument_transfer_residual_noncode | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | residual_alert_h73_toggle_decoy | visual_tool_routing_transfer_residual | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | residual_badge_q14_notice_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | residual_chip_n31_owner_note_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | residual_chip_v82_chart_decoy | visual_argument_transfer_residual_code | argument_mismatch | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | residual_field_m20_stale_selection_decoy | visual_tool_routing_transfer_residual | no_tool_call | exact | 1 | 1 | strict_gain |
| hybrid_label_guard_v8 | residual_notice_tile_email_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | residual_phase_pill_ticket_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| hybrid_label_guard_v8 | residual_state_pill_note_decoy | visual_argument_transfer_residual_noncode | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | residual_alert_h73_toggle_decoy | visual_tool_routing_transfer_residual | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | residual_badge_q14_notice_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | residual_chip_n31_owner_note_decoy | visual_argument_transfer_residual_code | exact | exact | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | residual_chip_v82_chart_decoy | visual_argument_transfer_residual_code | argument_mismatch | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| no_call_control_rescue_v10 | residual_field_m20_stale_selection_decoy | visual_tool_routing_transfer_residual | no_tool_call | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| no_call_control_rescue_v10 | residual_notice_tile_email_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | residual_phase_pill_ticket_decoy | visual_argument_transfer_residual_noncode | exact | exact | 0 | 0 | unchanged |
| no_call_control_rescue_v10 | residual_state_pill_note_decoy | visual_argument_transfer_residual_noncode | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
