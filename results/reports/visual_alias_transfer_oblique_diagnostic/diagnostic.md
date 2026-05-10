# Visual Live Alias Transfer Oblique Diagnostic

Generated: `2026-05-10T01:07:19.357406+00:00`

## Findings

- `strict_upper_bound`: argument_hints_v2 is the strict upper bound at 0.6666666666666666.
- `executor_equivalence_set`: Executor-equivalent full-success rows: .
- `executor_without_strict`: Rows with executor gain without strict gain: .
- `regressions`: Regression cases: none.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | mlx_gemma4_e2b_reasoner_only | 6 | 0.0 | 0.16666666666666666 | 0.16666666666666666 | 0.0 | 0.16666666666666666 | 0.16666666666666666 |
| role_catalog_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | 6 | 0.0 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 0.3333333333333333 | 0.3333333333333333 |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 6 | 0.0 | 0.6666666666666666 | 0.6666666666666666 | 0.0 | 0.6666666666666666 | 0.6666666666666666 |
| schema_field_hints_v4 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | 6 | 0.0 | 0.5 | 0.5 | 0.0 | 0.5 | 0.5 |
| schema_literal_targets_v5 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | transfer_oblique_alert_p55_toggle_decoy | visual_tool_routing_transfer_oblique | argument_mismatch | wrong_tool | 0 | 0 | unchanged |
| contracted | transfer_oblique_badge_m88_chart_decoy | visual_argument_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| contracted | transfer_oblique_cell_r42_notice_decoy | visual_argument_transfer_oblique | argument_mismatch | wrong_tool | 0 | 0 | unchanged |
| contracted | transfer_oblique_chip_z33_person_decoy | visual_argument_transfer_oblique | no_tool_call | wrong_tool | 0 | 0 | unchanged |
| contracted | transfer_oblique_field_e19_old_selection_decoy | visual_tool_routing_transfer_oblique | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| contracted | transfer_oblique_node_q17_table_decoy | visual_argument_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_oblique_alert_p55_toggle_decoy | visual_tool_routing_transfer_oblique | argument_mismatch | no_tool_call | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_oblique_badge_m88_chart_decoy | visual_argument_transfer_oblique | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_oblique_cell_r42_notice_decoy | visual_argument_transfer_oblique | argument_mismatch | no_tool_call | 0 | 0 | unchanged |
| role_catalog_v1 | transfer_oblique_chip_z33_person_decoy | visual_argument_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| role_catalog_v1 | transfer_oblique_field_e19_old_selection_decoy | visual_tool_routing_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| role_catalog_v1 | transfer_oblique_node_q17_table_decoy | visual_argument_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | transfer_oblique_alert_p55_toggle_decoy | visual_tool_routing_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | transfer_oblique_badge_m88_chart_decoy | visual_argument_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_oblique_cell_r42_notice_decoy | visual_argument_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | transfer_oblique_chip_z33_person_decoy | visual_argument_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_oblique_field_e19_old_selection_decoy | visual_tool_routing_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| argument_hints_v2 | transfer_oblique_node_q17_table_decoy | visual_argument_transfer_oblique | argument_mismatch | exact | 1 | 1 | strict_gain |
| schema_field_hints_v4 | transfer_oblique_alert_p55_toggle_decoy | visual_tool_routing_transfer_oblique | argument_mismatch | no_tool_call | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_oblique_badge_m88_chart_decoy | visual_argument_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_field_hints_v4 | transfer_oblique_cell_r42_notice_decoy | visual_argument_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| schema_field_hints_v4 | transfer_oblique_chip_z33_person_decoy | visual_argument_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_field_hints_v4 | transfer_oblique_field_e19_old_selection_decoy | visual_tool_routing_transfer_oblique | no_tool_call | exact | 1 | 1 | strict_gain |
| schema_field_hints_v4 | transfer_oblique_node_q17_table_decoy | visual_argument_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_oblique_alert_p55_toggle_decoy | visual_tool_routing_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_oblique_badge_m88_chart_decoy | visual_argument_transfer_oblique | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_oblique_cell_r42_notice_decoy | visual_argument_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_oblique_chip_z33_person_decoy | visual_argument_transfer_oblique | no_tool_call | argument_mismatch | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_oblique_field_e19_old_selection_decoy | visual_tool_routing_transfer_oblique | no_tool_call | no_tool_call | 0 | 0 | unchanged |
| schema_literal_targets_v5 | transfer_oblique_node_q17_table_decoy | visual_argument_transfer_oblique | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
