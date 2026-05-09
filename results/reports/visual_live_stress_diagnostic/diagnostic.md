# Visual Live Stress Diagnostic

Generated: `2026-05-09T23:11:38.039095+00:00`

## Findings

- `strict_upper_bound`: contracted is the strict upper bound at 1.0.
- `executor_equivalence_set`: Executor-equivalent full-success rows: contracted, schema_field_hints_v4, schema_literal_targets_v5.
- `executor_without_strict`: Rows with executor gain without strict gain: schema_field_hints_v4, schema_literal_targets_v5.
- `regressions`: Regression cases: role_catalog_v1:stress_form_error_stale_selection_warning_decoy.

## Summary

| label | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | mlx_gemma4_e2b_reasoner_only | 4 | 0.5 | 1.0 | 0.5 | 0.75 | 1.0 | 0.25 |
| role_catalog_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | 4 | 0.5 | 0.25 | -0.25 | 0.75 | 0.5 | -0.25 |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 4 | 0.5 | 0.5 | 0.0 | 0.75 | 0.75 | 0.0 |
| schema_field_hints_v4 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | 4 | 0.5 | 0.5 | 0.0 | 0.75 | 1.0 | 0.25 |
| schema_literal_targets_v5 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets | 4 | 0.5 | 0.5 | 0.0 | 0.75 | 1.0 | 0.25 |

## Case Transitions

| label | case_id | family | baseline_failure_mode | candidate_failure_mode | delta_exact_match | delta_executor_equivalence_match | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | stress_form_error_stale_selection_status_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| contracted | stress_form_error_stale_selection_warning_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| contracted | stress_metric_panel_with_chart_table_decoys | visual_argument_copying_stress | argument_mismatch | exact | 1 | 1 | strict_gain |
| contracted | stress_metric_panel_with_kpi_copy_decoy | visual_argument_copying_stress | executable_paraphrase | exact | 1 | 0 | strict_gain |
| role_catalog_v1 | stress_form_error_stale_selection_status_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| role_catalog_v1 | stress_form_error_stale_selection_warning_decoy | visual_tool_routing_stress | exact | no_tool_call | -1 | -1 | regression |
| role_catalog_v1 | stress_metric_panel_with_chart_table_decoys | visual_argument_copying_stress | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| role_catalog_v1 | stress_metric_panel_with_kpi_copy_decoy | visual_argument_copying_stress | executable_paraphrase | executable_paraphrase | 0 | 0 | unchanged |
| argument_hints_v2 | stress_form_error_stale_selection_status_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | stress_form_error_stale_selection_warning_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| argument_hints_v2 | stress_metric_panel_with_chart_table_decoys | visual_argument_copying_stress | argument_mismatch | argument_mismatch | 0 | 0 | unchanged |
| argument_hints_v2 | stress_metric_panel_with_kpi_copy_decoy | visual_argument_copying_stress | executable_paraphrase | executable_paraphrase | 0 | 0 | unchanged |
| schema_field_hints_v4 | stress_form_error_stale_selection_status_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| schema_field_hints_v4 | stress_form_error_stale_selection_warning_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| schema_field_hints_v4 | stress_metric_panel_with_chart_table_decoys | visual_argument_copying_stress | argument_mismatch | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| schema_field_hints_v4 | stress_metric_panel_with_kpi_copy_decoy | visual_argument_copying_stress | executable_paraphrase | executable_paraphrase | 0 | 0 | unchanged |
| schema_literal_targets_v5 | stress_form_error_stale_selection_status_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| schema_literal_targets_v5 | stress_form_error_stale_selection_warning_decoy | visual_tool_routing_stress | exact | exact | 0 | 0 | unchanged |
| schema_literal_targets_v5 | stress_metric_panel_with_chart_table_decoys | visual_argument_copying_stress | argument_mismatch | executable_paraphrase | 0 | 1 | executor_gain_without_strict |
| schema_literal_targets_v5 | stress_metric_panel_with_kpi_copy_decoy | visual_argument_copying_stress | executable_paraphrase | executable_paraphrase | 0 | 0 | unchanged |
