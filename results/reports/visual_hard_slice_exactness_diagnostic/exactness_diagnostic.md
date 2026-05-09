# Visual Hard-Slice Exactness Diagnostic

- packet_run_id: `20260509T_visual_hard_slice_executor_equivalence_v1`
- system_count: `2`
- case_row_count: `16`
- exactness_gap_count: `5`

## System Summary

| System | Exact | Executable | Non-Exact Executor Success | Label-Artifact Candidates | True Harness Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| catalog schema fields | 6 / 8 | 8 / 8 | 2 | 2 | 0 |
| catalog schema target literals | 5 / 8 | 7 / 8 | 2 | 2 | 1 |

## Exactness Gaps

| System | Case | Failure | Expected Target | Actual Target | Diagnosis | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| catalog schema fields | visual_metric_panel_vs_table_selector | executable_paraphrase | hard-metric-1001 | hard-metric-1001 | executable_selector_alias | benchmark_label_artifact_candidate |
| catalog schema fields | visual_callout_warning_with_user_decoy | executable_paraphrase | hard-callout-decoy-1102 | hard-callout-decoy-1102 | executable_selector_alias | benchmark_label_artifact_candidate |
| catalog schema target literals | visual_form_error_with_prior_selection_decoy | wrong_tool | hard-form-decoy-602 |  | wrong_tool_executor_failure | true_harness_failure |
| catalog schema target literals | visual_metric_panel_vs_table_selector | executable_paraphrase | hard-metric-1001 | hard-metric-1001 | executable_selector_alias | benchmark_label_artifact_candidate |
| catalog schema target literals | visual_callout_warning_with_user_decoy | executable_paraphrase | hard-callout-decoy-1102 | hard-callout-decoy-1102 | executable_selector_alias | benchmark_label_artifact_candidate |
