# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only`
- Baseline exact rate: `0.25`
- Candidate exact rate: `0.875`
- Delta exact rate: `0.625`
- Baseline executable rate: `0.625`
- Candidate executable rate: `1.0`
- Delta executable rate: `0.375`
- Baseline executor-equivalence rate: `0.625`
- Candidate executor-equivalence rate: `1.0`
- Delta executor-equivalence rate: `0.375`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stress_callout_warning_person_table_decoy | visual_argument_copying_stress | False | False | False | True | False | True | 0 | 1 | 1 | executable_paraphrase |
| stress_callout_warning_risk_note_decoy | visual_argument_copying_stress | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| stress_form_error_stale_selection_status_decoy | visual_tool_routing_stress | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| stress_form_error_stale_selection_warning_decoy | visual_tool_routing_stress | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| stress_metric_panel_status_banner_decoy | visual_argument_copying_stress | False | True | True | True | True | True | 1 | 1 | 0 | exact |
| stress_metric_panel_summary_card_decoy | visual_argument_copying_stress | False | True | True | True | True | True | 1 | 1 | 0 | exact |
| stress_metric_panel_with_chart_table_decoys | visual_argument_copying_stress | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| stress_metric_panel_with_kpi_copy_decoy | visual_argument_copying_stress | False | True | True | True | True | True | 1 | 1 | 0 | exact |
