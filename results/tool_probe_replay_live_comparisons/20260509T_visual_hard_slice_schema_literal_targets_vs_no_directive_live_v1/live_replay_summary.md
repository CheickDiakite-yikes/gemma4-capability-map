# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets`
- Baseline exact rate: `0.0`
- Candidate exact rate: `0.0`
- Delta exact rate: `0.0`
- Baseline executable rate: `0.0`
- Candidate executable rate: `0.5`
- Delta executable rate: `0.5`
- Baseline executor-equivalence rate: `0.0`
- Candidate executor-equivalence rate: `0.5`
- Delta executor-equivalence rate: `0.5`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| visual_form_error_with_prior_selection_decoy | visual_tool_routing | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| visual_metric_panel_vs_table_selector | visual_argument_copying | False | False | False | True | False | True | 1 | 1 | 0 | executable_paraphrase |
