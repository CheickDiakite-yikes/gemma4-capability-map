# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard`
- Baseline exact rate: `0.5`
- Candidate exact rate: `1.0`
- Delta exact rate: `0.5`
- Baseline executable rate: `0.5`
- Candidate executable rate: `1.0`
- Delta executable rate: `0.5`
- Baseline executor-equivalence rate: `0.5`
- Candidate executor-equivalence rate: `1.0`
- Delta executor-equivalence rate: `0.5`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1r_alert_s92_toggle_negation_decoy | h1r_code_label_exactness | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1r_assignee_field_previous_selection_summary_decoy | h1r_stale_selection_component_label | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1r_badge_c08_note_decoy | h1r_code_label_exactness | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1r_mode_toggle_note_value_decoy | h1r_nonstandard_component_class | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1r_owner_field_stale_selection_note_decoy | h1r_stale_selection_component_label | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1r_state_tag_log_value_decoy | h1r_nonstandard_component_class | False | True | False | True | False | True | 1 | 1 | 0 | exact |
