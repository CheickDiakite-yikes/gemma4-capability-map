# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints`
- Baseline exact rate: `0.4166666666666667`
- Candidate exact rate: `0.75`
- Delta exact rate: `0.3333333333333333`
- Baseline executable rate: `0.5`
- Candidate executable rate: `0.8333333333333334`
- Delta executable rate: `0.33333333333333337`
- Baseline executor-equivalence rate: `0.5`
- Candidate executor-equivalence rate: `0.8333333333333334`
- Delta executor-equivalence rate: `0.33333333333333337`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1o_activation_error_banner_previous_region_decoy | h1o_activation_no_call | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1o_activation_owner_field_stale_selection_decoy | h1o_activation_no_call | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1o_activation_status_badge_email_decoy | h1o_activation_no_call | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1o_activation_warning_tile_no_call_decoy | h1o_activation_no_call | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | False | False | True | True | True | True | 1 | 1 | 0 | executable_paraphrase |
| h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1o_code_field_u17_old_selection_decoy | h1o_code_negation_preservation | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1o_code_node_j44_table_decoy | h1o_code_negation_preservation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1o_component_phase_tile_value_decoy | h1o_component_value_boundary | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h1o_component_priority_chip_value_decoy | h1o_component_value_boundary | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1o_component_result_badge_value_decoy | h1o_component_value_boundary | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1o_component_state_pill_value_decoy | h1o_component_value_boundary | False | False | False | False | False | False | 1 | 0 | -1 | no_tool_call |
