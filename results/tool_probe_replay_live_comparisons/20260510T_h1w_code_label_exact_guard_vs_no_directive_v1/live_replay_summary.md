# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard`
- Baseline exact rate: `0.0`
- Candidate exact rate: `0.75`
- Delta exact rate: `0.75`
- Baseline executable rate: `0.0`
- Candidate executable rate: `0.75`
- Delta executable rate: `0.75`
- Baseline executor-equivalence rate: `0.0`
- Candidate executor-equivalence rate: `0.75`
- Delta executor-equivalence rate: `0.75`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1w_assignee_field_archive_summary_decoy | h1w_stale_field_routing | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1w_error_banner_no_call_history_decoy | h1w_activation_no_call | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1w_mode_toggle_settings_note_decoy | h1w_nonstandard_component_class | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1w_owner_field_memo_stale_selection_decoy | h1w_stale_field_routing | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1w_result_badge_comment_value_decoy | h1w_surface_component_value | False | False | False | False | False | False | 0 | 1 | 1 | argument_mismatch |
| h1w_state_tag_audit_log_value_decoy | h1w_nonstandard_component_class | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1w_status_pill_summary_value_decoy | h1w_surface_component_value | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h1w_warning_tile_no_call_note_decoy | h1w_activation_no_call | False | True | False | True | False | True | 0 | 1 | 1 | exact |
