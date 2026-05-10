# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_query_contract_visual_stale_selection_gate`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate`
- Baseline exact rate: `0.6`
- Candidate exact rate: `0.9`
- Delta exact rate: `0.30000000000000004`
- Baseline executable rate: `0.7`
- Candidate executable rate: `0.9`
- Delta executable rate: `0.20000000000000007`
- Baseline executor-equivalence rate: `0.7`
- Candidate executor-equivalence rate: `0.9`
- Delta executor-equivalence rate: `0.20000000000000007`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h2f_alert_t47_negated_switch_decoy | h2f_route_code_label | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2f_badge_m31_summary_value_decoy | h2f_route_code_label | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2f_error_notice_history_activation_decoy | h2f_activation_panel_notice | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2f_mode_switch_note_value_decoy | h2f_route_nonstandard_class | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2f_owner_field_previous_memo_decoy | h2f_route_stale_field | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2f_resolution_badge_log_result_decoy | h2f_route_component_class_transfer | False | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2f_result_tile_comment_value_decoy | h2f_route_component_class_transfer | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2f_reviewer_field_saved_summary_decoy | h2f_route_stale_field | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2f_warning_panel_note_activation_decoy | h2f_activation_panel_notice | True | True | True | True | True | True | 1 | 1 | 0 | exact |
