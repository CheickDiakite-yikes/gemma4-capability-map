# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_routed_residual_guard`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate`
- Baseline exact rate: `0.5`
- Candidate exact rate: `0.8`
- Delta exact rate: `0.30000000000000004`
- Baseline executable rate: `0.5`
- Candidate executable rate: `0.8`
- Delta executable rate: `0.30000000000000004`
- Baseline executor-equivalence rate: `0.5`
- Candidate executor-equivalence rate: `0.8`
- Delta executor-equivalence rate: `0.30000000000000004`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1y_alert_s92_negated_toggle_decoy | h1y_route_code_label | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h1y_badge_c08_table_value_decoy | h1y_route_code_label | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_escalation_contact_field_saved_summary_decoy | h1y_route_stale_field | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_lifecycle_state_tag_audit_value_decoy | h1y_route_nonstandard_class | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h1y_operation_mode_toggle_note_value_decoy | h1y_route_nonstandard_class | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_responsible_party_field_old_owner_memo_decoy | h1y_route_stale_field | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1y_result_badge_comment_value_holdout | h1y_preserve_surface_value | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1y_review_owner_field_previous_table_decoy | h1y_route_stale_field | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_status_pill_summary_value_holdout | h1y_preserve_surface_value | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1y_warning_tile_note_activation_decoy | h1y_activation_no_call | True | True | True | True | True | True | 1 | 1 | 0 | exact |
