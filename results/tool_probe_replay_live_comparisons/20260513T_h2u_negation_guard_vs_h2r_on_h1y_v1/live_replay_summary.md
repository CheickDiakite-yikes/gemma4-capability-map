# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard`
- Baseline exact rate: `1.0`
- Candidate exact rate: `1.0`
- Delta exact rate: `0.0`
- Baseline executable rate: `1.0`
- Candidate executable rate: `1.0`
- Delta executable rate: `0.0`
- Baseline executor-equivalence rate: `1.0`
- Candidate executor-equivalence rate: `1.0`
- Delta executor-equivalence rate: `0.0`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1y_alert_s92_negated_toggle_decoy | h1y_route_code_label | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_badge_c08_table_value_decoy | h1y_route_code_label | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_escalation_contact_field_saved_summary_decoy | h1y_route_stale_field | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_lifecycle_state_tag_audit_value_decoy | h1y_route_nonstandard_class | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_operation_mode_toggle_note_value_decoy | h1y_route_nonstandard_class | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_responsible_party_field_old_owner_memo_decoy | h1y_route_stale_field | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_result_badge_comment_value_holdout | h1y_preserve_surface_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_review_owner_field_previous_table_decoy | h1y_route_stale_field | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_status_pill_summary_value_holdout | h1y_preserve_surface_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1y_warning_tile_note_activation_decoy | h1y_activation_no_call | True | True | True | True | True | True | 1 | 1 | 0 | exact |
