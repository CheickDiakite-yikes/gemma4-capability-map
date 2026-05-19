# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard_visual_semantic_target_preservation`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_h2z_negated_component_target_preservation`
- Baseline exact rate: `0.75`
- Candidate exact rate: `0.8125`
- Delta exact rate: `0.0625`
- Baseline executable rate: `0.75`
- Candidate executable rate: `0.8125`
- Delta executable rate: `0.0625`
- Baseline executor-equivalence rate: `0.75`
- Candidate executor-equivalence rate: `0.8125`
- Delta executor-equivalence rate: `0.0625`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h2y_action_banner_quoted_not_banner_note | h2y_quoted_stale_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_approval_field_stale_selection_not_field | h2y_stale_selection_negation_context | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h2y_decision_panel_caption_says_not_panel | h2y_quoted_stale_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_due_date_field_do_not_use_note | h2y_instructional_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_escalation_lane_stale_selection_not_lane | h2y_stale_selection_negation_context | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h2y_exception_panel_stale_selection_not_panel | h2y_stale_selection_negation_context | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h2y_not_active_alert_banner_value_before_component | h2y_genuine_negated_target_value | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2y_not_available_owner_field_value_before_component | h2y_genuine_negated_target_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_not_escalated_risk_chip_value_before_component | h2y_genuine_negated_target_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_not_replied_status_pill_value_before_component | h2y_genuine_negated_target_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_not_required_approval_marker_value_before_component | h2y_genuine_negated_target_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_not_sent_delivery_tag_value_before_component | h2y_genuine_negated_target_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_not_started_phase_tile_value_before_component | h2y_genuine_negated_target_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_resolution_marker_old_quote_not_marker | h2y_quoted_stale_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_reviewer_tag_do_not_use_memo | h2y_instructional_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2y_vendor_chip_do_not_use_table | h2y_instructional_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
