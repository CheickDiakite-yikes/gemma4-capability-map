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
| h2k_alert_t47_archived_alert_s92_decoy | h2k_code_label_overlap | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2k_badge_c18_negated_badge_c08_decoy | h2k_code_label_overlap | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2k_error_banner_archived_error_notice_decoy | h2k_transfer_regression_guard | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2k_mode_field_before_reading_mode_switch_decoy | h2k_before_reading_decoy | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2k_mode_toggle_negated_consent_toggle_decoy | h2k_negated_same_component_decoy | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2k_priority_badge_negated_status_badge_decoy | h2k_negated_same_component_decoy | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2k_result_badge_negated_result_tile_decoy | h2k_negated_same_component_decoy | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2k_state_tag_before_reading_state_marker_decoy | h2k_before_reading_decoy | True | True | True | True | True | True | 1 | 1 | 0 | exact |
