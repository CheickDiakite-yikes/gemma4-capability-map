# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating`
- Baseline exact rate: `0.875`
- Candidate exact rate: `1.0`
- Delta exact rate: `0.125`
- Baseline executable rate: `0.875`
- Candidate executable rate: `1.0`
- Delta executable rate: `0.125`
- Baseline executor-equivalence rate: `0.875`
- Candidate executor-equivalence rate: `1.0`
- Delta executor-equivalence rate: `0.125`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h2m_error_notice_contextual_alias | h2m_contextual_alias_is_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2m_mode_field_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2m_mode_toggle_manual_contextual_value | h2m_less_direct_value_bearing_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2m_priority_badge_critical_contextual_value | h2m_less_direct_value_bearing_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2m_result_badge_blocked_contextual_value | h2m_less_direct_value_bearing_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2m_result_tile_contextual_alias | h2m_contextual_alias_is_target | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2m_state_tag_closed_contextual_value | h2m_less_direct_value_bearing_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2m_status_badge_contextual_regression_guard | h2m_h2k_regression_guard_less_direct | True | True | True | True | True | True | 1 | 1 | 0 | exact |
