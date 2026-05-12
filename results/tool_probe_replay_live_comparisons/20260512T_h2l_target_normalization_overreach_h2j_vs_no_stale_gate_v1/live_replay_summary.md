# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_target_query_normalization_no_stale_selection_gate`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization`
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
| h2l_error_notice_alias_is_target | h2l_alias_is_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2l_mode_field_short_label_regression_guard | h2l_h2k_regression_guard | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2l_mode_toggle_manual_value_is_target | h2l_value_bearing_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2l_priority_badge_critical_value_is_target | h2l_value_bearing_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2l_result_badge_blocked_value_is_target | h2l_value_bearing_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2l_result_tile_alias_is_target | h2l_alias_is_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2l_state_tag_closed_value_is_target | h2l_value_bearing_target | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2l_status_badge_short_label_regression_guard | h2l_h2k_regression_guard | True | True | True | True | True | True | 1 | 1 | 0 | exact |
