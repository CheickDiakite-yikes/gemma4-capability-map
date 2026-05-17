# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_h2u_no_controller_fallback`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard`
- Baseline exact rate: `0.375`
- Candidate exact rate: `0.375`
- Delta exact rate: `0.0`
- Baseline executable rate: `0.5`
- Candidate executable rate: `0.5`
- Delta executable rate: `0.0`
- Baseline executor-equivalence rate: `0.5`
- Candidate executor-equivalence rate: `0.5`
- Delta executor-equivalence rate: `0.0`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h2x_not_applicable_reason_chip_value_before_component | h2x_genuine_negated_target_value | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2x_not_approved_approval_toggle_value_before_component | h2x_genuine_negated_target_value | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2x_not_blocked_result_tile_value_before_component | h2x_genuine_negated_target_value | False | False | True | True | True | True | 1 | 1 | 0 | executable_paraphrase |
| h2x_not_ready_status_badge_value_before_component | h2x_genuine_negated_target_value | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2x_owner_field_do_not_use_memo | h2x_instructional_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2x_risk_lane_stale_selection_not_lane | h2x_stale_selection_negation_context | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h2x_status_badge_quoted_not_badge_note | h2x_quoted_stale_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2x_summary_tile_caption_says_not_tile | h2x_quoted_stale_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
