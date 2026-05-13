# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard`
- Baseline exact rate: `0.3`
- Candidate exact rate: `0.4`
- Delta exact rate: `0.10000000000000003`
- Baseline executable rate: `0.4`
- Candidate executable rate: `0.5`
- Delta executable rate: `0.09999999999999998`
- Baseline executor-equivalence rate: `0.4`
- Candidate executor-equivalence rate: `0.5`
- Delta executor-equivalence rate: `0.09999999999999998`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h2v_exception_notice_clean_control | h2v_clean_negation_control | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2v_metric_panel_quoted_not_label_note | h2v_quoted_negation_context | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2v_not_applicable_chip_genuine_value | h2v_genuine_negated_target | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2v_not_approved_toggle_genuine_value | h2v_genuine_negated_target | False | False | True | True | True | True | 1 | 1 | 0 | executable_paraphrase |
| h2v_not_ready_badge_genuine_value | h2v_genuine_negated_target | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2v_owner_field_do_not_use_memo | h2v_instructional_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2v_review_tile_stale_caption_old_not_tile | h2v_stale_example_negation_context | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2v_risk_lane_stale_example_not_lane | h2v_stale_example_negation_context | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2v_status_panel_do_not_use_note | h2v_instructional_negation_context | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2v_summary_tile_quoted_not_label_caption | h2v_quoted_negation_context | False | False | False | False | False | False | 0 | 0 | 0 | no_tool_call |
