# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating`
- Baseline exact rate: `0.3`
- Candidate exact rate: `1.0`
- Delta exact rate: `0.7`
- Baseline executable rate: `0.3`
- Candidate executable rate: `1.0`
- Delta executable rate: `0.7`
- Baseline executor-equivalence rate: `0.3`
- Candidate executor-equivalence rate: `1.0`
- Delta executor-equivalence rate: `0.7`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h2s_approval_panel_pending_negated_badge_note | h2s_negated_decoy_guard | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2s_delivery_field_paused_toggle_switch_decoys | h2s_contextual_alias_decoy_overlap | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2s_result_panel_waiting_stale_selection_hint | h2s_stale_surface_alias | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2s_review_tile_waiting_chip_note_decoys | h2s_surface_alias_same_value_decoy | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2s_reviewer_field_malik_old_owner_decoy | h2s_value_bearing_stale_decoy | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2s_severity_pill_critical_archived_badge_decoy | h2s_value_bearing_stale_decoy | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2s_signal_panel_green_tag_marker_decoys | h2s_surface_alias_same_value_decoy | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2s_status_badge_live_clean_control | h2s_clean_route_control | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2s_status_tile_ready_stale_selection_hint | h2s_stale_surface_alias | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2s_timeout_exception_notice_banner_log_decoys | h2s_contextual_alias_decoy_overlap | False | True | False | True | False | True | 1 | 1 | 0 | exact |
