# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing`
- Baseline exact rate: `0.125`
- Candidate exact rate: `0.375`
- Delta exact rate: `0.25`
- Baseline executable rate: `0.25`
- Candidate executable rate: `0.375`
- Delta executable rate: `0.125`
- Baseline executor-equivalence rate: `0.25`
- Candidate executor-equivalence rate: `0.375`
- Delta executor-equivalence rate: `0.125`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h2q_archive_panel_error_notice_banner_decoy | h2q_contextual_alias_decoy_overlap | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2q_mode_field_manual_switch_decoy | h2q_contextual_alias_decoy_overlap | True | False | True | False | True | False | 1 | 1 | 0 | argument_mismatch |
| h2q_owner_field_amina_archived_owner_decoy | h2q_value_bearing_stale_decoy | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2q_priority_badge_critical_stale_status_decoy | h2q_value_bearing_stale_decoy | False | True | True | True | True | True | 1 | 1 | 0 | exact |
| h2q_result_tile_blocked_value_badge_decoy | h2q_surface_alias_value_decoy | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h2q_result_tile_stale_selection_hint | h2q_stale_surface_alias | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h2q_state_panel_closed_value_tag_decoy | h2q_surface_alias_value_decoy | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h2q_state_panel_stale_selection_hint | h2q_stale_surface_alias | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
