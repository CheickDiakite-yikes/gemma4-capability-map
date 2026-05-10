# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard`
- Baseline exact rate: `0.0`
- Candidate exact rate: `0.75`
- Delta exact rate: `0.75`
- Baseline executable rate: `0.0`
- Candidate executable rate: `0.8333333333333334`
- Delta executable rate: `0.8333333333333334`
- Baseline executor-equivalence rate: `0.0`
- Candidate executor-equivalence rate: `0.8333333333333334`
- Delta executor-equivalence rate: `0.8333333333333334`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1p_compact_queue_badge_table_value_decoy | h1p_component_value_compact | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1p_compact_stage_chip_email_value_decoy | h1p_component_value_compact | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h1p_compact_status_pill_summary_value_decoy | h1p_component_value_compact | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1p_stale_phase_tile_archive_decoy | h1p_component_value_stale_selection | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1p_stale_priority_chip_old_selection_decoy | h1p_component_value_stale_selection | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1p_stale_risk_badge_old_selection_decoy | h1p_component_value_stale_selection | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1p_stale_severity_pill_previous_region_decoy | h1p_component_value_stale_selection | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1p_surface_lane_tile_board_value_decoy | h1p_component_value_surface | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | False | False | False | False | False | False | 0 | 1 | 1 | argument_mismatch |
| h1p_surface_owner_field_note_value_decoy | h1p_component_value_surface | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1p_surface_result_badge_comment_value_decoy | h1p_component_value_surface | False | False | False | True | False | True | 1 | 1 | 0 | executable_paraphrase |
