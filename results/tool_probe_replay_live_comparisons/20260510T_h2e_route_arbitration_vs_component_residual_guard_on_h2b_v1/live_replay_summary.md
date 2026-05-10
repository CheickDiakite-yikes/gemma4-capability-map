# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate`
- Baseline exact rate: `0.8`
- Candidate exact rate: `1.0`
- Delta exact rate: `0.19999999999999996`
- Baseline executable rate: `0.8`
- Candidate executable rate: `1.0`
- Delta executable rate: `0.19999999999999996`
- Baseline executor-equivalence rate: `0.8`
- Candidate executor-equivalence rate: `1.0`
- Delta executor-equivalence rate: `0.19999999999999996`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | True | True | True | True | True | True | 1 | 1 | 0 | exact |
