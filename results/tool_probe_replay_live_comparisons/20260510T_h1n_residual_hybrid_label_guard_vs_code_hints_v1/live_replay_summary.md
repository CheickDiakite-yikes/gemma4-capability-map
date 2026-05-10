# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard`
- Baseline exact rate: `0.75`
- Candidate exact rate: `0.875`
- Delta exact rate: `0.125`
- Baseline executable rate: `0.75`
- Candidate executable rate: `0.875`
- Delta executable rate: `0.125`
- Baseline executor-equivalence rate: `0.75`
- Candidate executor-equivalence rate: `0.875`
- Delta executor-equivalence rate: `0.125`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| residual_alert_h73_toggle_decoy | visual_tool_routing_transfer_residual | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| residual_badge_q14_notice_decoy | visual_argument_transfer_residual_code | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| residual_chip_n31_owner_note_decoy | visual_argument_transfer_residual_code | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| residual_chip_v82_chart_decoy | visual_argument_transfer_residual_code | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| residual_field_m20_stale_selection_decoy | visual_tool_routing_transfer_residual | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| residual_notice_tile_email_decoy | visual_argument_transfer_residual_noncode | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| residual_phase_pill_ticket_decoy | visual_argument_transfer_residual_noncode | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| residual_state_pill_note_decoy | visual_argument_transfer_residual_noncode | False | False | False | False | False | False | 0 | 1 | 1 | argument_mismatch |
