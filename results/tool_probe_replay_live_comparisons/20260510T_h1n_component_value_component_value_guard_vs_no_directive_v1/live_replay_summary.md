# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard`
- Baseline exact rate: `0.625`
- Candidate exact rate: `0.5`
- Delta exact rate: `-0.125`
- Baseline executable rate: `0.75`
- Candidate executable rate: `0.5`
- Delta executable rate: `-0.25`
- Baseline executor-equivalence rate: `0.75`
- Candidate executor-equivalence rate: `0.5`
- Delta executor-equivalence rate: `-0.25`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| component_value_owner_field_stale_selection_decoy | visual_tool_routing_component_value | False | False | False | False | False | False | 0 | 1 | 1 | argument_mismatch |
| component_value_phase_tile_ticket_decoy | visual_argument_transfer_component_value_nonpill | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| component_value_priority_chip_table_decoy | visual_argument_transfer_component_value_nonpill | False | False | True | False | True | False | 1 | 1 | 0 | argument_mismatch |
| component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | True | False | True | False | True | False | 1 | 1 | 0 | argument_mismatch |
| component_value_risk_badge_stale_selection_decoy | visual_tool_routing_component_value | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| component_value_severity_pill_chart_decoy | visual_argument_transfer_component_value_pill | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| component_value_state_pill_note_decoy | visual_argument_transfer_component_value_pill | True | False | True | False | True | False | 1 | 1 | 0 | argument_mismatch |
| component_value_status_badge_email_decoy | visual_argument_transfer_component_value_nonpill | False | True | False | True | False | True | 0 | 1 | 1 | exact |
