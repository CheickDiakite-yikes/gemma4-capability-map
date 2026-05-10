# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue`
- Baseline exact rate: `1.0`
- Candidate exact rate: `0.8333333333333334`
- Delta exact rate: `-0.16666666666666663`
- Baseline executable rate: `1.0`
- Candidate executable rate: `0.8333333333333334`
- Delta executable rate: `-0.16666666666666663`
- Baseline executor-equivalence rate: `1.0`
- Candidate executor-equivalence rate: `0.8333333333333334`
- Delta executor-equivalence rate: `-0.16666666666666663`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| transfer_oblique_alert_p55_toggle_decoy | visual_tool_routing_transfer_oblique | True | False | True | False | True | False | 1 | 1 | 0 | argument_mismatch |
| transfer_oblique_badge_m88_chart_decoy | visual_argument_transfer_oblique | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| transfer_oblique_cell_r42_notice_decoy | visual_argument_transfer_oblique | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| transfer_oblique_chip_z33_person_decoy | visual_argument_transfer_oblique | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| transfer_oblique_field_e19_old_selection_decoy | visual_tool_routing_transfer_oblique | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| transfer_oblique_node_q17_table_decoy | visual_argument_transfer_oblique | True | True | True | True | True | True | 1 | 1 | 0 | exact |
