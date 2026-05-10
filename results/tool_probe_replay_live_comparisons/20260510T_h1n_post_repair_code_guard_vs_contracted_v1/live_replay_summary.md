# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard`
- Baseline exact rate: `0.375`
- Candidate exact rate: `0.75`
- Delta exact rate: `0.375`
- Baseline executable rate: `0.375`
- Candidate executable rate: `0.75`
- Delta executable rate: `0.375`
- Baseline executor-equivalence rate: `0.375`
- Candidate executor-equivalence rate: `0.75`
- Delta executor-equivalence rate: `0.375`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| post_repair_alert_c77_toggle_decoy | visual_tool_routing_transfer_post_repair | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| post_repair_badge_t64_notice_decoy | visual_argument_transfer_post_repair_code | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| post_repair_chip_l90_person_decoy | visual_argument_transfer_post_repair_code | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| post_repair_field_b12_stale_selection_decoy | visual_tool_routing_transfer_post_repair | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| post_repair_node_k21_chart_decoy | visual_argument_transfer_post_repair_code | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| post_repair_review_tile_table_decoy | visual_argument_transfer_post_repair_noncode | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| post_repair_status_pill_note_decoy | visual_argument_transfer_post_repair_noncode | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| post_repair_warning_toast_email_decoy | visual_argument_transfer_post_repair_noncode | False | True | False | True | False | True | 1 | 1 | 0 | exact |
