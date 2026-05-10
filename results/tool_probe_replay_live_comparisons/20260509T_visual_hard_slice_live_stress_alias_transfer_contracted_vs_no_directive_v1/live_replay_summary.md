# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only`
- Baseline exact rate: `0.0`
- Candidate exact rate: `0.8333333333333334`
- Delta exact rate: `0.8333333333333334`
- Baseline executable rate: `0.3333333333333333`
- Candidate executable rate: `0.16666666666666666`
- Delta executable rate: `-0.16666666666666666`
- Baseline executor-equivalence rate: `0.3333333333333333`
- Candidate executor-equivalence rate: `0.16666666666666666`
- Delta executor-equivalence rate: `-0.16666666666666666`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| transfer_error_banner_note_decoy | visual_argument_transfer | False | True | True | False | True | False | 1 | 1 | 0 | exact |
| transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| transfer_queue_badge_person_decoy | visual_argument_transfer | False | True | True | False | True | False | 1 | 1 | 0 | exact |
| transfer_review_tile_notice_table_decoy | visual_argument_transfer | False | True | False | False | False | False | 0 | 1 | 1 | exact |
| transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | False | True | False | False | False | False | 1 | 1 | 0 | exact |
| transfer_status_pill_chart_decoy | visual_argument_transfer | False | False | False | False | False | False | 0 | 1 | 1 | wrong_tool |
