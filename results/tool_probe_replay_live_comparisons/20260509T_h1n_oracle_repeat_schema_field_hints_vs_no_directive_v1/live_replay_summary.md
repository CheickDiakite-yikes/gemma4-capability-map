# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints`
- Baseline exact rate: `0.3333333333333333`
- Candidate exact rate: `0.6666666666666666`
- Delta exact rate: `0.3333333333333333`
- Baseline executable rate: `0.3333333333333333`
- Candidate executable rate: `0.6666666666666666`
- Delta executable rate: `0.3333333333333333`
- Baseline executor-equivalence rate: `0.3333333333333333`
- Candidate executor-equivalence rate: `0.6666666666666666`
- Delta executor-equivalence rate: `0.3333333333333333`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| transfer_repeat_audit_card_email_decoy | visual_argument_transfer_repeat | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| transfer_repeat_consent_alert_toggle_decoy | visual_tool_routing_transfer_repeat | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| transfer_repeat_latency_chip_person_decoy | visual_argument_transfer_repeat | False | True | False | True | False | True | 0 | 1 | 1 | exact |
| transfer_repeat_missing_field_old_selection_decoy | visual_tool_routing_transfer_repeat | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| transfer_repeat_priority_tag_chart_decoy | visual_argument_transfer_repeat | False | False | False | False | False | False | 0 | 1 | 1 | argument_mismatch |
| transfer_repeat_warning_toast_note_decoy | visual_argument_transfer_repeat | False | True | False | True | False | True | 0 | 1 | 1 | exact |
