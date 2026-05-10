# H1n Oblique Code-Hints Delta

Generated: `2026-05-10T01:23:26.943461+00:00`

## Findings

- `net_gain_with_regression`: Oblique code hints repairs 2 cases and regresses 1 case for a net executor-equivalence gain of 1 case.
- `repair_cases`: Repair gains: transfer_oblique_alert_p55_toggle_decoy, transfer_oblique_cell_r42_notice_decoy.
- `regression_case`: transfer_oblique_field_e19_old_selection_decoy regresses from the argument-hints exact call to refine_selection selection_id=sel-e19-archive filter_query=not.
- `preserved_argument_hints_wins`: Preserved argument-hints successes: transfer_oblique_badge_m88_chart_decoy, transfer_oblique_chip_z33_person_decoy, transfer_oblique_node_q17_table_decoy.
- `next_test`: Before broad promotion, run the code-hints profile on earlier oracle/repeat packets and either constrain stale-selection routing or build a fresh post-repair holdout.

## Case Deltas

| case_id | transition | classification | expected_tool | candidate_tool | expected_target_query | candidate_selection_id | candidate_filter_query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transfer_oblique_alert_p55_toggle_decoy | repair_gain | code_suffix_or_negated_decoy_repaired | extract_layout | extract_layout | alert p55 |  |  |
| transfer_oblique_badge_m88_chart_decoy | preserved_success | preserved_argument_hints_win | extract_layout | extract_layout | badge m88 |  |  |
| transfer_oblique_cell_r42_notice_decoy | repair_gain | code_suffix_or_negated_decoy_repaired | extract_layout | extract_layout | cell r42 |  |  |
| transfer_oblique_chip_z33_person_decoy | preserved_success | preserved_argument_hints_win | extract_layout | extract_layout | chip z33 |  |  |
| transfer_oblique_field_e19_old_selection_decoy | regression | stale_selection_tool_attraction | extract_layout | refine_selection | field e19 | sel-e19-archive | not |
| transfer_oblique_node_q17_table_decoy | preserved_success | preserved_argument_hints_win | extract_layout | extract_layout | node q17 |  |  |
