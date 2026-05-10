# H1n Oblique Miss Analysis

Generated: `2026-05-10T01:11:43.032606+00:00`

## Findings

- `argument_hints_miss_count`: Argument hints has 2 misses: code_suffix_truncation, negated_or_semantic_decoy_selected.
- `schema_field_miss_count`: Schema-field hints has 3 misses: code_suffix_truncation, semantic_broad_selection, tool_entry_failure.
- `next_intervention_target`: Next target should preserve short code suffixes and negated visible-target instructions, not revive broad schema-target-literal wording.

## Misses

| label | case_id | expected_target_query | actual_target_query | actual_region_ids | classification |
| --- | --- | --- | --- | --- | --- |
| argument_hints_v2 | transfer_oblique_cell_r42_notice_decoy | cell r42 | cell | ["transfer-oblique-cell-5301", "transfer-oblique-cell-5302", "transfer-oblique-cell-5303"] | code_suffix_truncation |
| argument_hints_v2 | transfer_oblique_alert_p55_toggle_decoy | alert p55 | consent toggle | ["transfer-oblique-alert-5501"] | negated_or_semantic_decoy_selected |
| schema_field_hints_v4 | transfer_oblique_node_q17_table_decoy | node q17 | owner escalation | ["transfer-oblique-node-5001", "transfer-oblique-node-5002", "transfer-oblique-node-5003"] | semantic_broad_selection |
| schema_field_hints_v4 | transfer_oblique_cell_r42_notice_decoy | cell r42 | cell | ["transfer-oblique-cell-5301", "transfer-oblique-cell-5302", "transfer-oblique-cell-5303"] | code_suffix_truncation |
| schema_field_hints_v4 | transfer_oblique_alert_p55_toggle_decoy | alert p55 |  | [] | tool_entry_failure |
