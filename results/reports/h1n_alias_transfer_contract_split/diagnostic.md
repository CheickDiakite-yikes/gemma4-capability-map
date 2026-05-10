# H1n Alias-Transfer Contract Split Diagnostic

- Generated at: `2026-05-10T00:07:11.429761+00:00`
- Cases: `6`
- Replay runs: `6`
- Expected-call contract mismatches: `5`
- Contracted exact-but-not-executor rows: `4`
- Argument-hints executor successes: `6`

## Findings

| finding_id | finding | implication |
| --- | --- | --- |
| expected_calls_are_not_oracle_calls | 5 / 6 generated expected-call contracts do not satisfy the packet's own expected_execution oracle. | H1n strict exactness partly measures matching the heuristic planner, not reaching the visual target. |
| contracted_exactness_is_overstated_for_h1n | Contracted MLX has 4 exact rows that are not executor-equivalent. | The contracted 5/6 strict score should not be treated as a clean model-only upper bound on H1n target success. |
| argument_hints_are_executor_oracle_winner | Argument hints v2 reaches 6 / 6 executor-target successes. | For this transfer slice, executor-equivalence is the more faithful outcome metric than strict planner-call exactness. |

## Summary

| label | case_count | strict_exact_count | executor_target_count | exact_but_executor_miss_count | nonexact_executor_success_count | contract_mismatch_count | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| expected_call_contract | 6 |  | 1 |  |  | 5 | Generated expected calls audited against the packet oracle. |
| argument_hints_v2 | 6 | 1 | 6 | 0 | 5 | 5 | 1/6 strict and 6/6 executor-target successes. |
| contracted | 6 | 5 | 1 | 4 | 0 | 5 | 5/6 strict but 4 exact rows miss the executor target. |
| no_directive | 6 | 0 | 2 | 0 | 2 | 5 | 0/6 strict and 2/6 executor-target successes. |
| role_catalog_v1 | 6 | 1 | 3 | 0 | 2 | 5 | 1/6 strict and 3/6 executor-target successes. |
| schema_field_hints_v4 | 6 | 1 | 2 | 0 | 1 | 5 | 1/6 strict and 2/6 executor-target successes. |
| schema_literal_targets_v5 | 6 | 1 | 4 | 0 | 3 | 5 | 1/6 strict and 4/6 executor-target successes. |

## Expected-Call Contract Audit

| case_id | family | expected_call_count | expected_call_satisfies_execution | expected_call_validator_pass | expected_call_classification | expected_calls | expected_execution | expected_call_execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transfer_review_tile_notice_table_decoy | visual_argument_transfer | 1 | False | True | expected_call_returns_empty_region_selection | [{"name":"extract_layout","arguments":{"image_id":"img-transfer-review-tile","target_query":"invoice totals table"}}] | {"region_ids":["transfer-tile-3001"]} | [{"selected_tool":"extract_layout","arguments":{"image_id":"img-transfer-review-tile","target_query":"invoice totals table"},"validator_result":"pass","region_ids":[],"region_id":"","error":null}] |
| transfer_status_pill_chart_decoy | visual_argument_transfer | 1 | False | False | expected_call_invalid_empty_region_id | [{"name":"read_region_text","arguments":{"image_id":"img-transfer-status-pill","region_id":""}}] | {"region_ids":["transfer-pill-3101"]} | [{"selected_tool":"read_region_text","arguments":{"image_id":"img-transfer-status-pill","region_id":""},"validator_result":"fail","region_ids":[],"region_id":"","error":"'Region not found: '"}] |
| transfer_error_banner_note_decoy | visual_argument_transfer | 1 | False | True | expected_call_returns_empty_region_selection | [{"name":"extract_layout","arguments":{"image_id":"img-transfer-error-banner","target_query":"validation error"}}] | {"region_ids":["transfer-banner-3202"]} | [{"selected_tool":"extract_layout","arguments":{"image_id":"img-transfer-error-banner","target_query":"validation error"},"validator_result":"pass","region_ids":[],"region_id":"","error":null}] |
| transfer_queue_badge_person_decoy | visual_argument_transfer | 1 | False | False | expected_call_invalid_empty_region_id | [{"name":"read_region_text","arguments":{"image_id":"img-transfer-queue-badge","region_id":""}}] | {"region_ids":["transfer-queue-3302"]} | [{"selected_tool":"read_region_text","arguments":{"image_id":"img-transfer-queue-badge","region_id":""},"validator_result":"fail","region_ids":[],"region_id":"","error":"'Region not found: '"}] |
| transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | 1 | True | True | expected_call_reaches_executor_target | [{"name":"extract_layout","arguments":{"image_id":"img-transfer-form-chip","target_query":"validation error"}}] | {"region_ids":["transfer-form-3402"]} | [{"selected_tool":"extract_layout","arguments":{"image_id":"img-transfer-form-chip","target_query":"validation error"},"validator_result":"pass","region_ids":["transfer-form-3402"],"region_id":"transfer-form-3402","error":null}] |
| transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | 1 | False | False | expected_call_invalid_empty_region_id | [{"name":"read_region_text","arguments":{"image_id":"img-transfer-signature-warning","region_id":""}}] | {"region_ids":["transfer-signature-3502"]} | [{"selected_tool":"read_region_text","arguments":{"image_id":"img-transfer-signature-warning","region_id":""},"validator_result":"fail","region_ids":[],"region_id":"","error":"'Region not found: '"}] |

## Interpretation

H1n exposed a benchmark-contract flaw: the packet's strict expected calls were generated by the heuristic planner, and most of those calls do not reach the visual oracle target when executed. Executor-equivalence is therefore the faithful outcome metric for this slice, while strict exactness should be reported as planner-call fidelity until H1n is rebuilt with oracle expected calls.
