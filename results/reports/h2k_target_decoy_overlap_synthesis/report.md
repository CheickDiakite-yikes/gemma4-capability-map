# H2k Target/Decoy Overlap Synthesis

Generated: `2026-05-12T18:42:09.373093+00:00`

## Summary

H2k is a post-H2j holdout that stresses prompts where the true visual target and a decoy share role, component class, displayed value, or code-label structure. H2j passes the packet at 8/8 while H2e and H2h remain below it, which supports the target-query normalization mechanism on a fresh overlap gate. The next claim requires helper ablation, not another prompt-only profile.

![H2k target/decoy overlap gate](figures/h2k_target_decoy_overlap_gate.svg)

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2e_execute_v1 | 8 | 3 | 0.375 | 6 | 0.75 |
| h2h_component_identity_negative_examples | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate | results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2h_execute_v1 | 8 | 6 | 0.75 | 6 | 0.75 |
| h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |

## Comparison Rows

| comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2j_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_h2e_v1 | 8 | 0.375 | 1.0 | 0.625 | 0.75 | 1.0 | 0.25 |
| h2j_vs_h2h | results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_h2h_v1 | 8 | 0.75 | 1.0 | 0.25 | 0.75 | 1.0 | 0.25 |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_target_query | actual_tool | actual_target_query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2e_route_arbitration | h2k_mode_toggle_negated_consent_toggle_decoy | h2k_negated_same_component_decoy | executable_paraphrase | True | extract_layout | mode toggle | extract_layout | mode toggle Manual |
| h2e_route_arbitration | h2k_result_badge_negated_result_tile_decoy | h2k_negated_same_component_decoy | executable_paraphrase | True | extract_layout | result badge | extract_layout | result badge Blocked |
| h2e_route_arbitration | h2k_error_banner_archived_error_notice_decoy | h2k_transfer_regression_guard | argument_mismatch | False | extract_layout | error banner | extract_layout | error notice |
| h2e_route_arbitration | h2k_state_tag_before_reading_state_marker_decoy | h2k_before_reading_decoy | executable_paraphrase | True | extract_layout | state tag | extract_layout | state tag Closed |
| h2e_route_arbitration | h2k_mode_field_before_reading_mode_switch_decoy | h2k_before_reading_decoy | argument_mismatch | False | extract_layout | mode field | extract_layout | mode toggle |
| h2h_component_identity_negative_examples | h2k_error_banner_archived_error_notice_decoy | h2k_transfer_regression_guard | argument_mismatch | False | extract_layout | error banner | extract_layout | error notice |
| h2h_component_identity_negative_examples | h2k_state_tag_before_reading_state_marker_decoy | h2k_before_reading_decoy | argument_mismatch | False | extract_layout | state tag | extract_layout | state marker Closed |

## H2j Controller Intervention Rows

| case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2k_mode_toggle_negated_consent_toggle_decoy | h2k_negated_same_component_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-mode-toggle","target_query":"mode toggle Manual"} | extract_layout | {"image_id":"img-h2k-mode-toggle","target_query":"mode toggle"} | mode toggle |
| h2k_result_badge_negated_result_tile_decoy | h2k_negated_same_component_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-result-badge","target_query":"result badge Blocked"} | extract_layout | {"image_id":"img-h2k-result-badge","target_query":"result badge"} | result badge |
| h2k_error_banner_archived_error_notice_decoy | h2k_transfer_regression_guard | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-error-banner","target_query":"error notice"} | extract_layout | {"image_id":"img-h2k-error-banner","target_query":"error banner"} | error banner |
| h2k_state_tag_before_reading_state_marker_decoy | h2k_before_reading_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-state-tag","target_query":"state tag Closed"} | extract_layout | {"image_id":"img-h2k-state-tag","target_query":"state tag"} | state tag |
| h2k_mode_field_before_reading_mode_switch_decoy | h2k_before_reading_decoy | visual_target_query_normalization | extract_layout | {"image_id":"img-h2k-mode-field","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2k-mode-field","target_query":"mode field"} | mode field |

## Findings

| finding_id | finding |
| --- | --- |
| h2k_is_discriminative | H2k separates H2j from the prior candidates: H2e reaches 3/8 exact, H2h reaches 6/8, and H2j reaches 8/8. |
| h2j_passes_target_decoy_overlap | H2j improves over H2e by 0.625 exact-rate and over H2h by 0.25 on H2k, with 0 H2j non-exact rows. |
| h2j_mechanism_is_target_normalization | H2j records 5 target-query-normalization interventions and 0 stale-selection interventions on H2k, so this holdout isolates the target-normalizer mechanism rather than stale rescue. |
| next_ablation_required | The next step is not another prompt-profile candidate. Run H2j without target-query normalization and H2j without stale-selection rescue on H2k to quantify the exact controller contribution. |
