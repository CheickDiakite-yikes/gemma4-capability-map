# H2j Target-Query Normalization Transfer Synthesis

Generated: `2026-05-12T18:15:51.866187+00:00`

## Summary

H2j moves the component-identity repair from prompt wording into a controller-visible target-query normalization gate. The key result is not only that H2j closes the fresh H2f holdout at 10/10; it also preserves the older H2b and H1x transfer gates that rejected global H2h promotion. This is the first candidate in this line that repairs the displayed-value component-identity residual while retaining route-arbitration behavior on prior transfer slices.

![H2j transfer gate](figures/h2j_transfer_gate.svg)

## Packet Rows

| suite | profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2f | h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1 | 10 | 6 | 0.6 | 6 | 0.6 |
| h2f | h2h_component_identity_negative_examples | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1 | 10 | 9 | 0.9 | 9 | 0.9 |
| h2f | h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2f_execute_v2 | 10 | 10 | 1.0 | 10 | 1.0 |
| h2b | h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h2b_execute_v1 | 5 | 5 | 1.0 | 5 | 1.0 |
| h2b | h2h_component_identity_negative_examples | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2b_execute_v1 | 5 | 3 | 0.6 | 3 | 0.6 |
| h2b | h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2b_execute_v2 | 5 | 5 | 1.0 | 5 | 1.0 |
| h1x | h2e_route_arbitration | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |
| h1x | h2h_component_identity_negative_examples | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h1x_execute_v1 | 8 | 6 | 0.75 | 6 | 0.75 |
| h1x | h2j_target_query_normalization | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization | results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h1x_execute_v1 | 8 | 8 | 1.0 | 8 | 1.0 |

## Comparison Rows

| suite | comparison_label | comparison_dir | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2f | h2j_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2e_on_h2f_v2 | 10 | 0.6 | 1.0 | 0.4 | 0.6 | 1.0 | 0.4 |
| h2f | h2j_vs_h2h | results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2h_on_h2f_v2 | 10 | 0.9 | 1.0 | 0.09999999999999998 | 0.9 | 1.0 | 0.09999999999999998 |
| h2f | h2j_vs_h2i | results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2i_on_h2f_v2 | 10 | 0.6 | 1.0 | 0.4 | 0.6 | 1.0 | 0.4 |
| h2b | h2j_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2e_on_h2b_v2 | 5 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h2b | h2j_vs_h2h | results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2h_on_h2b_v2 | 5 | 0.6 | 1.0 | 0.4 | 0.6 | 1.0 | 0.4 |
| h1x | h2j_vs_h2e | results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2e_on_h1x_v1 | 8 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| h1x | h2j_vs_h2h | results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2h_on_h1x_v1 | 8 | 0.75 | 1.0 | 0.25 | 0.75 | 1.0 | 0.25 |

## H2j Non-Exact Rows

_None._

## Controller Intervention Rows

| suite | case_id | family | intervention_kind | from_tool | from_arguments | to_tool | to_arguments | prompt_state_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2f | h2f_result_tile_comment_value_decoy | h2f_route_component_class_transfer | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-result-tile","target_query":"Blocked"} | extract_layout | {"image_id":"img-h2f-result-tile","target_query":"result tile"} | result tile |
| h2f | h2f_resolution_badge_log_result_decoy | h2f_route_component_class_transfer | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-resolution-badge","target_query":"Deferred"} | extract_layout | {"image_id":"img-h2f-resolution-badge","target_query":"resolution badge"} | resolution badge |
| h2f | h2f_state_marker_history_value_decoy | h2f_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-state-marker","target_query":"lifecycle state marker"} | extract_layout | {"image_id":"img-h2f-state-marker","target_query":"state marker"} | state marker |
| h2f | h2f_mode_switch_note_value_decoy | h2f_route_nonstandard_class | visual_target_query_normalization | extract_layout | {"image_id":"img-h2f-mode-switch","target_query":"mode toggle"} | extract_layout | {"image_id":"img-h2f-mode-switch","target_query":"mode switch"} | mode switch |
| h2f | h2f_owner_field_previous_memo_decoy | h2f_route_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"owner field","selection_id":"sel-h2f-owner-memo"} | extract_layout | {"image_id":"img-h2f-owner-field","target_query":"owner field"} |  |
| h2b | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | visual_stale_selection_gate | refine_selection | {"filter_query":"approved","selection_id":null} | extract_layout | {"image_id":"img-component-result-pill","target_query":"result pill"} |  |
| h1x | h1x_responsible_party_field_old_owner_memo_decoy | h1x_oblique_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"responsible-party entry showing Iris","selection_id":"sel-owner-memo"} | extract_layout | {"image_id":"img-h1x-owner-field","target_query":"owner field"} |  |
| h1x | h1x_workstream_owner_field_previous_summary_decoy | h1x_oblique_stale_field | visual_stale_selection_gate | refine_selection | {"filter_query":"owner","selection_id":"sel-workstream-summary"} | extract_layout | {"image_id":"img-h1x-workstream-owner","target_query":"owner field"} |  |

## Findings

| finding_id | finding |
| --- | --- |
| h2j_closes_h2f | H2j reaches 10/10 strict and executor-equivalent on H2f, with exact-rate lift 0.4 versus H2e, 0.09999999999999998 versus H2h, and 0.4 versus H2i. |
| h2j_preserves_transfer_gates | H2j preserves the prior transfer gates: 5/5 on H2b and 8/8 on H1x. It ties H2e on both (0.0 H2b, 0.0 H1x) while beating H2h (0.4 H2b, 0.25 H1x). |
| h2j_controller_mechanism | H2j has 4 target-query-normalization interventions and 4 stale/missing selection interventions across H2f/H2b/H1x. The interventions are recorded per case in the replay artifacts, making the repair attributable to controller-visible state rather than hidden expected calls. |
| h2j_remaining_risk | H2j has 0 non-exact rows on the current H2f/H2b/H1x packet set. This supports promotion to a harder holdout, not global default status; the next test should target labels that appear both as requested targets and negated decoys. |
