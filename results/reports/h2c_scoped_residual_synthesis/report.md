# H2c Scoped Residual Synthesis

Generated: `2026-05-10T20:24:03.005612+00:00`

## Summary

H2c combines scoped residual-exactness wording with the existing stale-selection controller gate. On the five-row H2b residual packet it reaches `5 / 5` strict exact and `5 / 5` executor-equivalent, beating v12's `4 / 5` and H2a's `0 / 5` strict result. This is a local residual win, not a global promotion: H1s and H2a transfer still require a held-out transfer gate before any default change.

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_no_directive_execute_v1 | 5 | 1 | 0.2 | 2 | 0.4 |
| component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_value_guard_execute_v1 | 5 | 3 | 0.6 | 4 | 0.8 |
| component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_residual_guard_execute_v1 | 5 | 4 | 0.8 | 4 | 0.8 |
| h2a_stale_selection_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_h2a_execute_v1 | 5 | 0 | 0.0 | 3 | 0.6 |
| h2c_scoped_residual_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 5 | 1.0 | 5 | 1.0 |

## Case Matrix

| profile_label | case_id | family | source_failure_mode | replay_failure_mode | exact_match | executor_equivalence_match |
| --- | --- | --- | --- | --- | --- | --- |
| no_directive | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | exact | True | True |
| no_directive | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | False | True |
| no_directive | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | no_tool_call | False | False |
| no_directive | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | False | False |
| no_directive | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | no_tool_call | False | False |
| component_value_guard_v9 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | argument_mismatch | False | False |
| component_value_guard_v9 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | False | True |
| component_value_guard_v9 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | True | True |
| component_value_guard_v9 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | exact | True | True |
| component_value_guard_v9 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | True | True |
| component_residual_guard_v12 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | argument_mismatch | False | False |
| component_residual_guard_v12 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | True | True |
| component_residual_guard_v12 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | True | True |
| component_residual_guard_v12 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | exact | True | True |
| component_residual_guard_v12 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | True | True |
| h2a_stale_selection_gate | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | executable_paraphrase | False | True |
| h2a_stale_selection_gate | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | False | True |
| h2a_stale_selection_gate | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | False | True |
| h2a_stale_selection_gate | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | False | False |
| h2a_stale_selection_gate | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | argument_mismatch | False | False |
| h2c_scoped_residual_gate | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | exact | True | True |
| h2c_scoped_residual_gate | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | True | True |
| h2c_scoped_residual_gate | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | True | True |
| h2c_scoped_residual_gate | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | exact | True | True |
| h2c_scoped_residual_gate | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | True | True |

## Comparison Rows

| comparison_label | comparison_dir | baseline_packet_run_id | candidate_packet_run_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2c_vs_no_directive | results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_no_directive_on_h2b_v1 | 20260510T_h2b_residual_exactness_no_directive_execute_v1 | 20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 0.2 | 1.0 | 0.8 | 0.4 | 1.0 | 0.6 |
| h2c_vs_h2a | results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_h2a_on_h2b_v1 | 20260510T_h2b_residual_exactness_h2a_execute_v1 | 20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 0.0 | 1.0 | 1.0 | 0.6 | 1.0 | 0.4 |
| h2c_vs_v9 | results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_component_value_guard_on_h2b_v1 | 20260510T_h2b_residual_exactness_component_value_guard_execute_v1 | 20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 0.6 | 1.0 | 0.4 | 0.8 | 1.0 | 0.19999999999999996 |
| h2c_vs_v12 | results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_component_residual_guard_on_h2b_v1 | 20260510T_h2b_residual_exactness_component_residual_guard_execute_v1 | 20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 | 5 | 0.8 | 1.0 | 0.19999999999999996 | 0.8 | 1.0 | 0.19999999999999996 |

## Findings

| finding_id | finding |
| --- | --- |
| h2c_saturates_h2b_residuals | H2c reaches 5/5 strict and 5/5 executor-equivalent on the H2b residual packet. |
| h2c_beats_v12_strict_and_executor | H2c improves over v12 by 0.2 exact-rate and 0.2 executor-rate, fixing v12's remaining `result pill` miss while preserving its code-label and nonstandard-component wins. |
| h2c_separates_residual_exactness_from_h2a | H2a is still 0/5 strict and 3/5 executor-equivalent on H2b, while H2c gains 1.0 exact-rate. This keeps stale-selection mediation and residual exactness as distinct mechanisms. |
| h2c_surpasses_v9_executor_tie | V9 tied v12 on executor-equivalence at 4/5 but only reached 3/5 strict. H2c reaches 5/5 on both metrics. |
| next_slice | Do not promote H2c globally from a five-case residual fit. The next gate is a minimal transfer check over H1n/H1o/H1p/H1x residual families to detect whether scoped residual exactness harms the broader H2a executor profile. |
