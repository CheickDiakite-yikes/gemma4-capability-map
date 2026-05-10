# H2b Residual Exactness Synthesis

Generated: `2026-05-10T20:08:59.129329+00:00`

## Summary

H2b composes the five residual cases left by the H2a transfer gate. V12 is the strict winner at `4 / 5` and `4 / 5` executor-equivalent. V9 ties executor-equivalence at `4 / 5` but only reaches `3 / 5` strict. V15 fixes the two code-label rows plus result pill but misses both component-class rows. H2a itself falls to `0 / 5` strict and `3 / 5` executor-equivalent, confirming it is a stale-selection helper, not an alias exactness solution.

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | no_tool_call_count | argument_mismatch_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_no_directive_execute_v1 | `5` | `1` | `0.20000` | `2` | `0.40000` | `2` | `1` | `1` |
| component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_label_guard_execute_v1 | `5` | `0` | `0.00000` | `3` | `0.60000` | `0` | `2` | `3` |
| component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_value_guard_execute_v1 | `5` | `3` | `0.60000` | `4` | `0.80000` | `0` | `1` | `1` |
| component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_residual_guard_execute_v1 | `5` | `4` | `0.80000` | `4` | `0.80000` | `0` | `1` | `0` |
| code_label_exact_guard_v15 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_code_label_exact_guard_execute_v1 | `5` | `3` | `0.60000` | `3` | `0.60000` | `0` | `2` | `0` |
| h2a_stale_selection_gate | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate | results/tool_probe_replay_live/20260510T_h2b_residual_exactness_h2a_execute_v1 | `5` | `0` | `0.00000` | `3` | `0.60000` | `0` | `2` | `3` |

## Case Matrix

| profile_label | case_id | family | source_failure_mode | replay_failure_mode | exact_match | executor_equivalence_match |
| --- | --- | --- | --- | --- | --- | --- |
| no_directive | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | exact | `true` | `true` |
| no_directive | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | `false` | `true` |
| no_directive | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | no_tool_call | `false` | `false` |
| no_directive | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | `false` | `false` |
| no_directive | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | no_tool_call | `false` | `false` |
| component_label_guard_v11 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | executable_paraphrase | `false` | `true` |
| component_label_guard_v11 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | `false` | `true` |
| component_label_guard_v11 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | `false` | `true` |
| component_label_guard_v11 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | `false` | `false` |
| component_label_guard_v11 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | argument_mismatch | `false` | `false` |
| component_value_guard_v9 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | argument_mismatch | `false` | `false` |
| component_value_guard_v9 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | `false` | `true` |
| component_value_guard_v9 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | `true` | `true` |
| component_value_guard_v9 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | exact | `true` | `true` |
| component_value_guard_v9 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | `true` | `true` |
| component_residual_guard_v12 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | argument_mismatch | `false` | `false` |
| component_residual_guard_v12 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | `true` | `true` |
| component_residual_guard_v12 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | `true` | `true` |
| component_residual_guard_v12 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | exact | `true` | `true` |
| component_residual_guard_v12 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | exact | `true` | `true` |
| code_label_exact_guard_v15 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | exact | `true` | `true` |
| code_label_exact_guard_v15 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | `true` | `true` |
| code_label_exact_guard_v15 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | exact | `true` | `true` |
| code_label_exact_guard_v15 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | `false` | `false` |
| code_label_exact_guard_v15 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | argument_mismatch | `false` | `false` |
| h2a_stale_selection_gate | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | executable_paraphrase | `false` | `true` |
| h2a_stale_selection_gate | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | `false` | `true` |
| h2a_stale_selection_gate | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | executable_paraphrase | `false` | `true` |
| h2a_stale_selection_gate | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | argument_mismatch | `false` | `false` |
| h2a_stale_selection_gate | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | argument_mismatch | `false` | `false` |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | expected_tool | expected_arguments | actual_tool | actual_arguments | actual_region_ids |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | `true` | extract_layout | {"image_id": "img-h1o-code-alert-s92", "target_query": "alert s92"} | extract_layout | {"image_id": "img-h1o-code-alert-s92", "target_query": "alert"} | h1o-alert-s92-9122 |
| no_directive | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | no_tool_call | `false` | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "badge c08"} |  | {} |  |
| no_directive | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | `false` | extract_layout | {"image_id": "img-h1p-compact-state-tag", "target_query": "state tag"} | extract_layout | {"image_id": "img-h1p-compact-state-tag", "target_query": "state tag component"} |  |
| no_directive | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | no_tool_call | `false` | extract_layout | {"image_id": "img-h1p-surface-mode-toggle", "target_query": "mode toggle"} |  | {} |  |
| component_label_guard_v11 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | `true` | extract_layout | {"image_id": "img-component-result-pill", "target_query": "result pill"} | extract_layout | {"image_id": "img-component-result-pill", "target_query": "pill"} | component-result-pill-8502 |
| component_label_guard_v11 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | `true` | extract_layout | {"image_id": "img-h1o-code-alert-s92", "target_query": "alert s92"} | extract_layout | {"image_id": "img-h1o-code-alert-s92", "target_query": "alert"} | h1o-alert-s92-9122 |
| component_label_guard_v11 | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | `true` | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "badge c08"} | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "escalated badge"} | h1o-badge-c08-9132 |
| component_label_guard_v11 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | `false` | extract_layout | {"image_id": "img-h1p-compact-state-tag", "target_query": "state tag"} | extract_layout | {"image_id": "img-h1p-compact-state-tag", "target_query": "state pill"} |  |
| component_label_guard_v11 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | `false` | extract_layout | {"image_id": "img-h1p-surface-mode-toggle", "target_query": "mode toggle"} | extract_layout | {"image_id": "img-h1p-surface-mode-toggle", "target_query": "Manual"} | h1p-mode-note-9351,h1p-mode-toggle-9352,h1p-mode-table-9353 |
| component_value_guard_v9 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | argument_mismatch | `false` | extract_layout | {"image_id": "img-component-result-pill", "target_query": "result pill"} | extract_layout | {"image_id": "img-component-result-pill", "target_query": "approved"} | component-log-8501,component-result-pill-8502,component-review-8503 |
| component_value_guard_v9 | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | `true` | extract_layout | {"image_id": "img-h1o-code-alert-s92", "target_query": "alert s92"} | extract_layout | {"image_id": "img-h1o-code-alert-s92", "target_query": "alert"} | h1o-alert-s92-9122 |
| component_residual_guard_v12 | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | argument_mismatch | `false` | extract_layout | {"image_id": "img-component-result-pill", "target_query": "result pill"} | extract_layout | {"image_id": "img-component-result-pill", "target_query": "result badge"} |  |
| code_label_exact_guard_v15 | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | `false` | extract_layout | {"image_id": "img-h1p-compact-state-tag", "target_query": "state tag"} | extract_layout | {"image_id": "img-h1p-compact-state-tag", "target_query": "Closed"} | h1p-state-log-9331,h1p-state-tag-9332,h1p-state-table-9333 |
| code_label_exact_guard_v15 | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | `false` | extract_layout | {"image_id": "img-h1p-surface-mode-toggle", "target_query": "mode toggle"} | extract_layout | {"image_id": "img-h1p-surface-mode-toggle", "target_query": "Manual"} | h1p-mode-note-9351,h1p-mode-toggle-9352,h1p-mode-table-9353 |
| h2a_stale_selection_gate | component_value_result_pill_log_decoy | visual_argument_transfer_component_value_pill | executable_paraphrase | `true` | extract_layout | {"image_id": "img-component-result-pill", "target_query": "result pill"} | extract_layout | {"image_id": "img-component-result-pill", "target_query": "pill"} | component-result-pill-8502 |
| h2a_stale_selection_gate | h1o_code_alert_s92_negated_toggle_decoy | h1o_code_negation_preservation | executable_paraphrase | `true` | extract_layout | {"image_id": "img-h1o-code-alert-s92", "target_query": "alert s92"} | extract_layout | {"image_id": "img-h1o-code-alert-s92", "target_query": "alert"} | h1o-alert-s92-9122 |
| h2a_stale_selection_gate | h1o_code_badge_c08_note_decoy | h1o_code_negation_preservation | executable_paraphrase | `true` | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "badge c08"} | extract_layout | {"image_id": "img-h1o-code-badge-c08", "target_query": "escalated badge"} | h1o-badge-c08-9132 |
| h2a_stale_selection_gate | h1p_compact_state_tag_log_value_decoy | h1p_component_value_compact | argument_mismatch | `false` | extract_layout | {"image_id": "img-h1p-compact-state-tag", "target_query": "state tag"} | extract_layout | {"image_id": "img-h1p-compact-state-tag", "target_query": "state pill"} |  |
| h2a_stale_selection_gate | h1p_surface_mode_toggle_note_value_decoy | h1p_component_value_surface | argument_mismatch | `false` | extract_layout | {"image_id": "img-h1p-surface-mode-toggle", "target_query": "mode toggle"} | extract_layout | {"image_id": "img-h1p-surface-mode-toggle", "target_query": "Manual"} | h1p-mode-note-9351,h1p-mode-toggle-9352,h1p-mode-table-9353 |

## Findings

| finding_id | finding |
| --- | --- |
| h2b_is_a_real_residual_breaker | No-directive reaches 1/5 strict and 2/5 executor-equivalent; v11 reaches 0/5 strict and 3/5 executor-equivalent. The packet preserves pressure instead of washing out the residual mechanism. |
| v12_is_strict_winner | Component-residual guard v12 reaches 4/5 strict and 4/5 executor-equivalent, the best strict score on H2b. Its remaining miss is: component_value_result_pill_log_decoy. |
| v9_ties_executor_but_not_exact | Component-value guard v9 reaches 3/5 strict and 4/5 executor-equivalent, tying v12 on executor-equivalence but missing strict exactness on: component_value_result_pill_log_decoy, h1o_code_alert_s92_negated_toggle_decoy. |
| v15_solves_code_not_component_class | Code-label exact guard v15 reaches 3/5 strict and 3/5 executor-equivalent. It fixes the code-label rows plus result pill, but misses the component-class rows: h1p_compact_state_tag_log_value_decoy, h1p_surface_mode_toggle_note_value_decoy. |
| h2a_is_not_residual_exactness_solution | H2a reaches 0/5 strict and 3/5 executor-equivalent on H2b. It remains useful for stale-selection mediation, but does not solve the alias/code-label residual by itself. |
| next_slice | Do not globalize v12 despite the H2b win; H1s already showed transfer cost. The next move is H2c: a scoped residual route/factor that activates v12-like language only for exact alias/code-label contexts while preserving H2a's stale-selection controller gate. |
