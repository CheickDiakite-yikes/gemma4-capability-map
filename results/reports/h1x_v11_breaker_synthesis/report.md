# H1x V11-Breaker Synthesis

Generated: `2026-05-10T18:50:42.692248+00:00`

## Summary

H1x is the first focused post-H1w packet that breaks v11 saturation. No-directive reaches `2 / 8`, v11 reaches `7 / 8`, v12 reaches `8 / 8`, and v15 reaches `6 / 8` strict exact with `7 / 8` executor-equivalent. The result strengthens the routed-helper hypothesis: residual wording is locally useful on oblique stale-field rows, while code-label exactness remains too narrow.

## Packet Rows

| profile_label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | argument_mismatch_count | executable_paraphrase_count | wrong_tool_count | no_tool_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1 | `8` | `2` | `0.25000` | `2` | `0.25000` | `2` | `0` | `0` | `4` |
| component_label_guard_v11 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_label_guard_execute_v1 | `8` | `7` | `0.87500` | `7` | `0.87500` | `0` | `0` | `1` | `0` |
| component_residual_guard_v12 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_residual_guard_execute_v1 | `8` | `8` | `1.00000` | `8` | `1.00000` | `0` | `0` | `0` | `0` |
| code_label_exact_guard_v15 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | results/tool_probe_replay_live/20260510T_h1x_v11_breaker_code_label_exact_guard_execute_v1 | `8` | `6` | `0.75000` | `7` | `0.87500` | `0` | `1` | `1` | `0` |

## Family Rows

| profile_label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| no_directive | h1x_oblique_activation_no_call | `2` | `2` | `1.00000` | `2` | `1.00000` |
| no_directive | h1x_oblique_nonstandard_class | `2` | `0` | `0.00000` | `0` | `0.00000` |
| no_directive | h1x_oblique_stale_field | `2` | `0` | `0.00000` | `0` | `0.00000` |
| no_directive | h1x_oblique_surface_value | `2` | `0` | `0.00000` | `0` | `0.00000` |
| component_label_guard_v11 | h1x_oblique_activation_no_call | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_label_guard_v11 | h1x_oblique_nonstandard_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_label_guard_v11 | h1x_oblique_stale_field | `2` | `1` | `0.50000` | `1` | `0.50000` |
| component_label_guard_v11 | h1x_oblique_surface_value | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1x_oblique_activation_no_call | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1x_oblique_nonstandard_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1x_oblique_stale_field | `2` | `2` | `1.00000` | `2` | `1.00000` |
| component_residual_guard_v12 | h1x_oblique_surface_value | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1x_oblique_activation_no_call | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1x_oblique_nonstandard_class | `2` | `2` | `1.00000` | `2` | `1.00000` |
| code_label_exact_guard_v15 | h1x_oblique_stale_field | `2` | `1` | `0.50000` | `1` | `0.50000` |
| code_label_exact_guard_v15 | h1x_oblique_surface_value | `2` | `1` | `0.50000` | `2` | `1.00000` |

## Comparison Rows

| comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tool_probe_replay_live_comparisons/20260510T_h1x_component_label_guard_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | `8` | `0.25000` | `0.87500` | `0.62500` | `0.25000` | `0.87500` | `0.62500` |
| results/tool_probe_replay_live_comparisons/20260510T_h1x_component_residual_guard_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `8` | `0.25000` | `1.00000` | `0.75000` | `0.25000` | `1.00000` | `0.75000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1x_code_label_exact_guard_vs_no_directive_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `8` | `0.25000` | `0.75000` | `0.50000` | `0.25000` | `0.87500` | `0.62500` |
| results/tool_probe_replay_live_comparisons/20260510T_h1x_component_residual_guard_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | `8` | `0.87500` | `1.00000` | `0.12500` | `0.87500` | `1.00000` | `0.12500` |
| results/tool_probe_replay_live_comparisons/20260510T_h1x_code_label_exact_guard_vs_component_label_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `8` | `0.87500` | `0.75000` | `-0.12500` | `0.87500` | `0.87500` | `0.00000` |
| results/tool_probe_replay_live_comparisons/20260510T_h1x_code_label_exact_guard_vs_component_residual_guard_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard | `8` | `1.00000` | `0.75000` | `-0.25000` | `1.00000` | `0.87500` | `-0.12500` |

## Non-Exact Rows

| profile_label | case_id | family | failure_mode | executor_equivalence_match | output_dir |
| --- | --- | --- | --- | --- | --- |
| no_directive | h1x_responsible_party_field_old_owner_memo_decoy | h1x_oblique_stale_field | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1/runs/h1x_responsible_party_field_old_owner_memo_decoy |
| no_directive | h1x_workstream_owner_field_previous_summary_decoy | h1x_oblique_stale_field | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1/runs/h1x_workstream_owner_field_previous_summary_decoy |
| no_directive | h1x_resolution_chip_comment_result_decoy | h1x_oblique_surface_value | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1/runs/h1x_resolution_chip_comment_result_decoy |
| no_directive | h1x_progress_marker_summary_status_decoy | h1x_oblique_surface_value | argument_mismatch | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1/runs/h1x_progress_marker_summary_status_decoy |
| no_directive | h1x_lifecycle_marker_log_state_tag_decoy | h1x_oblique_nonstandard_class | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1/runs/h1x_lifecycle_marker_log_state_tag_decoy |
| no_directive | h1x_operation_mode_control_note_toggle_decoy | h1x_oblique_nonstandard_class | no_tool_call | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1/runs/h1x_operation_mode_control_note_toggle_decoy |
| component_label_guard_v11 | h1x_responsible_party_field_old_owner_memo_decoy | h1x_oblique_stale_field | wrong_tool | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_label_guard_execute_v1/runs/h1x_responsible_party_field_old_owner_memo_decoy |
| code_label_exact_guard_v15 | h1x_responsible_party_field_old_owner_memo_decoy | h1x_oblique_stale_field | wrong_tool | `false` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_code_label_exact_guard_execute_v1/runs/h1x_responsible_party_field_old_owner_memo_decoy |
| code_label_exact_guard_v15 | h1x_resolution_chip_comment_result_decoy | h1x_oblique_surface_value | executable_paraphrase | `true` | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260510T_h1x_v11_breaker_code_label_exact_guard_execute_v1/runs/h1x_resolution_chip_comment_result_decoy |

## Findings

| finding_id | finding |
| --- | --- |
| h1x_breaks_no_directive | No-directive reaches 2/8 exact and 2/8 executor-equivalent; it only solves the activation/no-call rows and fails the oblique stale-field, surface-value, and nonstandard-class rows. |
| h1x_breaks_v11_saturation | Component-label guard v11 drops to 7/8 exact and 7/8 executor-equivalent; the miss is concentrated in oblique stale-field routing at 1/2. |
| v12_local_winner | Component-residual guard v12 reaches 8/8 exact and 8/8 executor-equivalent, a +0.750 exact-rate delta over no-directive and +0.125 over v11 on H1x. |
| v15_over_narrows_again | Code-label exact guard v15 reaches 6/8 exact and 7/8 executor-equivalent. It is only 1/2 strict exact on oblique surface-value rows; non-exact rows: h1x_responsible_party_field_old_owner_memo_decoy, h1x_resolution_chip_comment_result_decoy. |
| next_slice | Treat H1x as evidence for a routed residual helper, not a global default replacement. The next hard slice should retest v12 against the old transfer packets and a new mixed packet with oblique stale-field plus surface-value rows in the same workflow family. |
