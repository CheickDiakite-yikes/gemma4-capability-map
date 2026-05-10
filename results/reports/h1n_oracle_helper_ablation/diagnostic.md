# H1n Oracle Helper-Ablation Diagnostic

This diagnostic compares the H1n oracle argument-hints row against variants that disable one controller helper at a time.

## Findings

- helper rows: `3`
- strict rate: `0.8333333333333334`
- executor-equivalence rate: `1.0`
- all helpers preserve strict rate: `True`
- all helpers preserve executor-equivalence rate: `True`

## Summary

| helper_removed | comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_controller_repair | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_controller_repair_vs_argument_hints_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_repair | 6 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 1.0 | 1.0 | 0.0 | no_observed_helper_dependence |
| no_controller_fallback | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_controller_fallback_vs_argument_hints_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_fallback | 6 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 1.0 | 1.0 | 0.0 | no_observed_helper_dependence |
| no_argument_repair | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_argument_repair_vs_argument_hints_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_argument_repair | 6 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 1.0 | 1.0 | 0.0 | no_observed_helper_dependence |

## Case Deltas

| helper_removed | case_id | family | baseline_exact_match | candidate_exact_match | delta_exact_match | baseline_executor_equivalence_match | candidate_executor_equivalence_match | delta_executor_equivalence_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_controller_repair | transfer_error_banner_note_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_controller_repair | transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | None | None | 0 | None | None | 0 |
| no_controller_repair | transfer_queue_badge_person_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_controller_repair | transfer_review_tile_notice_table_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_controller_repair | transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | None | None | 0 | None | None | 0 |
| no_controller_repair | transfer_status_pill_chart_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_controller_fallback | transfer_error_banner_note_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_controller_fallback | transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | None | None | 0 | None | None | 0 |
| no_controller_fallback | transfer_queue_badge_person_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_controller_fallback | transfer_review_tile_notice_table_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_controller_fallback | transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | None | None | 0 | None | None | 0 |
| no_controller_fallback | transfer_status_pill_chart_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_argument_repair | transfer_error_banner_note_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_argument_repair | transfer_form_error_old_selection_chip_decoy | visual_tool_routing_transfer | None | None | 0 | None | None | 0 |
| no_argument_repair | transfer_queue_badge_person_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_argument_repair | transfer_review_tile_notice_table_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |
| no_argument_repair | transfer_signature_warning_checkbox_decoy | visual_tool_routing_transfer | None | None | 0 | None | None | 0 |
| no_argument_repair | transfer_status_pill_chart_decoy | visual_argument_transfer | None | None | 0 | None | None | 0 |

Interpretation: on this deterministic six-case oracle transfer packet, the argument-hints gain is not explained by controller repair, controller fallback, or argument repair. The result is negative for helper dependence, not broad proof that helpers never matter.
