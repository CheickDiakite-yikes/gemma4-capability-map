# H1n Oracle Transfer Synthesis

Generated: `2026-05-10T00:57:03.305903+00:00`

## Findings

- `oracle_v2_winner`: Oracle v2 winner set: exact=argument_hints_v2, executor=argument_hints_v2.
- `repeat_winner_set`: Repeat winner set: exact=argument_hints_v2, schema_literal_targets_v5, executor=argument_hints_v2, schema_literal_targets_v5.
- `helper_dependence`: Argument-hints helper ablations preserve both metrics: True.
- `contracted_not_upper_bound`: Contracted repeat candidate exact/executor rates are 0.0 / 0.0.

## Transfer Surfaces

| surface | label | shared_case_count | candidate_exact_rate | candidate_executor_equivalence_rate | delta_exact_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_v2 | contracted | 6 | 0.16666666666666666 | 0.16666666666666666 | -0.16666666666666666 | -0.16666666666666666 |
| oracle_v2 | role_catalog_v1 | 6 | 0.5 | 0.5 | 0.16666666666666669 | 0.16666666666666669 |
| oracle_v2 | argument_hints_v2 | 6 | 0.8333333333333334 | 1.0 | 0.5 | 0.6666666666666667 |
| oracle_v2 | schema_field_hints_v4 | 6 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 0.0 |
| oracle_v2 | schema_literal_targets_v5 | 6 | 0.6666666666666666 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 |
| repeat_v1 | contracted | 6 | 0.0 | 0.0 | -0.3333333333333333 | -0.3333333333333333 |
| repeat_v1 | role_catalog_v1 | 6 | 0.6666666666666666 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 |
| repeat_v1 | argument_hints_v2 | 6 | 0.8333333333333334 | 1.0 | 0.5 | 0.6666666666666667 |
| repeat_v1 | schema_field_hints_v4 | 6 | 0.6666666666666666 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 |
| repeat_v1 | schema_literal_targets_v5 | 6 | 0.8333333333333334 | 1.0 | 0.5 | 0.6666666666666667 |

## Helper Ablation

| helper_removed | comparison_dir | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executor_equivalence_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_controller_repair | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_controller_repair_vs_argument_hints_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_repair | 6 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 1.0 | 1.0 | 0.0 | no_observed_helper_dependence |
| no_controller_fallback | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_controller_fallback_vs_argument_hints_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_fallback | 6 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 1.0 | 1.0 | 0.0 | no_observed_helper_dependence |
| no_argument_repair | /Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_argument_repair_vs_argument_hints_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_argument_repair | 6 | 0.8333333333333334 | 0.8333333333333334 | 0.0 | 1.0 | 1.0 | 0.0 | no_observed_helper_dependence |

Interpretation: H1n now has a two-packet oracle-backed transfer result. Argument hints wins the first oracle packet and ties schema target literals on the repeat. The argument-hints gain is not explained by the three tested controller helpers, while contracted prompting is not a reliable upper bound on these transfer packets.
