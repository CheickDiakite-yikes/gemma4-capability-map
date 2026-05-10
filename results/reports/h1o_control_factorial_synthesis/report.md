# H1o Control-Factorial Synthesis

Generated: `2026-05-10T16:43:23.940988+00:00`

## Findings

- `strict_upper_bound`: Strict upper bound is argument_hints_v2, component_value_guard_v9 at 9/12.
- `executor_upper_bound`: Executor-equivalence upper bound is argument_hints_v2, hybrid_label_guard_v8, component_value_guard_v9 at 10/12; no H1o profile reaches full executor success.
- `activation_saturated_without_rescue`: Activation/no-call is not the remaining bottleneck on H1o: no-directive already reaches 4/4 exact, while no-call rescue reaches 3/4 and introduces one regression.
- `code_negation_is_repairable`: Code/negation failures are controller-sensitive: argument hints reaches 3/4 exact and 4/4 executor-equivalent versus no-directive at 1/4 exact and 2/4 executor-equivalent.
- `component_boundary_remains_residual`: Component/value boundaries remain the hard residue: component-value guard and argument hints both top out at 2/4 exact and 2/4 executor-equivalent on this family.
- `promotion_decision`: Promote argument hints as the conservative H1o default; do not promote no-call rescue globally; treat component-value guard as a tied candidate that needs a fresh component-only holdout.

## Profile Summary

| label | system_id | packet_dir | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | no_tool_call_count | argument_mismatch_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_directive_execute_v1 | 12 | 5 | 0.4166666666666667 | 6 | 0.5 | 3 | 3 |
| argument_hints_v2 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | results/tool_probe_replay_live/20260510T_h1o_control_factorial_argument_hints_execute_v1 | 12 | 9 | 0.75 | 10 | 0.8333333333333334 | 1 | 1 |
| hybrid_label_guard_v8 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard | results/tool_probe_replay_live/20260510T_h1o_control_factorial_hybrid_label_guard_execute_v1 | 12 | 8 | 0.6666666666666666 | 10 | 0.8333333333333334 | 0 | 2 |
| no_call_control_rescue_v10 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue | results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_call_control_rescue_execute_v1 | 12 | 7 | 0.5833333333333334 | 8 | 0.6666666666666666 | 1 | 3 |
| oblique_code_guard_v7 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard | results/tool_probe_replay_live/20260510T_h1o_control_factorial_oblique_code_guard_execute_v1 | 12 | 8 | 0.6666666666666666 | 9 | 0.75 | 0 | 3 |
| component_value_guard_v9 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_value_guard | results/tool_probe_replay_live/20260510T_h1o_control_factorial_component_value_guard_execute_v1 | 12 | 9 | 0.75 | 10 | 0.8333333333333334 | 0 | 2 |

## Family Summary

| label | family | case_count | exact_success_count | exact_rate | executor_success_count | executor_rate | no_tool_call_count | argument_mismatch_count | executable_paraphrase_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_directive | h1o_activation_no_call | 4 | 4 | 1.0 | 4 | 1.0 | 0 | 0 | 0 |
| no_directive | h1o_code_negation_preservation | 4 | 1 | 0.25 | 2 | 0.5 | 2 | 0 | 1 |
| no_directive | h1o_component_value_boundary | 4 | 0 | 0.0 | 0 | 0.0 | 1 | 3 | 0 |
| argument_hints_v2 | h1o_activation_no_call | 4 | 4 | 1.0 | 4 | 1.0 | 0 | 0 | 0 |
| argument_hints_v2 | h1o_code_negation_preservation | 4 | 3 | 0.75 | 4 | 1.0 | 0 | 0 | 1 |
| argument_hints_v2 | h1o_component_value_boundary | 4 | 2 | 0.5 | 2 | 0.5 | 1 | 1 | 0 |
| hybrid_label_guard_v8 | h1o_activation_no_call | 4 | 4 | 1.0 | 4 | 1.0 | 0 | 0 | 0 |
| hybrid_label_guard_v8 | h1o_code_negation_preservation | 4 | 3 | 0.75 | 4 | 1.0 | 0 | 0 | 1 |
| hybrid_label_guard_v8 | h1o_component_value_boundary | 4 | 1 | 0.25 | 2 | 0.5 | 0 | 2 | 1 |
| no_call_control_rescue_v10 | h1o_activation_no_call | 4 | 3 | 0.75 | 3 | 0.75 | 1 | 0 | 0 |
| no_call_control_rescue_v10 | h1o_code_negation_preservation | 4 | 2 | 0.5 | 3 | 0.75 | 0 | 1 | 1 |
| no_call_control_rescue_v10 | h1o_component_value_boundary | 4 | 2 | 0.5 | 2 | 0.5 | 0 | 2 | 0 |
| oblique_code_guard_v7 | h1o_activation_no_call | 4 | 4 | 1.0 | 4 | 1.0 | 0 | 0 | 0 |
| oblique_code_guard_v7 | h1o_code_negation_preservation | 4 | 3 | 0.75 | 4 | 1.0 | 0 | 0 | 1 |
| oblique_code_guard_v7 | h1o_component_value_boundary | 4 | 1 | 0.25 | 1 | 0.25 | 0 | 3 | 0 |
| component_value_guard_v9 | h1o_activation_no_call | 4 | 4 | 1.0 | 4 | 1.0 | 0 | 0 | 0 |
| component_value_guard_v9 | h1o_code_negation_preservation | 4 | 3 | 0.75 | 4 | 1.0 | 0 | 0 | 1 |
| component_value_guard_v9 | h1o_component_value_boundary | 4 | 2 | 0.5 | 2 | 0.5 | 0 | 2 | 0 |

## Family Deltas Versus No-Directive

| label | family | case_count | delta_exact_success_count | delta_exact_rate | delta_executor_success_count | delta_executor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| argument_hints_v2 | h1o_activation_no_call | 4 | 0 | 0.0 | 0 | 0.0 |
| argument_hints_v2 | h1o_code_negation_preservation | 4 | 2 | 0.5 | 2 | 0.5 |
| argument_hints_v2 | h1o_component_value_boundary | 4 | 2 | 0.5 | 2 | 0.5 |
| hybrid_label_guard_v8 | h1o_activation_no_call | 4 | 0 | 0.0 | 0 | 0.0 |
| hybrid_label_guard_v8 | h1o_code_negation_preservation | 4 | 2 | 0.5 | 2 | 0.5 |
| hybrid_label_guard_v8 | h1o_component_value_boundary | 4 | 1 | 0.25 | 2 | 0.5 |
| no_call_control_rescue_v10 | h1o_activation_no_call | 4 | -1 | -0.25 | -1 | -0.25 |
| no_call_control_rescue_v10 | h1o_code_negation_preservation | 4 | 1 | 0.25 | 1 | 0.25 |
| no_call_control_rescue_v10 | h1o_component_value_boundary | 4 | 2 | 0.5 | 2 | 0.5 |
| oblique_code_guard_v7 | h1o_activation_no_call | 4 | 0 | 0.0 | 0 | 0.0 |
| oblique_code_guard_v7 | h1o_code_negation_preservation | 4 | 2 | 0.5 | 2 | 0.5 |
| oblique_code_guard_v7 | h1o_component_value_boundary | 4 | 1 | 0.25 | 1 | 0.25 |
| component_value_guard_v9 | h1o_activation_no_call | 4 | 0 | 0.0 | 0 | 0.0 |
| component_value_guard_v9 | h1o_code_negation_preservation | 4 | 2 | 0.5 | 2 | 0.5 |
| component_value_guard_v9 | h1o_component_value_boundary | 4 | 2 | 0.5 | 2 | 0.5 |
