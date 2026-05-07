# MLX Tool-Contract Harnessing Report

Generated: `2026-05-07T01:36:01.613587+00:00`

## Executive Read

The current local-Gemma research frontier is no longer top-line readiness on the aligned `32 / 26` surface. The strongest remaining signal is whether MLX Gemma can stay inside Moonie's tool interface without controller repair, fallback, and argument normalization.

H1h confirmed that the compact H1f no-directive finding survives the full ten-workflow live surface. H1i then compressed the worst H1h workflow families into a faster packet and amplified the same causal ordering.

The main finding is blunt: the tool-turn directive is a real model-side harness intervention, not presentation polish. When it is removed, no-directive MLX can still match readiness only because the controller repairs or substitutes calls. Raw no-directive tool compliance collapses on the probe suite.

## Figures

![H1i readiness, strict interface, and recovered execution](figures/h1i_readiness_strict_recovered.svg)

![H1h vs H1i no-directive controller burden](figures/h1h_h1i_controller_burden.svg)

![Tool probe contract gap](figures/tool_probe_contract_gap.svg)

![H1i failure modes](figures/h1i_failure_modes.svg)

![Prompt contract candidate targets](figures/prompt_contract_candidate_targets.svg)

![Executed prompt contract probe gate](figures/prompt_contract_probe_gate.svg)

![H1i prompt-contract repeat3 burden](figures/h1i_prompt_contract_repeat3_burden.svg)

## Packet Summary

| packet | episode_count | contracted_readiness | no_directive_readiness | readiness_delta_no_directive_vs_contracted | no_directive_controller_repair | no_directive_controller_fallback | no_directive_argument_repair | no_directive_raw_clean | no_repair_readiness | no_fallback_readiness | no_argument_repair_readiness | failure_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1f compact | 5 | 0.97936 | 0.97936 | 0.0 | 0.7 | 0.2 | 0.5 | 0.3 | 0.73818 | 0.92104 | 0.82036 | 12 |
| H1h full | 10 | 0.96891 | 0.96891 | 0.0 | 0.7 | 0.25 | 0.45 | 0.3 | 0.73801 | 0.89598 | 0.83016 | 24 |
| H1i worst-family | 4 | 0.9771 | 0.9771 | 0.0 | 1.0 | 0.5 | 0.5 | 0.0 | 0.64697 | 0.83125 | 0.8122 | 12 |

## H1i System Metrics

| label | system_id | runs | real_world_readiness_avg | strict_interface_avg | recovered_execution_avg | controller_repair_avg | controller_fallback_avg | argument_repair_avg | raw_planning_clean_rate_avg | disabled_controls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contracted | mlx_gemma4_e2b_reasoner_only | 4 | 0.9771 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| no directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | 4 | 0.9771 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 | 0.0 | disable_tool_turn_directive |
| no directive + no repair | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair | 4 | 0.64697 | 0.29688 | 0.0 | 1.25 | 1.25 | 0.0 | 0.725 | disable_controller_repair;disable_tool_turn_directive |
| no directive + no fallback | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback | 4 | 0.83125 | 0.625 | 0.5 | 0.5 | 0.0 | 0.5 | 0.5 | disable_controller_fallback;disable_tool_turn_directive |
| no directive + no arg repair | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair | 4 | 0.8122 | 0.71875 | 0.5 | 0.5 | 0.5 | 0.0 | 0.5 | disable_argument_repair;disable_tool_turn_directive |

## Probe Failure Modes

| side | failure_mode | case_count |
| --- | --- | --- |
| candidate | argument_mismatch | 4 |
| candidate | no_tool_call | 4 |
| baseline_non_exact | executable_paraphrase | 1 |

## Prompt-Contract Candidate Queue

| system_id | short_label | tool_prompt_contract_id | disable_tool_turn_directive | label | hypothesis | tags |
| --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard | Gemma 4 MLX literal guard | literal_argument_guard_v1 | True | Literal Argument Guard v1 | No-directive rows often choose the right tool but drift on arguments; stronger literal-copy rules may reduce repair burden. | arguments;canonicalization;cli;api;visual |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor | Gemma 4 MLX schema anchor | schema_anchor_v1 | True | Schema Anchor v1 | No-directive CLI/API misses may improve if the model is reminded that tool names and fields are literal interface tokens. | schema;json;cli;api |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required | Gemma 4 MLX tool required | tool_required_parallel_v1 | True | Tool Required Parallel v1 | No-directive visual and parallel cases may fail because the model exits the tool protocol; stronger tool-required wording should reduce no-call failures. | no_tool_call;parallel;visual |

These candidates are generic prompt contracts for the no-directive row. They deliberately avoid embedding the expected planned call, so they can be tested on the probe before spending H1i or H1h runs.

## Executed Prompt-Contract Probe Gate

| system_id | tool_prompt_contract_id | exact_match_rate | executable_match_rate | delta_exact_vs_contracted | delta_exact_vs_no_directive | probe_gate | improved_case_count | regressed_case_count | dominant_failure_mode | failure_modes | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor | schema_anchor_v1 | 0.125 | 0.0 | -0.75 | 0.125 | probe_improved_vs_no_directive | 1 | 0 | argument_mismatch | argument_mismatch:3;call_count_mismatch:1;exact:1;no_tool_call:3 | weak_exact_gain |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard | literal_argument_guard_v1 | 0.0 | 1.0 | -0.875 | 0.0 | probe_improved_vs_no_directive | 1 | 0 | no_tool_call | argument_mismatch:3;executable_paraphrase:1;no_tool_call:4 | visual_executable_gain_only |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required | tool_required_parallel_v1 | 0.0 | 1.0 | -0.875 | 0.0 | probe_improved_vs_no_directive | 1 | 0 | no_tool_call | argument_mismatch:1;executable_paraphrase:1;no_tool_call:6 | visual_executable_gain_only |

The first executed probe gate shows only partial gains. `schema_anchor_v1` recovers one exact visual readback case over no-directive, while `literal_argument_guard_v1` and `tool_required_parallel_v1` recover the executable visual target without improving exact JSON copy rate. All three remain far below the contracted MLX probe row.

## H1i Prompt-Contract Candidate Packet

| system_id | lane | disabled_controls | tool_turn_directive_enabled | real_world_readiness_avg | delta_vs_contracted_real_world_readiness_avg | delta_vs_no_directive_real_world_readiness_avg | strict_interface_avg | delta_vs_contracted_strict_interface_avg | delta_vs_no_directive_strict_interface_avg | recovered_execution_avg | delta_vs_contracted_recovered_execution_avg | delta_vs_no_directive_recovered_execution_avg | controller_repair_avg | delta_vs_contracted_controller_repair_avg | delta_vs_no_directive_controller_repair_avg | argument_repair_avg | delta_vs_contracted_argument_repair_avg | delta_vs_no_directive_argument_repair_avg | controller_fallback_avg | delta_vs_contracted_controller_fallback_avg | delta_vs_no_directive_controller_fallback_avg | intent_override_avg | delta_vs_contracted_intent_override_avg | delta_vs_no_directive_intent_override_avg | raw_planning_clean_rate_avg | delta_vs_contracted_raw_planning_clean_rate_avg | delta_vs_no_directive_raw_planning_clean_rate_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only | live_web_stress |  | True | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | live_web_stress | disable_tool_turn_directive | False | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |

The H1i candidate packet is saturated: contracted, no-directive, and all three prompt-contract candidates match on readiness, strict interface, recovered execution, controller burden, and raw clean rate. That means this H1i packet did not discriminate after the probe gate; the next second-stage slice needs harder or repeated no-directive cases.

## H1i Prompt-Contract Repeat3 Packet

| system_id | lane | disabled_controls | tool_turn_directive_enabled | real_world_readiness_avg | delta_vs_contracted_real_world_readiness_avg | delta_vs_no_directive_real_world_readiness_avg | strict_interface_avg | delta_vs_contracted_strict_interface_avg | delta_vs_no_directive_strict_interface_avg | recovered_execution_avg | delta_vs_contracted_recovered_execution_avg | delta_vs_no_directive_recovered_execution_avg | controller_repair_avg | delta_vs_contracted_controller_repair_avg | delta_vs_no_directive_controller_repair_avg | argument_repair_avg | delta_vs_contracted_argument_repair_avg | delta_vs_no_directive_argument_repair_avg | controller_fallback_avg | delta_vs_contracted_controller_fallback_avg | delta_vs_no_directive_controller_fallback_avg | intent_override_avg | delta_vs_contracted_intent_override_avg | delta_vs_no_directive_intent_override_avg | raw_planning_clean_rate_avg | delta_vs_contracted_raw_planning_clean_rate_avg | delta_vs_no_directive_raw_planning_clean_rate_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only | live_web_stress |  | True | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | live_web_stress | disable_tool_turn_directive | False | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9771000000000001 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |

The repeated H1i second-stage packet is also saturated. It expands the candidate packet to three attempts per workflow family per row, but all rows still remain controller-clean with raw clean rate `1.0`. The useful conclusion is negative: these packaged H1i workflows are now too deterministic to validate the prompt-contract candidates. The next harder slice should be probe-derived live cases, especially visual/parallel no-call cases.

## Gemini CLI Baseline Status

- Packet: `20260507T_h1h_gemini_cli_dry_run_baseline_v1`
- H1 slice: `knowledge_work_h1h_mlx_full_tool_contract_ablation:v1`
- Workflow count: `10`
- Dry run: `True`
- Binary: `definitely-missing-gemini-cli`

This packet is deliberately a dry-run prompt and command manifest. It is an external-reference baseline, not a replacement for Moonie's local MLX harness.

## Interpretation

- H1f established the compact causal ordering: no directive plus no controller repair was the largest drop.
- H1h verified that the ordering survives all ten H1e live workflow families.
- H1i is now the best fast loop because it targets the worst H1h no-repair families and makes the repair/fallback gaps larger.
- The no-directive probe explains why: CLI/API calls often keep the right tool but drift on canonical arguments, while visual referent and parallel-tool cases collapse to no tool call.
- The next prompt-contract experiment should be evaluated first on the probe suite and then on H1i before spending another full H1h run.

## Source Artifacts

- H1f compact: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1`
- H1h full: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1`
- H1i worst-family: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1`
- Probe comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1/probe_comparison.json`
- Prompt-contract probe packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1`
- H1i prompt-contract packet: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet`
- H1i prompt-contract repeat packet: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet`
- Gemini dry-run baseline: `/Users/cheickdiakite/Codex/moonie/results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1`
