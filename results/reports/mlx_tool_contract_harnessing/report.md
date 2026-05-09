# MLX Tool-Contract Harnessing Report

Generated: `2026-05-09T00:35:45.433862+00:00`

## Executive Read

The current local-Gemma research frontier is no longer top-line readiness on the aligned `32 / 26` surface. The strongest remaining signal is whether MLX Gemma can stay inside Moonie's tool interface without controller repair, fallback, and argument normalization.

H1h confirmed that the compact H1f no-directive finding survives the full ten-workflow live surface. H1i then compressed the worst H1h workflow families into a faster packet and amplified the same causal ordering.

The main finding is blunt: the tool-turn directive is a real model-side harness intervention, not presentation polish. When it is removed, no-directive MLX can still match readiness only because the controller repairs or substitutes calls. Raw no-directive tool compliance collapses on the probe suite.

The visual catalog branch now includes an explicit negative-result loop. `visual_role_catalog_argument_hints_v2` remains the best exact visual candidate. `visual_role_catalog_split_selector_hints_v3` was rejected before live replay because it regressed exact readback, and `visual_role_catalog_schema_field_hints_v4` tied v2 on exactness without recovering executable form targeting.

## Figures

![H1i readiness, strict interface, and recovered execution](figures/h1i_readiness_strict_recovered.svg)

![H1h vs H1i no-directive controller burden](figures/h1h_h1i_controller_burden.svg)

![Tool probe contract gap](figures/tool_probe_contract_gap.svg)

![H1i failure modes](figures/h1i_failure_modes.svg)

![Prompt contract candidate targets](figures/prompt_contract_candidate_targets.svg)

![Executed prompt contract probe gate](figures/prompt_contract_probe_gate.svg)

![Prompt contract wave two probe gate](figures/prompt_contract_wave2_probe_gate.svg)

![Prompt contract wave three probe gate](figures/prompt_contract_wave3_probe_gate.svg)

![Prompt contract wave four probe gate](figures/prompt_contract_wave4_probe_gate.svg)

![Prompt contract wave five probe gate](figures/prompt_contract_wave5_probe_gate.svg)

![Tool catalog profile probe gate](figures/tool_catalog_profile_probe_gate.svg)

![Prompt contract wave six probe gate](figures/prompt_contract_wave6_probe_gate.svg)

![H1i prompt-contract repeat3 burden](figures/h1i_prompt_contract_repeat3_burden.svg)

![H1j probe-derived candidate burden](figures/h1j_probe_derived_burden.svg)

![H1j probe-derived helper burden](figures/h1j_probe_derived_helper_burden.svg)

![H1k parallel-audit candidate burden](figures/h1k_parallel_audit_burden.svg)

![H1k parallel-audit helper burden](figures/h1k_parallel_audit_helper_burden.svg)

![Exact probe replay gap](figures/exact_probe_replay_gap.svg)

![Focused exact replay gaps](figures/exact_probe_replay_focus_gap.svg)

![CLI-live parallel replay gap](figures/live_parallel_replay_gap.svg)

![CLI-live focused replay gaps](figures/live_replay_focus_gap.svg)

![Wave three live replay gate](figures/wave3_live_candidate_replay_gate.svg)

![Wave four live replay gate](figures/wave4_live_candidate_replay_gate.svg)

![Visual catalog live replay gate](figures/visual_catalog_live_candidate_replay_gate.svg)

![Visual catalog argument-hints live replay gate](figures/visual_catalog_argument_hints_live_candidate_replay_gate.svg)

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

| system_id | short_label | tool_prompt_contract_id | tool_catalog_profile_id | disable_tool_turn_directive | label | hypothesis | tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_canonical_json_copy | Gemma 4 MLX canonical JSON | canonical_json_copy_v3 |  | True | Canonical JSON Copy v3 | Live replay shows no-directive MLX often enters the tool protocol but drifts on canonical CLI/API arguments; tighter token-copy rules may reduce argument repair without leaking the planned call. | schema;arguments;canonicalization;json;cli;api |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard | Gemma 4 MLX literal guard | literal_argument_guard_v1 |  | True | Literal Argument Guard v1 | No-directive rows often choose the right tool but drift on arguments; stronger literal-copy rules may reduce repair burden. | arguments;canonicalization;cli;api;visual |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required | Gemma 4 MLX parallel array | parallel_array_required_v2 |  | True | Parallel Array Required v2 | The parallel probe collapsed to no tool call under no-directive prompting; explicit array-shape rules may recover parallel calls. | parallel;no_tool_call;json_array;multi_source |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_two_call_array | Gemma 4 MLX parallel two-call | parallel_two_call_array_v3 |  | True | Parallel Two-Call Array v3 | CLI-live parallel replay shows no-directive MLX asks the operator for inputs already present; explicit source-count and array-shape rules may preserve the two-call contract. | parallel;no_tool_call;json_array;multi_source;arguments |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor | Gemma 4 MLX schema anchor | schema_anchor_v1 |  | True | Schema Anchor v1 | No-directive CLI/API misses may improve if the model is reminded that tool names and fields are literal interface tokens. | schema;json;cli;api |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required | Gemma 4 MLX schema literal required | schema_literal_tool_required_v2 |  | True | Schema Literal Tool-Required v2 | The first wave split exact-copy and executable visual gains; a combined contract may preserve schema obedience while reducing no-call failures. | schema;arguments;no_tool_call;json;cli;api;visual |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required | Gemma 4 MLX tool required | tool_required_parallel_v1 |  | True | Tool Required Parallel v1 | No-directive visual and parallel cases may fail because the model exits the tool protocol; stronger tool-required wording should reduce no-call failures. | no_tool_call;parallel;visual |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_next_call_state | Gemma 4 MLX visual next call | visual_next_call_state_v2 |  | True | Visual Next-Call State v2 | No-directive visual failures are concentrated in no-call behavior after a visual referent exists; explicit state-transition wording may reduce that collapse. | visual;no_tool_call;state_machine;readback |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_refine_selection | Gemma 4 MLX visual refine | visual_refine_selection_v5 |  | True | Visual Refine Selection v5 | Wave four proved broad visual state wording did not fix the filter/refinement case; a narrower contract that prioritizes refine_selection for existing selection_id filtering may preserve visual initiation while changing the targeted wrong-tool failure. | visual;tool_selection;refine_selection;filtering;arguments |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_literal_guard | Gemma 4 MLX visual catalog literal | literal_argument_guard_v1 | visual_role_catalog_v1 | True | Literal Argument Guard v1 | No-directive rows often choose the right tool but drift on arguments; stronger literal-copy rules may reduce repair burden. | arguments;canonicalization;cli;api;visual |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection | Gemma 4 MLX visual state tool | visual_state_tool_selection_v4 |  | True | Visual State Tool Selection v4 | Wave three recovered visual tool initiation but still chose the wrong visual tool for a filter/refinement case; state-specific selection rules may preserve tool entry while improving exact visual replay. | visual;state_machine;tool_selection;no_tool_call;arguments |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation | Gemma 4 MLX visual initiation | visual_tool_initiation_v3 |  | True | Visual Tool Initiation v3 | CLI-live visual replay shows no-directive MLX often answers or defers instead of initiating the next visual tool call; a compact state-transition contract may recover tool entry before exact selector tuning. | visual;no_tool_call;state_machine;readback;arguments |

These candidates are generic prompt contracts for the no-directive row. They deliberately avoid embedding the expected planned call, so they can be tested on the probe before spending H1i or H1h runs.

## Executed Prompt-Contract Probe Gate

| system_id | tool_prompt_contract_id | exact_match_rate | executable_match_rate | delta_exact_vs_contracted | delta_exact_vs_no_directive | probe_gate | improved_case_count | regressed_case_count | dominant_failure_mode | failure_modes | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor | schema_anchor_v1 | 0.125 | 0.0 | -0.75 | 0.125 | probe_improved_vs_no_directive | 1 | 0 | argument_mismatch | argument_mismatch:3;call_count_mismatch:1;exact:1;no_tool_call:3 | weak_exact_gain |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard | literal_argument_guard_v1 | 0.0 | 1.0 | -0.875 | 0.0 | probe_improved_vs_no_directive | 1 | 0 | no_tool_call | argument_mismatch:3;executable_paraphrase:1;no_tool_call:4 | visual_executable_gain_only |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required | tool_required_parallel_v1 | 0.0 | 1.0 | -0.875 | 0.0 | probe_improved_vs_no_directive | 1 | 0 | no_tool_call | argument_mismatch:1;executable_paraphrase:1;no_tool_call:6 | visual_executable_gain_only |

The first executed probe gate shows only partial gains. `schema_anchor_v1` recovers one exact visual readback case over no-directive, while `literal_argument_guard_v1` and `tool_required_parallel_v1` recover the executable visual target without improving exact JSON copy rate. All three remain far below the contracted MLX probe row.

## Prompt-Contract Wave Two Probe Gate

| system_id | tool_prompt_contract_id | exact_match_rate | executable_match_rate | delta_exact_vs_contracted | delta_exact_vs_no_directive | probe_gate | improved_case_count | regressed_case_count | dominant_failure_mode | failure_modes | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required | schema_literal_tool_required_v2 | 0.125 | 0.0 | -0.75 | 0.125 | probe_improved_vs_no_directive | 1 | 0 | argument_mismatch | argument_mismatch:3;call_count_mismatch:1;exact:1;no_tool_call:3 | weak_exact_gain |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_next_call_state | visual_next_call_state_v2 | 0.0 | 1.0 | -0.875 | 0.0 | probe_improved_vs_no_directive | 1 | 0 | no_tool_call | call_count_mismatch:1;executable_paraphrase:1;no_tool_call:6 | visual_executable_gain_only |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required | parallel_array_required_v2 | 0.0 | 0.0 | -0.875 | 0.0 | no_probe_improvement_vs_no_directive | 0 | 0 | no_tool_call | argument_mismatch:2;no_tool_call:5;wrong_tool:1 | no_probe_gain |

The second wave confirms the same shape rather than changing the direction. `schema_literal_tool_required_v2` gives a weak one-case exact gain, `visual_next_call_state_v2` restores executable visual behavior without exact JSON fidelity, and `parallel_array_required_v2` does not improve the parallel/no-call family. None of the wave-two candidates is strong enough to replace the final tool-turn directive.

## Prompt-Contract Wave Three Probe Gate

| system_id | tool_prompt_contract_id | exact_match_rate | executable_match_rate | delta_exact_vs_contracted | delta_exact_vs_no_directive | probe_gate | improved_case_count | regressed_case_count | dominant_failure_mode | failure_modes | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_canonical_json_copy | canonical_json_copy_v3 | 0.125 | 0.0 | -0.75 | 0.125 | probe_improved_vs_no_directive | 1 | 0 | no_tool_call | argument_mismatch:2;call_count_mismatch:1;exact:1;no_tool_call:3;wrong_tool:1 | weak_exact_gain |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation | visual_tool_initiation_v3 | 0.125 | 1.0 | -0.75 | 0.125 | probe_improved_vs_no_directive | 2 | 0 | no_tool_call | call_count_mismatch:1;exact:1;executable_paraphrase:1;no_tool_call:4;wrong_tool:1 | weak_exact_gain |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_two_call_array | parallel_two_call_array_v3 | 0.0 | 0.0 | -0.875 | 0.0 | no_probe_improvement_vs_no_directive | 0 | 0 | no_tool_call | argument_mismatch:1;no_tool_call:7 | no_probe_gain |

The third wave targets the mechanisms exposed by CLI-live replay: canonical argument copying, visual tool initiation, and two-call parallel array shape. It produces the same hard boundary in sharper form: canonical and visual-initiation wording recover one exact case, the visual-initiation contract also recovers the executable visual target, and the parallel two-call contract still does not recover the parallel no-call family.

## Prompt-Contract Wave Four Probe Gate

| system_id | tool_prompt_contract_id | exact_match_rate | executable_match_rate | delta_exact_vs_contracted | delta_exact_vs_no_directive | probe_gate | improved_case_count | regressed_case_count | dominant_failure_mode | failure_modes | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection | visual_state_tool_selection_v4 | 0.125 | 0.0 | -0.75 | 0.125 | probe_improved_vs_no_directive | 1 | 0 | no_tool_call | argument_mismatch:2;call_count_mismatch:1;exact:1;no_tool_call:3;wrong_tool:1 | weak_exact_gain |

`visual_state_tool_selection_v4` was the narrow follow-up to wave three's best partial result. Raw probe exact rate again reaches only `0.125`: enough to improve over the no-directive row by one case, but still far below the contracted row. The dominant failure remains `no_tool_call`, so the contract should be treated as a targeted visual replay candidate, not a general harness fix.

## Prompt-Contract Wave Five Probe Gate

| system_id | tool_prompt_contract_id | exact_match_rate | executable_match_rate | delta_exact_vs_contracted | delta_exact_vs_no_directive | probe_gate | improved_case_count | regressed_case_count | dominant_failure_mode | failure_modes | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_refine_selection | visual_refine_selection_v5 | 0.0 | 0.0 | -0.875 | 0.0 | no_probe_improvement_vs_no_directive | 0 | 0 | no_tool_call | argument_mismatch:1;call_count_mismatch:1;no_tool_call:6 | no_probe_gain |

`visual_refine_selection_v5` was more surgical: it targeted only latest-selection filtering and `refine_selection`. The raw probe rejected it before live replay: exact rate stayed `0.0`, executable rate stayed `0.0`, and the dominant failure shifted further toward `no_tool_call`. Under the current gate, this candidate should not spend CLI-live replay or H1 budget.

## Tool-Catalog Profile Probe Gate

| system_id | tool_catalog_profile_id | execute | output_dir | comparison_path | no_directive_comparison_path | exact_match_rate | executable_match_rate | delta_exact_vs_contracted | delta_exact_vs_no_directive | probe_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | visual_role_catalog_v1 | True | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog/comparison_vs_contracted/probe_comparison.json | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog/comparison_vs_no_directive/probe_comparison.json | 0.125 | 1.0 | -0.75 | 0.125 | probe_improved_vs_no_directive |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | visual_role_catalog_argument_hints_v2 | True | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints/comparison_vs_contracted/probe_comparison.json | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints/comparison_vs_no_directive/probe_comparison.json | 0.25 | 0.0 | -0.625 | 0.25 | probe_improved_vs_no_directive |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints | visual_role_catalog_split_selector_hints_v3 | True | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints/comparison_vs_contracted/probe_comparison.json | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints/comparison_vs_no_directive/probe_comparison.json | 0.125 | 0.0 | -0.75 | 0.125 | probe_improved_vs_no_directive |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | visual_role_catalog_schema_field_hints_v4 | True | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints/comparison_vs_contracted/probe_comparison.json | /Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints/comparison_vs_no_directive/probe_comparison.json | 0.25 | 0.0 | -0.625 | 0.25 | probe_improved_vs_no_directive |

`visual_role_catalog_v1` moves the intervention from standalone prompt-contract wording into the tool-catalog presentation. It keeps the exact directive disabled, improves raw exact rate from `0.0` to `0.125`, restores the visual executable target to `1.0`, and changes the live visual failure from wrong-tool/no-call into literal argument mismatch. `visual_role_catalog_argument_hints_v2` then tests the next narrow question: can field-level selector semantics fix that literal mismatch while preserving routing?

## Tool-Catalog Argument-Hints vs Role-Catalog Probe Delta

| case_id | family | baseline_exact_match | candidate_exact_match | delta_exact_match | baseline_failure_mode | candidate_failure_mode | baseline_executable_match | candidate_executable_match | delta_executable_match | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api_form_issue_fetch | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| api_invoice_lock_hold_update | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_invoice_lock_hyphen_query | cli_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_phone_patch_latest_only | cli_patch_copying | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| parallel_audit_array_literal | parallel_tool_calling | False | False | 0 | no_tool_call | no_tool_call |  |  |  | 0 | 0 | 0 |
| visual_form_target_literal | visual_argument_copying | False | False | 0 | executable_paraphrase | argument_mismatch | True | False | -1 | 1 | 1 | 0 |
| visual_latest_filter_literal | visual_referent_carryover | False | True | 1 | argument_mismatch | exact |  |  |  | 1 | 1 | 0 |
| visual_readback_region_literal | visual_referent_carryover | True | True | 0 | exact | exact |  |  |  | 1 | 1 | 0 |

The raw answer is mixed but materially informative. Argument hints raise probe exactness from `1 / 8` to `2 / 8` by making `visual_latest_filter_literal` exact, while preserving exact readback. The cost is that `visual_form_target_literal` drops from executable paraphrase to non-executable argument mismatch, so this is a candidate for visual referent exactness, not a complete visual recovery profile.

## Tool-Catalog Split-Selector Negative Probe Delta

| case_id | family | baseline_exact_match | candidate_exact_match | delta_exact_match | baseline_failure_mode | candidate_failure_mode | baseline_executable_match | candidate_executable_match | delta_executable_match | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api_form_issue_fetch | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| api_invoice_lock_hold_update | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_invoice_lock_hyphen_query | cli_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_phone_patch_latest_only | cli_patch_copying | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| parallel_audit_array_literal | parallel_tool_calling | False | False | 0 | no_tool_call | no_tool_call |  |  |  | 0 | 0 | 0 |
| visual_form_target_literal | visual_argument_copying | False | False | 0 | argument_mismatch | argument_mismatch | False | False | 0 | 1 | 1 | 0 |
| visual_latest_filter_literal | visual_referent_carryover | True | True | 0 | exact | exact |  |  |  | 1 | 1 | 0 |
| visual_readback_region_literal | visual_referent_carryover | True | False | -1 | exact | no_tool_call |  |  |  | 1 | 0 | -1 |

| case_id | family | baseline_exact_match | candidate_exact_match | delta_exact_match | baseline_failure_mode | candidate_failure_mode | baseline_executable_match | candidate_executable_match | delta_executable_match | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api_form_issue_fetch | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| api_invoice_lock_hold_update | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_invoice_lock_hyphen_query | cli_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_phone_patch_latest_only | cli_patch_copying | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| parallel_audit_array_literal | parallel_tool_calling | False | False | 0 | no_tool_call | no_tool_call |  |  |  | 0 | 0 | 0 |
| visual_form_target_literal | visual_argument_copying | False | False | 0 | executable_paraphrase | argument_mismatch | True | False | -1 | 1 | 1 | 0 |
| visual_latest_filter_literal | visual_referent_carryover | False | True | 1 | argument_mismatch | exact |  |  |  | 1 | 1 | 0 |
| visual_readback_region_literal | visual_referent_carryover | True | False | -1 | exact | no_tool_call |  |  |  | 1 | 0 | -1 |

| packet_run_id | candidate_system_id | tool_catalog_profile_id | decision | reason | candidate_exact_match_rate | candidate_executable_match_rate | best_current_exact_candidate | best_current_exact_candidate_rate | best_current_executable_routing_candidate | best_current_executable_routing_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260508T_visual_split_selector_hints_live_replay_skipped_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints | visual_role_catalog_split_selector_hints_v3 | skip_live_replay | v3 did not beat the v2 raw exact gate and lost readback exactness, so live replay would spend budget on a weaker candidate. | 0.125 | 0.0 | visual_role_catalog_argument_hints_v2 | 0.25 | visual_role_catalog_v1 | 1.0 |

`visual_role_catalog_split_selector_hints_v3` is useful as negative evidence. It preserved the v2 latest-filter exact call, but dropped overall raw exactness from `2 / 8` to `1 / 8` versus v2 and regressed readback by emitting `tool_name` instead of `name`. It also failed to recover the v1 executable form-target behavior, so focused live replay was intentionally skipped.

## Tool-Catalog Schema-Field Negative Probe Delta

| case_id | family | baseline_exact_match | candidate_exact_match | delta_exact_match | baseline_failure_mode | candidate_failure_mode | baseline_executable_match | candidate_executable_match | delta_executable_match | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api_form_issue_fetch | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| api_invoice_lock_hold_update | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_invoice_lock_hyphen_query | cli_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_phone_patch_latest_only | cli_patch_copying | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| parallel_audit_array_literal | parallel_tool_calling | False | False | 0 | no_tool_call | no_tool_call |  |  |  | 0 | 0 | 0 |
| visual_form_target_literal | visual_argument_copying | False | False | 0 | argument_mismatch | wrong_tool | False | False | 0 | 1 | 1 | 0 |
| visual_latest_filter_literal | visual_referent_carryover | True | True | 0 | exact | exact |  |  |  | 1 | 1 | 0 |
| visual_readback_region_literal | visual_referent_carryover | True | True | 0 | exact | exact |  |  |  | 1 | 1 | 0 |

| case_id | family | baseline_exact_match | candidate_exact_match | delta_exact_match | baseline_failure_mode | candidate_failure_mode | baseline_executable_match | candidate_executable_match | delta_executable_match | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api_form_issue_fetch | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| api_invoice_lock_hold_update | api_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_invoice_lock_hyphen_query | cli_canonicalization | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| cli_phone_patch_latest_only | cli_patch_copying | False | False | 0 | argument_mismatch | argument_mismatch |  |  |  | 1 | 1 | 0 |
| parallel_audit_array_literal | parallel_tool_calling | False | False | 0 | no_tool_call | no_tool_call |  |  |  | 0 | 0 | 0 |
| visual_form_target_literal | visual_argument_copying | False | False | 0 | argument_mismatch | wrong_tool | False | False | 0 | 1 | 1 | 0 |
| visual_latest_filter_literal | visual_referent_carryover | True | True | 0 | exact | exact |  |  |  | 1 | 1 | 0 |
| visual_readback_region_literal | visual_referent_carryover | False | True | 1 | no_tool_call | exact |  |  |  | 0 | 1 | 1 |

| packet_run_id | candidate_system_id | tool_catalog_profile_id | decision | reason | candidate_exact_match_rate | candidate_executable_match_rate | best_current_exact_candidate | best_current_exact_candidate_rate | best_current_executable_routing_candidate | best_current_executable_routing_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260509T_visual_schema_field_hints_live_replay_skipped_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | visual_role_catalog_schema_field_hints_v4 | skip_live_replay | v4 tied the v2 raw exact rate but did not improve it, failed executable form targeting, and over-preferred refine_selection on the form-target case. | 0.25 | 0.0 | visual_role_catalog_argument_hints_v2 | 0.25 | visual_role_catalog_v1 | 1.0 |

`visual_role_catalog_schema_field_hints_v4` is cleaner than v3 because it avoids broad prose and restores the exact readback case. It still does not beat v2: raw exact stays `2 / 8`, executable visual-form recovery stays `0 / 1`, and the form-target case over-prefers `refine_selection` with `selection_id="latest"`. Live replay was skipped because it tied the current best exact candidate while remaining below the executable routing baseline.

## Prompt-Contract Wave Six Probe Gate

| system_id | tool_prompt_contract_id | tool_catalog_profile_id | exact_match_rate | executable_match_rate | delta_exact_vs_contracted | delta_exact_vs_no_directive | probe_gate | improved_case_count | regressed_case_count | dominant_failure_mode | failure_modes | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_literal_guard | literal_argument_guard_v1 | visual_role_catalog_v1 | 0.125 | 0.0 | -0.75 | 0.125 | probe_improved_vs_no_directive | 1 | 0 | argument_mismatch | argument_mismatch:4;exact:1;no_tool_call:3 | weak_exact_gain |

Wave six composes the visual role catalog with `literal_argument_guard_v1`. It keeps the same one-case exact gain but loses the catalog-only executable visual rescue and introduces no-call regressions on CLI/API cases. Treat it as a negative composition result: routing guidance and literal-copy wording interfere in this form.

## Prompt-Contract Promotion Decisions

| wave | tool_prompt_contract_id | tool_catalog_profile_id | exact_match_rate | executable_match_rate | delta_exact_vs_no_directive | probe_gate | recommendation | promotion_decision | promotion_reason | next_use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | schema_anchor_v1 |  | 0.125 | 0.0 | 0.125 | probe_improved_vs_no_directive | weak_exact_gain | hold_for_exact_probe_replay | probe gain is too weak for H1 promotion without a stricter replay discriminator | test through exact-probe live replay before any H1 spend |
| v1 | literal_argument_guard_v1 |  | 0.0 | 1.0 | 0.0 | probe_improved_vs_no_directive | visual_executable_gain_only | hold_for_exact_probe_replay | executable recovery exists, but exact JSON/tool-call fidelity did not improve | use in visual replay only, not as a general H1 candidate |
| v1 | tool_required_parallel_v1 |  | 0.0 | 1.0 | 0.0 | probe_improved_vs_no_directive | visual_executable_gain_only | hold_for_exact_probe_replay | executable recovery exists, but exact JSON/tool-call fidelity did not improve | use in visual replay only, not as a general H1 candidate |
| v2 | schema_literal_tool_required_v2 |  | 0.125 | 0.0 | 0.125 | probe_improved_vs_no_directive | weak_exact_gain | hold_for_exact_probe_replay | probe gain is too weak for H1 promotion without a stricter replay discriminator | test through exact-probe live replay before any H1 spend |
| v2 | visual_next_call_state_v2 |  | 0.0 | 1.0 | 0.0 | probe_improved_vs_no_directive | visual_executable_gain_only | hold_for_exact_probe_replay | executable recovery exists, but exact JSON/tool-call fidelity did not improve | use in visual replay only, not as a general H1 candidate |
| v2 | parallel_array_required_v2 |  | 0.0 | 0.0 | 0.0 | no_probe_improvement_vs_no_directive | no_probe_gain | reject_for_h1_promotion | no exact or executable probe gain over the no-directive baseline | replace with a sharper contract or a faithful live parallel workflow |
| v3 | canonical_json_copy_v3 |  | 0.125 | 0.0 | 0.125 | probe_improved_vs_no_directive | weak_exact_gain | hold_for_exact_probe_replay | probe gain is too weak for H1 promotion without a stricter replay discriminator | test through exact-probe live replay before any H1 spend |
| v3 | visual_tool_initiation_v3 |  | 0.125 | 1.0 | 0.125 | probe_improved_vs_no_directive | weak_exact_gain | hold_for_exact_probe_replay | probe gain is too weak for H1 promotion without a stricter replay discriminator | test through exact-probe live replay before any H1 spend |
| v3 | parallel_two_call_array_v3 |  | 0.0 | 0.0 | 0.0 | no_probe_improvement_vs_no_directive | no_probe_gain | reject_for_h1_promotion | no exact or executable probe gain over the no-directive baseline | replace with a sharper contract or a faithful live parallel workflow |
| v4 | visual_state_tool_selection_v4 |  | 0.125 | 0.0 | 0.125 | probe_improved_vs_no_directive | weak_exact_gain | hold_for_exact_probe_replay | probe gain is too weak for H1 promotion without a stricter replay discriminator | test through exact-probe live replay before any H1 spend |
| v5 | visual_refine_selection_v5 |  | 0.0 | 0.0 | 0.0 | no_probe_improvement_vs_no_directive | no_probe_gain | reject_for_h1_promotion | no exact or executable probe gain over the no-directive baseline | replace with a sharper contract or a faithful live parallel workflow |
| v6 | literal_argument_guard_v1 | visual_role_catalog_v1 | 0.125 | 0.0 | 0.125 | probe_improved_vs_no_directive | weak_exact_gain | hold_for_exact_probe_replay | probe gain is too weak for H1 promotion without a stricter replay discriminator | test through exact-probe live replay before any H1 spend |

The promotion gate is intentionally conservative: weak one-case exact gains and visual executable-only gains are held for exact-probe replay, while candidates with no probe gain are rejected for H1 promotion.

## Exact-Probe Replay Comparison

- Baseline exact rate: `0.875`
- Candidate exact rate: `0.0`
- Delta exact rate: `-0.875`

| case_id | family | baseline_failure_mode | candidate_failure_mode | baseline_exact_match | candidate_exact_match | delta_exact_match | baseline_executable_match | candidate_executable_match | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api_form_issue_fetch | api_canonicalization | exact | argument_mismatch | True | False | -1 |  |  | 1 | 1 | 0 |
| api_invoice_lock_hold_update | api_canonicalization | exact | argument_mismatch | True | False | -1 |  |  | 1 | 1 | 0 |
| cli_invoice_lock_hyphen_query | cli_canonicalization | exact | argument_mismatch | True | False | -1 |  |  | 1 | 1 | 0 |
| cli_phone_patch_latest_only | cli_patch_copying | exact | argument_mismatch | True | False | -1 |  |  | 1 | 1 | 0 |
| parallel_audit_array_literal | parallel_tool_calling | exact | no_tool_call | True | False | -1 |  |  | 2 | 0 | -2 |
| visual_form_target_literal | visual_argument_copying | executable_paraphrase | no_tool_call | False | False | 0 | True | False | 1 | 0 | -1 |
| visual_latest_filter_literal | visual_referent_carryover | exact | no_tool_call | True | False | -1 |  |  | 1 | 0 | -1 |
| visual_readback_region_literal | visual_referent_carryover | exact | no_tool_call | True | False | -1 |  |  | 1 | 0 | -1 |

## Focused Exact-Replay Slices

| slice | shared_case_count | baseline_exact_match_rate | candidate_exact_match_rate | delta_exact_match_rate | case_delta_count |
| --- | --- | --- | --- | --- | --- |
| all failures | 8 | 0.875 | 0.0 | -0.875 | 7 |
| canonical arguments | 4 | 1.0 | 0.0 | -1.0 | 4 |
| visual no-call | 3 | 0.6666666666666666 | 0.0 | -0.6666666666666666 | 2 |
| parallel array | 1 | 1.0 | 0.0 | -1.0 | 1 |

## CLI-Live Parallel Replay Comparison

- Contracted exact rate: `1.0`
- No-directive exact rate: `0.0`
- Delta exact rate: `-1.0`

| case_id | family | source_failure_mode | baseline_replay_exact_match | candidate_replay_exact_match | delta_exact_match | baseline_replay_executable_match | candidate_replay_executable_match | delta_executable_match | baseline_replay_failure_mode | candidate_replay_failure_mode | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parallel_audit_array_literal | parallel_tool_calling | no_tool_call | True | False | -1 |  |  |  | exact | no_tool_call | 2 | 0 | -2 |

This is the live-operator counterpart to the focused parallel-array replay. The contracted row emits both expected tool calls, while the no-directive row emits no tool calls and asks the operator to provide inputs that were already present in the replay context.

- Visual contracted exact rate: `0.6666666666666666`
- Visual no-directive exact rate: `0.0`
- Visual delta exact rate: `-0.6666666666666666`

| case_id | family | source_failure_mode | baseline_replay_exact_match | candidate_replay_exact_match | delta_exact_match | baseline_replay_executable_match | candidate_replay_executable_match | delta_executable_match | baseline_replay_failure_mode | candidate_replay_failure_mode | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | True | False | -1 | executable_paraphrase | no_tool_call | 1 | 0 | -1 |
| visual_latest_filter_literal | visual_referent_carryover | no_tool_call | True | False | -1 |  |  |  | exact | no_tool_call | 1 | 0 | -1 |
| visual_readback_region_literal | visual_referent_carryover | no_tool_call | True | False | -1 |  |  |  | exact | no_tool_call | 1 | 0 | -1 |

The visual CLI-live comparison mirrors the focused visual replay: no-directive emits no tool calls in all three cases, while contracted MLX recovers two exact calls and one executable visual paraphrase.

- Canonical contracted exact rate: `1.0`
- Canonical no-directive exact rate: `0.0`
- Canonical delta exact rate: `-1.0`

| case_id | family | source_failure_mode | baseline_replay_exact_match | candidate_replay_exact_match | delta_exact_match | baseline_replay_executable_match | candidate_replay_executable_match | delta_executable_match | baseline_replay_failure_mode | candidate_replay_failure_mode | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api_form_issue_fetch | api_canonicalization | argument_mismatch | True | False | -1 |  |  |  | exact | argument_mismatch | 1 | 1 | 0 |
| api_invoice_lock_hold_update | api_canonicalization | argument_mismatch | True | False | -1 |  |  |  | exact | argument_mismatch | 1 | 1 | 0 |
| cli_invoice_lock_hyphen_query | cli_canonicalization | argument_mismatch | True | False | -1 |  |  |  | exact | argument_mismatch | 1 | 1 | 0 |
| cli_phone_patch_latest_only | cli_patch_copying | argument_mismatch | True | False | -1 |  |  |  | exact | argument_mismatch | 1 | 1 | 0 |

The canonical CLI/API comparison isolates argument fidelity: both rows emit one tool call per case, but no-directive misses canonical paths, ids, or query strings in all four cases.

## CLI-Live Focused Replay Summary

| slice | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | case_delta_count |
| --- | --- | --- | --- | --- | --- |
| canonical arguments | 4 | 1.0 | 0.0 | -1.0 | 4 |
| parallel array | 1 | 1.0 | 0.0 | -1.0 | 1 |
| visual no-call | 3 | 0.6666666666666666 | 0.0 | -0.6666666666666666 | 3 |

## Wave Three CLI-Live Candidate Replay

| comparison | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executable_rate | candidate_executable_rate | delta_executable_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical JSON vs no directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_canonical_json_copy | 4 | 0.0 | 0.0 | 0.0 |  |  |  |
| canonical JSON vs contracted | mlx_gemma4_e2b_reasoner_only | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_canonical_json_copy | 4 | 1.0 | 0.0 | -1.0 |  |  |  |
| visual initiation vs no directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation | 3 | 0.0 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 1.0 | 1.0 |
| visual initiation vs contracted | mlx_gemma4_e2b_reasoner_only | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation | 3 | 0.6666666666666666 | 0.3333333333333333 | -0.3333333333333333 | 1.0 | 1.0 | 0.0 |

The live replay gate rejects `canonical_json_copy_v3` for canonical argument promotion: exact rate stays `0.0` against no-directive and two cases regress from argument mismatch to no tool call. `visual_tool_initiation_v3` is the first candidate with live family movement: it improves visual exact rate from `0.0` to `0.3333333333333333`, restores the executable visual-form target, and emits one tool call in all three visual cases. It remains below contracted MLX because one visual referent case still uses the wrong visual tool.

| comparison | case_id | family | source_failure_mode | baseline_replay_exact_match | candidate_replay_exact_match | delta_exact_match | baseline_replay_executable_match | candidate_replay_executable_match | delta_executable_match | baseline_replay_failure_mode | candidate_replay_failure_mode | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical JSON vs no directive | api_form_issue_fetch | api_canonicalization | argument_mismatch | False | False | 0 | None | None | None | argument_mismatch | argument_mismatch | 1 | 1 | 0 |
| canonical JSON vs no directive | api_invoice_lock_hold_update | api_canonicalization | argument_mismatch | False | False | 0 | None | None | None | argument_mismatch | no_tool_call | 1 | 0 | -1 |
| canonical JSON vs no directive | cli_invoice_lock_hyphen_query | cli_canonicalization | argument_mismatch | False | False | 0 | None | None | None | argument_mismatch | argument_mismatch | 1 | 1 | 0 |
| canonical JSON vs no directive | cli_phone_patch_latest_only | cli_patch_copying | argument_mismatch | False | False | 0 | None | None | None | argument_mismatch | no_tool_call | 1 | 0 | -1 |
| canonical JSON vs contracted | api_form_issue_fetch | api_canonicalization | argument_mismatch | True | False | -1 | None | None | None | exact | argument_mismatch | 1 | 1 | 0 |
| canonical JSON vs contracted | api_invoice_lock_hold_update | api_canonicalization | argument_mismatch | True | False | -1 | None | None | None | exact | no_tool_call | 1 | 0 | -1 |
| canonical JSON vs contracted | cli_invoice_lock_hyphen_query | cli_canonicalization | argument_mismatch | True | False | -1 | None | None | None | exact | argument_mismatch | 1 | 1 | 0 |
| canonical JSON vs contracted | cli_phone_patch_latest_only | cli_patch_copying | argument_mismatch | True | False | -1 | None | None | None | exact | no_tool_call | 1 | 0 | -1 |
| visual initiation vs no directive | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | False | True | 1 | no_tool_call | executable_paraphrase | 0 | 1 | 1 |
| visual initiation vs no directive | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | False | False | 0 | None | None | None | no_tool_call | wrong_tool | 0 | 1 | 1 |
| visual initiation vs no directive | visual_readback_region_literal | visual_referent_carryover | no_tool_call | False | True | 1 | None | None | None | no_tool_call | exact | 0 | 1 | 1 |
| visual initiation vs contracted | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | True | True | 0 | executable_paraphrase | executable_paraphrase | 1 | 1 | 0 |
| visual initiation vs contracted | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | True | False | -1 | None | None | None | exact | wrong_tool | 1 | 1 | 0 |
| visual initiation vs contracted | visual_readback_region_literal | visual_referent_carryover | no_tool_call | True | True | 0 | None | None | None | exact | exact | 1 | 1 | 0 |

## Wave Four CLI-Live Candidate Replay

| comparison | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executable_rate | candidate_executable_rate | delta_executable_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual state tool selection vs no directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection | 3 | 0.0 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 0.0 | 0.0 |
| visual state tool selection vs contracted | mlx_gemma4_e2b_reasoner_only | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection | 3 | 0.6666666666666666 | 0.3333333333333333 | -0.3333333333333333 | 1.0 | 0.0 | -1.0 |

`visual_state_tool_selection_v4` keeps the same exact live ceiling as wave three, not a promotion path. It improves over no-directive from `0 / 3` to `1 / 3`, but trails contracted MLX at `2 / 3`, loses executable visual-form recovery, and still fails `visual_latest_filter_literal` with the wrong visual tool. This is useful negative evidence: adding state/tool-selection wording did not fix the remaining visual referent failure.

| comparison | case_id | family | source_failure_mode | baseline_replay_exact_match | candidate_replay_exact_match | delta_exact_match | baseline_replay_executable_match | candidate_replay_executable_match | delta_executable_match | baseline_replay_failure_mode | candidate_replay_failure_mode | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual state tool selection vs no directive | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | False | False | 0 | no_tool_call | no_tool_call | 0 | 0 | 0 |
| visual state tool selection vs no directive | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | False | False | 0 | None | None | None | no_tool_call | wrong_tool | 0 | 1 | 1 |
| visual state tool selection vs no directive | visual_readback_region_literal | visual_referent_carryover | no_tool_call | False | True | 1 | None | None | None | no_tool_call | exact | 0 | 1 | 1 |
| visual state tool selection vs contracted | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | True | False | -1 | executable_paraphrase | no_tool_call | 1 | 0 | -1 |
| visual state tool selection vs contracted | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | True | False | -1 | None | None | None | exact | wrong_tool | 1 | 1 | 0 |
| visual state tool selection vs contracted | visual_readback_region_literal | visual_referent_carryover | no_tool_call | True | True | 0 | None | None | None | exact | exact | 1 | 1 | 0 |

## Visual Catalog CLI-Live Candidate Replay

| comparison | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executable_rate | candidate_executable_rate | delta_executable_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual role catalog vs no directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | 3 | 0.0 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 1.0 | 1.0 |
| visual role catalog vs visual initiation | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | 3 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 1.0 | 1.0 | 0.0 |
| visual role catalog vs visual state tool | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | 3 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 0.0 | 1.0 | 1.0 |

`visual_role_catalog_v1` matches wave three's `1 / 3` exact ceiling, beats wave four on executable visual-form recovery, and converts the remaining latest-filter failure from `wrong_tool` to `argument_mismatch`. The next useful move is not more broad visual state wording; it is a narrow argument-literal mechanism that preserves the catalog routing win.

| comparison | case_id | family | source_failure_mode | baseline_replay_exact_match | candidate_replay_exact_match | delta_exact_match | baseline_replay_executable_match | candidate_replay_executable_match | delta_executable_match | baseline_replay_failure_mode | candidate_replay_failure_mode | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual role catalog vs no directive | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | False | True | 1 | no_tool_call | executable_paraphrase | 0 | 1 | 1 |
| visual role catalog vs no directive | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | False | False | 0 | None | None | None | no_tool_call | argument_mismatch | 0 | 1 | 1 |
| visual role catalog vs no directive | visual_readback_region_literal | visual_referent_carryover | no_tool_call | False | True | 1 | None | None | None | no_tool_call | exact | 0 | 1 | 1 |
| visual role catalog vs visual initiation | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | True | True | 0 | executable_paraphrase | executable_paraphrase | 1 | 1 | 0 |
| visual role catalog vs visual initiation | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | False | False | 0 | None | None | None | wrong_tool | argument_mismatch | 1 | 1 | 0 |
| visual role catalog vs visual initiation | visual_readback_region_literal | visual_referent_carryover | no_tool_call | True | True | 0 | None | None | None | exact | exact | 1 | 1 | 0 |
| visual role catalog vs visual state tool | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | False | True | 1 | no_tool_call | executable_paraphrase | 0 | 1 | 1 |
| visual role catalog vs visual state tool | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | False | False | 0 | None | None | None | wrong_tool | argument_mismatch | 1 | 1 | 0 |
| visual role catalog vs visual state tool | visual_readback_region_literal | visual_referent_carryover | no_tool_call | True | True | 0 | None | None | None | exact | exact | 1 | 1 | 0 |

## Visual Catalog Argument-Hints CLI-Live Candidate Replay

| comparison | baseline_system_id | candidate_system_id | shared_case_count | baseline_exact_rate | candidate_exact_rate | delta_exact_rate | baseline_executable_rate | candidate_executable_rate | delta_executable_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual argument hints vs no directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 3 | 0.0 | 0.6666666666666666 | 0.6666666666666666 | 0.0 | 0.0 | 0.0 |
| visual argument hints vs contracted | mlx_gemma4_e2b_reasoner_only | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 3 | 0.6666666666666666 | 0.6666666666666666 | 0.0 | 1.0 | 0.0 | -1.0 |
| visual argument hints vs role catalog | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 3 | 0.3333333333333333 | 0.6666666666666666 | 0.3333333333333333 | 1.0 | 0.0 | -1.0 |

`visual_role_catalog_argument_hints_v2` is the first no-directive candidate to match contracted MLX on this focused visual exact replay: `2 / 3` exact. It fixes `visual_latest_filter_literal` exactly and preserves exact readback. The remaining gap is important: the candidate loses the contracted/v1 executable visual-form rescue, turning `visual_form_target_literal` into a non-executable argument mismatch. This is progress on selector literalness, but not yet a full replacement for controller-backed visual recovery.

| comparison | case_id | family | source_failure_mode | baseline_replay_exact_match | candidate_replay_exact_match | delta_exact_match | baseline_replay_executable_match | candidate_replay_executable_match | delta_executable_match | baseline_replay_failure_mode | candidate_replay_failure_mode | baseline_actual_call_count | candidate_actual_call_count | delta_actual_call_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual argument hints vs no directive | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | False | False | 0 | no_tool_call | argument_mismatch | 0 | 1 | 1 |
| visual argument hints vs no directive | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | False | True | 1 | None | None | None | no_tool_call | exact | 0 | 1 | 1 |
| visual argument hints vs no directive | visual_readback_region_literal | visual_referent_carryover | no_tool_call | False | True | 1 | None | None | None | no_tool_call | exact | 0 | 1 | 1 |
| visual argument hints vs contracted | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | True | False | -1 | executable_paraphrase | argument_mismatch | 1 | 1 | 0 |
| visual argument hints vs contracted | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | True | True | 0 | None | None | None | exact | exact | 1 | 1 | 0 |
| visual argument hints vs contracted | visual_readback_region_literal | visual_referent_carryover | no_tool_call | True | True | 0 | None | None | None | exact | exact | 1 | 1 | 0 |
| visual argument hints vs role catalog | visual_form_target_literal | visual_argument_copying | no_tool_call | False | False | 0 | True | False | -1 | executable_paraphrase | argument_mismatch | 1 | 1 | 0 |
| visual argument hints vs role catalog | visual_latest_filter_literal | visual_referent_carryover | no_tool_call | False | True | 1 | None | None | None | argument_mismatch | exact | 1 | 1 | 0 |
| visual argument hints vs role catalog | visual_readback_region_literal | visual_referent_carryover | no_tool_call | True | True | 0 | None | None | None | exact | exact | 1 | 1 | 0 |

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

## H1j Probe-Derived Candidate Packet

| system_id | lane | disabled_controls | tool_turn_directive_enabled | real_world_readiness_avg | delta_vs_contracted_real_world_readiness_avg | delta_vs_no_directive_real_world_readiness_avg | strict_interface_avg | delta_vs_contracted_strict_interface_avg | delta_vs_no_directive_strict_interface_avg | recovered_execution_avg | delta_vs_contracted_recovered_execution_avg | delta_vs_no_directive_recovered_execution_avg | controller_repair_avg | delta_vs_contracted_controller_repair_avg | delta_vs_no_directive_controller_repair_avg | argument_repair_avg | delta_vs_contracted_argument_repair_avg | delta_vs_no_directive_argument_repair_avg | controller_fallback_avg | delta_vs_contracted_controller_fallback_avg | delta_vs_no_directive_controller_fallback_avg | intent_override_avg | delta_vs_contracted_intent_override_avg | delta_vs_no_directive_intent_override_avg | raw_planning_clean_rate_avg | delta_vs_contracted_raw_planning_clean_rate_avg | delta_vs_no_directive_raw_planning_clean_rate_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only | live_web_stress |  | True | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | live_web_stress | disable_tool_turn_directive | False | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |

H1j maps the no-directive probe failures back into six packaged live workflow families. This first candidate packet is also saturated: contracted, no-directive, and all three candidate rows remain controller-clean with raw clean rate `1.0`. That widens the evidence that benchmark-style packaged workflows are easier than the raw tool-contract probe, even when selected from the same failure families.

## H1j Probe-Derived Helper Packet

| system_id | lane | disabled_controls | tool_turn_directive_enabled | real_world_readiness_avg | delta_vs_contracted_real_world_readiness_avg | delta_vs_no_directive_real_world_readiness_avg | strict_interface_avg | delta_vs_contracted_strict_interface_avg | delta_vs_no_directive_strict_interface_avg | recovered_execution_avg | delta_vs_contracted_recovered_execution_avg | delta_vs_no_directive_recovered_execution_avg | controller_repair_avg | delta_vs_contracted_controller_repair_avg | delta_vs_no_directive_controller_repair_avg | argument_repair_avg | delta_vs_contracted_argument_repair_avg | delta_vs_no_directive_argument_repair_avg | controller_fallback_avg | delta_vs_contracted_controller_fallback_avg | delta_vs_no_directive_controller_fallback_avg | intent_override_avg | delta_vs_contracted_intent_override_avg | delta_vs_no_directive_intent_override_avg | raw_planning_clean_rate_avg | delta_vs_contracted_raw_planning_clean_rate_avg | delta_vs_no_directive_raw_planning_clean_rate_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only | live_web_stress |  | True | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | live_web_stress | disable_tool_turn_directive | False | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair | live_web_stress | disable_argument_repair;disable_tool_turn_directive | False | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback | live_web_stress | disable_controller_fallback;disable_tool_turn_directive | False | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair | live_web_stress | disable_controller_repair;disable_tool_turn_directive | False | 0.9657666666666667 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |

The H1j helper-ablation packet is saturated too. Removing controller repair, controller fallback, or argument repair does not change readiness, strict interface, recovered execution, or raw clean rate on this probe-derived packaged workflow set. The trace miner records disabled-helper markers, but no failure candidates.

## H1k Parallel-Audit Candidate Packet

| system_id | lane | disabled_controls | tool_turn_directive_enabled | real_world_readiness_avg | delta_vs_contracted_real_world_readiness_avg | delta_vs_no_directive_real_world_readiness_avg | strict_interface_avg | delta_vs_contracted_strict_interface_avg | delta_vs_no_directive_strict_interface_avg | recovered_execution_avg | delta_vs_contracted_recovered_execution_avg | delta_vs_no_directive_recovered_execution_avg | controller_repair_avg | delta_vs_contracted_controller_repair_avg | delta_vs_no_directive_controller_repair_avg | argument_repair_avg | delta_vs_contracted_argument_repair_avg | delta_vs_no_directive_argument_repair_avg | controller_fallback_avg | delta_vs_contracted_controller_fallback_avg | delta_vs_no_directive_controller_fallback_avg | intent_override_avg | delta_vs_contracted_intent_override_avg | delta_vs_no_directive_intent_override_avg | raw_planning_clean_rate_avg | delta_vs_contracted_raw_planning_clean_rate_avg | delta_vs_no_directive_raw_planning_clean_rate_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only | live_web_stress |  | True | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | live_web_stress | disable_tool_turn_directive | False | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required | live_web_stress | disable_tool_turn_directive;tool_prompt_contract_id | False | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |

H1k promotes the deferred `parallel_audit_array_literal` probe pressure into one packaged live workflow, `ops_parallel_audit_review`. The candidate packet is still saturated: the contracted row, no-directive row, and prompt-contract candidates all match readiness `0.91780`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, and zero controller burden.

## H1k Parallel-Audit Helper Packet

| system_id | lane | disabled_controls | tool_turn_directive_enabled | real_world_readiness_avg | delta_vs_contracted_real_world_readiness_avg | delta_vs_no_directive_real_world_readiness_avg | strict_interface_avg | delta_vs_contracted_strict_interface_avg | delta_vs_no_directive_strict_interface_avg | recovered_execution_avg | delta_vs_contracted_recovered_execution_avg | delta_vs_no_directive_recovered_execution_avg | controller_repair_avg | delta_vs_contracted_controller_repair_avg | delta_vs_no_directive_controller_repair_avg | argument_repair_avg | delta_vs_contracted_argument_repair_avg | delta_vs_no_directive_argument_repair_avg | controller_fallback_avg | delta_vs_contracted_controller_fallback_avg | delta_vs_no_directive_controller_fallback_avg | intent_override_avg | delta_vs_contracted_intent_override_avg | delta_vs_no_directive_intent_override_avg | raw_planning_clean_rate_avg | delta_vs_contracted_raw_planning_clean_rate_avg | delta_vs_no_directive_raw_planning_clean_rate_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx_gemma4_e2b_reasoner_only | live_web_stress |  | True | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | live_web_stress | disable_tool_turn_directive | False | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair | live_web_stress | disable_argument_repair;disable_tool_turn_directive | False | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback | live_web_stress | disable_controller_fallback;disable_tool_turn_directive | False | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair | live_web_stress | disable_controller_repair;disable_tool_turn_directive | False | 0.9178 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |

The H1k helper packet confirms the negative result. Removing controller repair, controller fallback, or argument repair does not move the staged parallel-audit workflow. The result is useful because it narrows the next experiment: the discriminator must preserve exact one-turn replay shape instead of further decomposing the parallel task into staged packaged steps.

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
- The visual catalog path now gives a sharper positive result than the prompt-contract path: argument-hints cataloging reaches `2 / 3` live exact visual replay without the exact directive, but still misses executable form-target recovery.
- The next experiment should preserve the argument-hints selector win while separately recovering form-target executability before spending H1 budget.

## Source Artifacts

- H1f compact: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1`
- H1h full: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1`
- H1i worst-family: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1`
- Probe comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1/probe_comparison.json`
- Prompt-contract probe packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1`
- Prompt-contract wave two packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1`
- Prompt-contract wave three packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1`
- Prompt-contract wave four packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1`
- Prompt-contract wave five packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1`
- Tool catalog profile packet: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe`
- Tool catalog argument-hints packet: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe`
- Tool catalog argument-hints vs role-catalog comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_comparisons/20260508T_visual_argument_hints_vs_role_catalog_v1`
- Tool catalog split-selector packet: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe`
- Tool catalog split-selector vs argument-hints comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2`
- Tool catalog split-selector vs role-catalog comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_role_catalog_v1`
- Tool catalog split-selector live decision: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1`
- Tool catalog schema-field packet: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe`
- Tool catalog schema-field vs argument-hints comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2`
- Tool catalog schema-field vs split-selector comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_split_selector_v3`
- Tool catalog schema-field vs role-catalog comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_role_catalog_v1`
- Tool catalog schema-field live decision: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1`
- Prompt-contract wave six packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe`
- H1i prompt-contract packet: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet`
- H1i prompt-contract repeat packet: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet`
- H1j probe-derived prompt-contract packet: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet`
- H1j probe-derived helper packet: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet`
- H1k parallel-audit prompt-contract packet: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet`
- H1k parallel-audit helper packet: `/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet`
- Exact replay comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1`
- Canonical argument replay comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_v1`
- Visual replay comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_comparisons/20260507T_visual_state_contracted_vs_no_directive_v1`
- Parallel replay comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_comparisons/20260507T_parallel_array_contracted_vs_no_directive_v1`
- CLI-live parallel replay comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1`
- CLI-live visual replay comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1`
- CLI-live canonical replay comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1`
- Wave four live visual vs no-directive comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1`
- Wave four live visual vs contracted comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1`
- Argument-hints live visual vs no-directive comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1`
- Argument-hints live visual vs contracted comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1`
- Argument-hints live visual vs role-catalog comparison: `/Users/cheickdiakite/Codex/moonie/results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1`
- Gemini dry-run baseline: `/Users/cheickdiakite/Codex/moonie/results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1`
