# H3b/H4 Saturation-Breaker Design

## Why this exists

H3a now closes the fresh H3 packet and preserves the broad H2w-era transfer battery. The next benchmark should therefore stop proving the same surface is saturated and instead create harder evidence about controller generalization, stateful operation, approval boundaries, and live CLI execution.

The design follows the standard implied by frontier agent benchmark tables: every score is grouped by a named benchmark family, but Moonie keeps an extra attribution layer for controller dependence.

## Manifest

- Planned families: `6`
- Planned cases: `24`
- Score metrics: `6`
- Current candidate: `mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined`
- First execution packet: `h3b_saturation_breaker_v27`

## Families

| family_id | planned_cases | pressure_type | paper_role |
| --- | --- | --- | --- |
| h3b_unseen_stale_origin_paraphrase | 4 | controller_generalization | tests whether stale-selection repair learned a mechanism or memorized marker phrasing |
| h3b_extended_negative_value_vocabulary | 4 | semantic_generalization | tests whether value preservation is lexical coverage or general target binding |
| h3b_state_order_flip | 4 | order_sensitivity | separates target binding from row-order and first-match bias |
| h3b_current_selection_stepwise_refine | 4 | workflow_state | checks that stale-selection repair does not damage valid stateful CLI operation |
| h4_latest_instruction_retention | 4 | instruction_order | links local harness evidence to direction-following and UI-control benchmark claims |
| h4_approval_stop_boundary | 4 | operator_safety | moves from exact tool-call replay toward publishable live-agent safety evidence |

## Score Contract

| metric_id | required | definition | reporting_grain |
| --- | --- | --- | --- |
| strict_exact | True | actual tool call array exactly equals the oracle call array | overall, family, case, comparison |
| executor_equivalence | True | actual execution reaches the expected target region or session state | overall, family, case, comparison |
| controller_trace | True | controller helper metadata attached to each repaired or preserved call | helper kind, case, family, profile |
| regression_count | True | candidate misses a case that the baseline passed | comparison, family, case |
| helper_overtrigger | True | new helper fires outside its intended family or on transfer rows where it should be silent | helper kind, family, transfer packet |
| live_operator_artifact | True | CLI Rich/live run leaves inspectable manifest, summary, commands, case states, and run outputs | session, workflow family, replay packet |

## Baselines

| baseline_label | system_id | role |
| --- | --- | --- |
| h2w_semantic_target_preservation | mlx_gemma4_e2b_reasoner_only_h2w_no_controller_fallback | strong pre-H3a transfer/back-compat reference |
| h2z_boundary_combined | mlx_gemma4_e2b_reasoner_only_h2z_boundary_combined | shows whether H3a repairs are still needed on the new packet |
| h3a_combined | mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined | current candidate to break before adding new helpers |
| h3a_no_fallback | mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined_no_controller_fallback | planned ablation row for fallback dependence once the registry row exists |
| gemini_cli_external_reference | gemini_cli_external_baseline | non-replacement external operator reference for packaged workflow behavior |

## External Benchmark Alignment

| external_benchmark | benchmark_group | moonie_mapping | claim_boundary |
| --- | --- | --- | --- |
| Terminal-bench style | Coding / terminal agency | CLI packaged workflow runs, replay-live commands, sandbox manifests, and session event traces | Moonie reports local Gemma harnessing quality, not Terminal-bench leaderboard parity |
| Toolathlon style | Agentic tool use | strict tool-call, executor-equivalence, helper-causal ablations, and regression rows | Tool-use claims require helper attribution and no hidden controller-credit collapse |
| OSWorld-Verified style | UI control | visual executor target contracts, stale/current selection state, and live CLI operator evidence | Local visual replay is a controlled UI-substrate proxy, not full desktop OS control |
| SWE-Bench / Terminal repair style | Recovered execution | repaired versus strict correctness, fallback-causal deltas, and output usability records | Recovered success is separated from strict success in every table |
| Long-context direction-following style | Instruction retention | latest-instruction override cases with prior tool state and stale provenance decoys | Claims are limited to packaged workflow/replay contexts until broader long-context runs exist |

## Decision Gate

Score H3a before adding new helpers; promote no new controller unless it fixes a named family, has clean transfer/back-compat evidence, and leaves helper traces attributable.

Every top-line score must be paired with controller-dependence, executor-equivalence, regression, family, and live artifact evidence.
