# MLX Tool-Contract Harnessing Report

This is the current human-readable report for the local Gemma-on-MLX tool-contract research wave. The generated packet that backs this document lives at:

- [`results/reports/mlx_tool_contract_harnessing/report.md`](../../results/reports/mlx_tool_contract_harnessing/report.md)
- [`results/reports/mlx_tool_contract_harnessing/report.json`](../../results/reports/mlx_tool_contract_harnessing/report.json)
- [`results/reports/mlx_tool_contract_harnessing/manifest.json`](../../results/reports/mlx_tool_contract_harnessing/manifest.json)

Regenerate it with:

```bash
uv run python scripts/build_mlx_tool_contract_report.py
```

## Executive Read

The current Moonie frontier is no longer whether `mlx_gemma4_e2b_reasoner_only` can match top-line readiness on the aligned `32 / 26` comparison surface. That surface is partially saturated. The active research question is now narrower and more useful:

> Which harness interventions make local MLX Gemma stay inside the tool interface without controller repair, fallback, or argument normalization?

The newest evidence says the final tool-turn directive is a causal model-side harness intervention. With the directive present, MLX Gemma is controller-clean on the current H1 live workflow packets. When the directive is removed, no-directive MLX can still match top-line readiness only because Moonie's controller repairs arguments, falls back to a planner, or substitutes the correct call. Raw no-directive tool compliance collapses on the focused probe suite.

The sharpest new movement is in the visual catalog line. A catalog-only role profile first moved the remaining visual failure from wrong-tool/no-call into argument mismatch. A narrower `visual_role_catalog_argument_hints_v2` profile then fixed the targeted selector literal in raw and live replay, reaching `2 / 3` live visual exactness without the exact directive. The follow-up `visual_role_catalog_split_selector_hints_v3` is negative evidence: it preserved latest-filter exactness but regressed readback JSON shape and did not earn live replay. The schema-local `visual_role_catalog_schema_field_hints_v4` profile is split evidence: it is still negative on the old focused three-case slice, but it is the strongest fresh visual hard-slice profile at `6 / 8` exact and `8 / 8` executable. H1n then moved through oracle alias-transfer, repeat, oblique, post-repair, residual, and component-value holdouts. H1q resolves the first transfer tension: the narrower `visual_role_catalog_component_label_guard_v11` reaches `26 / 32` exact and `29 / 32` executor-equivalent across H1n/H1o/H1p, ahead of broad v9 at `23 / 32` and `25 / 32`. H1s is the promotion gate: v12 improves aggregate strict exactness to `27 / 32`, but drops executor-equivalence to `27 / 32` versus v11's `29 / 32`. H1x finally breaks v11 saturation: no-directive reaches `2 / 8`, v11 `7 / 8`, v12 `8 / 8`, and v15 `6 / 8` exact with `7 / 8` executor-equivalent. H1y/H1z then reject prompt-only routed residual and selection-origin fixes: both stay at `5 / 10` on the mixed H1y packet. H2a changes the intervention layer, adding a controller-side stale-selection gate on top of v11, and reaches `8 / 10` exact and executor-equivalent on H1y. Its transfer gate is positive: across H1n/H1o/H1p/H1x it reaches `35 / 40` strict exact and `38 / 40` executor-equivalent, beating v11 and tying v12 strict while beating v12 execution. H2b then isolates the five remaining residual rows and shows v12 is the strict residual-exactness winner at `4 / 5`, while H2a drops to `0 / 5` strict. H2c adds scoped residual exactness while preserving the stale-selection controller gate and reaches `5 / 5` on H2b, but transfers to only `7 / 8` on H1x through a `result chip` -> `result pill` class swap. H2d repairs H1x to `8 / 8` but gives back one H2b strict row. H2e route arbitration reconciles both: `5 / 5` on H2b and `8 / 8` on H1x, with executor-equivalence saturated. H2f then breaks the global promotion claim: on a fresh ten-case holdout, H2e ties H2c at `6 / 10`, remains far above no-directive at `1 / 10`, and fails four rows by preserving the right tool while substituting displayed values or aliases for requested component identities. H2g's first component-identity contract is only a partial executor gain: strict stays `6 / 10`, executor-equivalence rises to `7 / 10`, and the exact-query failure remains. H2h's explicit negative examples repair most of H2f, reaching `9 / 10`, but the transfer backtest rejects global promotion: H2h falls to `3 / 5` on H2b and `6 / 8` on H1x where H2e had saturated both. H2i tries conditionalizing H2h in prompt prose, but fails the H2f gate at `6 / 10`, tying H2e and trailing H2h by three rows. H2j is the first structural answer on this line: target-query normalization reaches `10 / 10` on H2f while preserving `5 / 5` on H2b and `8 / 8` on H1x. H2k supplies the harder post-H2j target/decoy-overlap holdout and helper ablation: on adversarial overlap, full H2j and H2j without stale-selection both reach `8 / 8`, H2h reaches `6 / 8`, and H2e reaches `3 / 8` strict exactness. Both H2j rows record `5` target-query-normalization interventions and `0` stale-selection interventions. H2l then flips the overreach risk under direct target wording: full H2j and H2j without stale-selection still reach `8 / 8`, H2e reaches `7 / 8`, and the only H2j intervention repairs `critical chip` to `status badge`. H2m removes that direct wording and exposes the boundary: H2j and H2j-no-stale fall to `3 / 8`, H2e is `1 / 8` strict and `3 / 8` executor-equivalent, and H2j records three value-bearing over-strip rows. H2n scopes the controller intervention: it ties H2j strict exactness on H2m at `3 / 8`, improves H2m executor-equivalence from `3 / 8` to `5 / 8`, preserves H2k and H2l at `8 / 8`, preserves H2f at `10 / 10`, and records three value-bearing blocks plus two contextual-label rewrites. H2o turns that scope into strict construction: H2m rises to `7 / 8` strict and executor-equivalent, H2k/H2l/H2f remain saturated, and the one remaining row is contextual surface-type alias routing (`result tile`) rather than value-bearing target synthesis. H2p closes that row: H2m reaches `8 / 8` strict and executor-equivalent, with zero delta versus H2o on H2k/H2l/H2f. H2q then breaks the post-H2p saturation: on composed surface/value/stale pressure, H2p remains best at `3 / 8`, H2o reaches `2 / 8`, H2n reaches `0 / 8` strict and `1 / 8` executor-equivalent, and H2e reaches `1 / 8` strict and `2 / 8` executor-equivalent. H2r repairs H2q locally: composed route gating reaches `8 / 8` strict and executor-equivalent and records five interventions matching H2p's five H2q misses. The H2r transfer backtest then reaches `81 / 81` strict and executor-equivalent across current transfer packets, `89 / 89` strict including H2q, and zero non-exact rows. With H2s executed, the current conclusion is sharper than routed prompt help: stale selection-origin failures are controller-addressable, exact alias/code-label residuals and component-class transfer need route arbitration, displayed-value component-identity repair is attributable controller-side normalization, value-bearing target construction and surface-class aliases are locally repairable, and composed route failures are now transfer-positive and fresh-holdout-positive. The remaining question is harder independence or packaged workflow transfer, not whether H2s itself passes.

H2t updates that conclusion: H2r is transfer-positive and H2s-positive, but not yet globally safe. On H2t, H2r/H2p/H2o/H2j tie at `8 / 10`, H2e reaches `6 / 10` strict and `9 / 10` executor-equivalent, and the two H2r misses are raw-exact MLX calls rewritten by target-query normalization from `metric panel` and `summary tile` to `training note` and `caption`. H2u is the first tested answer: it makes both target-query normalization and composed-route gating negation-aware, repairs H2t to `10 / 10`, and preserves `99 / 99` strict and executor-equivalent across the current H2s/H2q/H2m/H2k/H2l/H2f/H2b/H1x/H1y/H1o/H1p transfer subtotal. H2v then breaks the apparent saturation: H2u reaches `4 / 10` strict and `5 / 10` executor-equivalent, while H2r and H2j each reach `3 / 10` strict and `4 / 10` executor-equivalent. H2u helps by one quoted-negation row, but stale-example context and genuine negated targets remain unsolved under strict exactness.

H2w is the semantic repair for H2v. It adds semantic target preservation on top of H2u and reaches `10 / 10` strict and executor-equivalent on the same H2v packet. Relative to H2u, H2w fixes `6` strict misses with `+0.60` exact-rate and `+0.50` executor-equivalence-rate gains. The mechanism is mixed and attributable: `4` semantic-preservation interventions prevent stale/quoted context from hijacking the requested layout target, `3` target-query normalizations canonicalize genuine negated values into component-qualified labels, `1` stale-selection gate repairs the stale risk-lane row, and `1` composed-route block remains active. The follow-on H2w transfer backtest is clean: `109 / 109` strict and executor-equivalent across H2s/H2t/H2q/H2m/H2k/H2l/H2f/H2b/H1x/H1y/H1o/H1p, zero regressions versus H2u, and only the inherited H2t gain versus H2r. The remaining promotion risk is no longer replay-shaped transfer; it is whether packaged workflow or harder live CLI tasks preserve the same semantic pressure instead of resolving it upstream.

H2x is the first CLI-first packaged semantic-pressure gate for that risk. It keeps stale quoted negation, stale selection, instructional negation, and genuine displayed negated values visible in an attributable workflow family. On H2x, H2u drops to `3 / 8` strict and `4 / 8` executor-equivalent, while H2w reaches `8 / 8` on both metrics. The no-fallback controls are the important attribution result: H2u no-fallback ties H2u, H2w no-fallback ties H2w, and all fallback deltas are `0.0`. This makes the current causal story sharper: semantic target preservation and component-qualified target normalization are doing the work on this slice, not broad fallback rescue.

H2y scales H2x into a harder CLI semantic-pressure gate and gives the current boundary. It doubles the packet to `16` cases across quoted stale negation, stale selection negation, instructional negation, and genuine displayed negated values. H2u reaches `4 / 16` strict and `5 / 16` executor-equivalent; H2w reaches `12 / 16` on both metrics, improving over H2u by `+0.50` exact-rate and `+0.4375` executor-equivalence-rate. The no-fallback controls again tie their full-controller rows, so fallback remains non-causal. The unresolved rows are now specific enough to drive the next ablation: all three stale-selection-negation rows still choose stale `refine_selection`, and `h2y_not_active_alert_banner_value_before_component` collapses the intended `alert banner not active` target into the short query `alert`.

That means the next useful work is not broad leaderboard reruns or UI polish. It is a CLI-first, benchmark-backed harness loop around:

- tool-call contract prompts
- controller repair and fallback dependence
- argument fidelity
- approval-safe stop behavior
- attributable packaged workflow families
- local live execution with sandbox evidence

## Current Figures

![H1i readiness, strict interface, and recovered execution](../../results/reports/mlx_tool_contract_harnessing/figures/h1i_readiness_strict_recovered.svg)

![H1h vs H1i no-directive controller burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1h_h1i_controller_burden.svg)

![Tool probe contract gap](../../results/reports/mlx_tool_contract_harnessing/figures/tool_probe_contract_gap.svg)

![H1i failure modes](../../results/reports/mlx_tool_contract_harnessing/figures/h1i_failure_modes.svg)

![Prompt contract candidate targets](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_candidate_targets.svg)

![Executed prompt contract probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_probe_gate.svg)

![Prompt contract wave two probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_wave2_probe_gate.svg)

![Prompt contract wave three probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_wave3_probe_gate.svg)

![Prompt contract wave four probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_wave4_probe_gate.svg)

![Prompt contract wave five probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_wave5_probe_gate.svg)

![Tool catalog profile probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/tool_catalog_profile_probe_gate.svg)

![Prompt contract wave six probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_wave6_probe_gate.svg)

![H1i prompt-contract repeat3 burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1i_prompt_contract_repeat3_burden.svg)

![H1j probe-derived candidate burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1j_probe_derived_burden.svg)

![H1j probe-derived helper burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1j_probe_derived_helper_burden.svg)

![H1k parallel-audit candidate burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1k_parallel_audit_burden.svg)

![H1k parallel-audit helper burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1k_parallel_audit_helper_burden.svg)

![H1l visual executor-equivalence burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1l_visual_executor_equivalence_burden.svg)

![H2k target/decoy overlap gate](../../results/reports/h2k_target_decoy_overlap_synthesis/figures/h2k_target_decoy_overlap_gate.svg)

![H2l target-normalization overreach gate](../../results/reports/h2l_target_normalization_overreach_synthesis/figures/h2l_target_normalization_overreach_gate.svg)

![H2m less-direct overreach gate](../../results/reports/h2m_less_direct_overreach_synthesis/figures/h2m_less_direct_overreach_gate.svg)

![H2n scoped target-normalization gate](../../results/reports/h2n_scoped_target_normalization_synthesis/figures/h2n_scoped_target_normalization_gate.svg)

![H2o value-bearing target synthesis gate](../../results/reports/h2o_value_bearing_target_synthesis/figures/h2o_value_bearing_target_synthesis_gate.svg)

![H2p contextual surface alias routing gate](../../results/reports/h2p_contextual_surface_alias_routing_synthesis/figures/h2p_contextual_surface_alias_routing_gate.svg)

![H2q composed surface value stale gate](../../results/reports/h2q_composed_surface_value_stale_synthesis/figures/h2q_composed_surface_value_stale_gate.svg)

![H2r composed route gating gate](../../results/reports/h2r_composed_route_gating_synthesis/figures/h2r_composed_route_gating_gate.svg)

![H2s fresh composed holdout gate](../../results/reports/h2s_fresh_composed_holdout_synthesis/figures/h2s_fresh_composed_holdout_gate.svg)

![H2t overreach independence gate](../../results/reports/h2t_overreach_independence_synthesis/figures/h2t_overreach_independence_gate.svg)

![H2u negation guard transfer gate](../../results/reports/h2u_negation_guard_synthesis/figures/h2u_negation_guard_transfer_gate.svg)

![H2v semantic negation gate](../../results/reports/h2v_semantic_negation_synthesis/figures/h2v_semantic_negation_gate.svg)

![H2w semantic target preservation gate](../../results/reports/h2w_semantic_target_preservation_synthesis/figures/h2w_semantic_target_preservation_gate.svg)

![H2w transfer backtest gate](../../results/reports/h2w_transfer_backtest_synthesis/figures/h2w_transfer_backtest_gate.svg)

![H2x CLI semantic pressure gate](../../results/reports/h2x_cli_semantic_pressure_synthesis/figures/h2x_cli_semantic_pressure_gate.svg)

![H2y scaled CLI semantic pressure gate](../../results/reports/h2y_scaled_cli_semantic_pressure_synthesis/figures/h2y_scaled_cli_semantic_pressure_gate.svg)

![H1m visual alias-repeat burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1m_visual_alias_repeat_burden.svg)

![Visual hard-slice alias-transfer live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_transfer_live_replay_gate.svg)

![Visual hard-slice alias-transfer oracle live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_transfer_oracle_live_replay_gate.svg)

![Visual hard-slice post-repair live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_post_repair_live_replay_gate.svg)

![Visual hard-slice residual live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_residual_live_replay_gate.svg)

![Visual hard-slice component-value live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_component_value_live_replay_gate.svg)

![Visual hard-slice H1o control-factorial live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1o_live_replay_gate.svg)

![Visual hard-slice H1p component-value live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1p_live_replay_gate.svg)

![H1q component-label guard transfer gate](../../results/reports/mlx_tool_contract_harnessing/figures/h1q_component_label_guard_transfer_gate.svg)

![H1s component-residual transfer gate](../../results/reports/mlx_tool_contract_harnessing/figures/h1s_component_residual_transfer_gate.svg)

![H1x v11-breaker replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/h1x_v11_breaker_gate.svg)

![H1y/H2a routed residual replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/h1y_routed_residual_gate.svg)

![H2a stale-selection transfer gate](../../results/reports/mlx_tool_contract_harnessing/figures/h2a_stale_selection_transfer_gate.svg)

![H2b residual exactness gate](../../results/reports/mlx_tool_contract_harnessing/figures/h2b_residual_exactness_gate.svg)

![H2c scoped residual gate](../../results/reports/mlx_tool_contract_harnessing/figures/h2c_scoped_residual_gate.svg)

![H2e route arbitration gate](../../results/reports/h2e_route_arbitration_synthesis/figures/h2e_route_arbitration_gate.svg)

![H2f fresh holdout bars](../../results/reports/h2f_route_arbitration_holdout_synthesis/figures/h2f_holdout_profile_bars.svg)

![H2h component identity tradeoff gate](../../results/reports/h2h_component_identity_tradeoff_synthesis/figures/h2h_tradeoff_gate.svg)

![H2j target-query normalization transfer gate](../../results/reports/h2j_target_query_normalization_transfer_synthesis/figures/h2j_transfer_gate.svg)

![Exact probe replay gap](../../results/reports/mlx_tool_contract_harnessing/figures/exact_probe_replay_gap.svg)

![Focused exact replay gaps](../../results/reports/mlx_tool_contract_harnessing/figures/exact_probe_replay_focus_gap.svg)

![CLI-live parallel replay gap](../../results/reports/mlx_tool_contract_harnessing/figures/live_parallel_replay_gap.svg)

![CLI-live focused replay gaps](../../results/reports/mlx_tool_contract_harnessing/figures/live_replay_focus_gap.svg)

![Wave three live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/wave3_live_candidate_replay_gate.svg)

![Wave four live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/wave4_live_candidate_replay_gate.svg)

![Visual catalog live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_catalog_live_candidate_replay_gate.svg)

![Visual catalog argument-hints live replay gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_catalog_argument_hints_live_candidate_replay_gate.svg)

![Visual hard-slice probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_probe_gate.svg)

## Evidence Sources

| Artifact | Purpose |
| --- | --- |
| [`H1f compact packet`](../../results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1) | First compact no-directive causal test on five live workflow families. |
| [`H1h full packet`](../../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1) | Full ten-workflow no-directive replication. |
| [`H1i worst-family packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1) | Smaller fast loop derived from the worst H1h workflow families. |
| [`contracted tool probe`](../../results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4) | Exact-call probe for MLX with the tool-turn directive. |
| [`no-directive tool probe`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1) | Exact-call probe after removing the directive. |
| [`executed prompt-contract probe packet`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1) | Three generic no-directive prompt-contract candidates compared against both contracted and no-directive probe baselines. |
| [`executed prompt-contract wave-two packet`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1) | Three second-wave prompt-contract candidates tested on the raw probe before any further H1 promotion. |
| [`executed prompt-contract wave-three packet`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1) | Three mechanism-targeted candidates derived from CLI-live replay: canonical JSON copy, visual tool initiation, and parallel two-call array. |
| [`executed prompt-contract wave-four packet`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1) | Narrow visual state/tool-selection candidate tested after wave three exposed the remaining wrong-tool visual referent failure. |
| [`executed prompt-contract wave-five packet`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1) | Surgical latest-selection refinement candidate tested after wave four failed to improve live visual tool selection. |
| [`visual role catalog probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe) | Isolated tool-catalog presentation intervention for visual routing roles, tested without any prompt contract. |
| [`visual catalog argument-hints probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe) | Narrow catalog-profile argument semantics test that preserves the visual routing profile while targeting selector literal drift. |
| [`visual split-selector probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe) | Negative catalog-profile follow-up showing broader split-selector prose regresses readback shape and does not beat v2. |
| [`visual schema-field hints probe`](../../results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe) | Schema-local field-hint follow-up that ties v2 exactness, restores readback versus v3, and still fails executable form targeting. |
| [`publication evidence ledger`](../../results/reports/publication_evidence_ledger/ledger.md) | Paper-facing claim ledger mapping each claim to packet-backed evidence, limitations, and next tests. |
| [`publication readiness audit`](../../results/reports/publication_readiness_audit/publication_readiness_audit.md) | Blocking/recommended audit of whether the current evidence tree is ready to support a manuscript draft. |
| [`visual hard-slice design`](../../results/reports/visual_hard_slice_design/design.md) | Shared case-design source for the fresh visual discriminator across argument copying, routing, referent carryover, and readback. |
| [`executed visual hard-slice packet`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1) | Eight-system hard-slice probe showing schema-field hints as the strongest no-directive executable visual profile and v5 target-literal wording as a negative repair attempt. |
| [`visual hard-slice v5-vs-v4 comparison`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints) | Direct comparison showing v5 loses exactness and executability versus v4 by introducing a stale-selection wrong-tool regression. |
| [`visual hard-slice exactness diagnostic`](../../results/reports/visual_hard_slice_exactness_diagnostic) | Separates strict benchmark-canonical target labels from executor-visible target success for v4 and v5. |
| [`visual catalog literal-guard v6 packet`](../../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe) | Composition test combining the visual role catalog with `literal_argument_guard_v1`. |
| [`H1i prompt-contract repeat3 packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet) | Repeated second-stage candidate packet: three attempts per H1i workflow family per row. |
| [`H1j probe-derived candidate packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet) | Six packaged live workflows selected from exact no-directive probe failure families. |
| [`H1j probe-derived helper packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet) | Controller-helper ablation on the same H1j probe-derived packaged workflow set. |
| [`H1k parallel-audit candidate packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet) | Packaged live promotion of the deferred `parallel_audit_array_literal` replay case. |
| [`H1k parallel-audit helper packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet) | Controller-helper ablation on the packaged parallel-audit workflow. |
| [`H1l visual executor-equivalence candidate packet`](../../results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet) | Packaged visual live promotion of the visual hard-slice executor-equivalence result. |
| [`exact-probe replay packet`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1) | Dry-run replay artifacts for the eight failed no-directive exact-call probe cases. |
| [`CLI-live parallel replay comparison`](../../results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1) | Operator-surface A/B for the parallel-array exact replay case. |
| [`CLI-live visual replay comparison`](../../results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1) | Operator-surface A/B for the visual no-call exact replay cases. |
| [`CLI-live canonical replay comparison`](../../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1) | Operator-surface A/B for CLI/API canonical argument exact replay cases. |
| [`wave-three visual live candidate comparison`](../../results/tool_probe_replay_live_comparisons/20260507T_visual_state_visual_tool_initiation_vs_no_directive_live_v1) | Candidate live replay showing visual tool initiation improves over no-directive but remains below contracted. |
| [`wave-three canonical live candidate comparison`](../../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_canonical_json_copy_vs_no_directive_live_v1) | Candidate live replay showing canonical JSON copy does not improve exact canonical argument replay. |
| [`wave-four visual live candidate comparison`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1) | Candidate live replay showing visual state/tool-selection wording preserves one exact visual recovery but does not beat wave three. |
| [`visual tool-choice diagnostic`](../../results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1) | Expected-vs-actual tool-choice diagnostic showing wave three/four choose `extract_layout`, while the catalog profile reaches `refine_selection` but drifts on the selector literal. |
| [`Gemini CLI dry-run baseline`](../../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1) | External-reference prompt and command manifest over the H1h workflow families. |
| [`H1n post-repair holdout`](../../results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1) | Fresh eight-case replay-shaped holdout testing argument hints, v6 code hints, and v7 code guard on code-like labels, stale-selection hazards, and non-code labels. |
| [`H1n post-repair diagnostic`](../../results/reports/visual_alias_transfer_post_repair_diagnostic/diagnostic.md) | Matrix diagnostic showing v7 code guard as the current post-repair upper bound and recording the remaining `chip l90` / `status pill` misses. |
| [`H1n residual holdout`](../../results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1) | Fresh eight-case residual packet targeting the post-repair chip, pill, and stale-selection misses with a v8 hybrid label guard. |
| [`H1n residual diagnostic`](../../results/reports/visual_alias_transfer_residual_diagnostic/diagnostic.md) | Matrix diagnostic showing v8 hybrid label guard as the current strict upper bound at `7 / 8`, while `state pill` remains unresolved. |
| [`H1n component-value holdout`](../../results/tool_probe_replay_packets/20260510T_visual_hard_slice_component_value_oracle_dry_run_v1) | Fresh eight-case component-role/value packet targeting pill, badge, chip, field, and stale-selection ambiguity after the residual `state pill` miss. |
| [`H1n component-value diagnostic`](../../results/reports/visual_component_value_diagnostic/diagnostic.md) | Matrix diagnostic showing v9 component-value guard as negative evidence and v10 no-call control rescue as the current strict/executor upper bound. |
| [`H1n no-call rescue transfer synthesis`](../../results/reports/h1n_no_call_rescue_transfer_synthesis/report.md) | Aggregate showing v10 as a strong no-directive activation improvement but not a replacement for per-packet visual incumbents. |
| [`H1o control-factorial packet`](../../results/tool_probe_replay_packets/20260510T_h1o_control_factorial_oracle_dry_run_v1) | Twelve-case mechanism split over activation/no-call, code/negation, and component/value families. |
| [`H1o control-factorial diagnostic`](../../results/reports/visual_h1o_control_factorial_diagnostic/diagnostic.md) | Matrix diagnostic showing argument hints and component-value guard tie strict upper bound, while v10 regresses one activation case. |
| [`H1o mechanism-family synthesis`](../../results/reports/h1o_control_factorial_synthesis/report.md) | Family-level synthesis showing activation saturation, code/negation repairability, and component/value residue. |
| [`H1p component-only holdout`](../../results/tool_probe_replay_packets/20260510T_h1p_component_value_holdout_oracle_dry_run_v1) | Twelve-case component/value holdout spanning compact components, surface labels, and stale-selection decoys without activation-focused wording. |
| [`H2d transfer tradeoff synthesis`](../../results/reports/h2d_transfer_tradeoff_synthesis/report.md) | Shows class-preserving transfer repairs H2c's H1x miss while giving back one H2b strict row that remains executor-equivalent. |
| [`H2e route arbitration synthesis`](../../results/reports/h2e_route_arbitration_synthesis/report.md) | Shows H2e reaches `5 / 5` on H2b and `8 / 8` on H1x, preserving executor-equivalence on both current gates. |
| [`H2f route arbitration holdout synthesis`](../../results/reports/h2f_route_arbitration_holdout_synthesis/report.md) | Fresh ten-case holdout showing H2e ties H2c at `6 / 10`, stays above no-directive at `1 / 10`, H2g improves executor-equivalence without strict exactness gain, H2h repairs H2f to `9 / 10`, and H2i conditionalization falls back to `6 / 10`. |
| [`H2h component-identity tradeoff synthesis`](../../results/reports/h2h_component_identity_tradeoff_synthesis/report.md) | Transfer report showing H2h is scoped positive but globally negative: `9 / 10` on H2f, `3 / 5` on H2b, and `6 / 8` on H1x. |
| [`H2j target-query normalization transfer synthesis`](../../results/reports/h2j_target_query_normalization_transfer_synthesis/report.md) | Transfer report showing H2j reaches `10 / 10` on H2f, preserves `5 / 5` on H2b and `8 / 8` on H1x, and records target-query-normalization/stale-selection interventions in replay metadata. |
| [`H2k target/decoy overlap synthesis`](../../results/reports/h2k_target_decoy_overlap_synthesis/report.md) | Fresh post-H2j holdout and helper ablation showing full H2j and H2j-no-stale at `8 / 8`, H2h at `6 / 8`, H2e at `3 / 8` strict exactness, and `5` target-query-normalization interventions with `0` stale-selection interventions for both H2j rows. |
| [`H2l target-normalization overreach synthesis`](../../results/reports/h2l_target_normalization_overreach_synthesis/report.md) | Fresh overreach holdout showing full H2j and H2j-no-stale at `8 / 8`, H2e at `7 / 8`, one target-query-normalization repair from `critical chip` to `status badge`, and `0` stale-selection interventions. |
| [`H2m less-direct overreach synthesis`](../../results/reports/h2m_less_direct_overreach_synthesis/report.md) | Less-direct overreach holdout showing full H2j and H2j-no-stale at `3 / 8`, H2e at `1 / 8` strict and `3 / 8` executor-equivalent, `5` target-query-normalization interventions, `0` stale-selection interventions, and `3` value-bearing over-strip rows. |
| [`H2n scoped target-normalization synthesis`](../../results/reports/h2n_scoped_target_normalization_synthesis/report.md) | Scoped normalizer report showing H2n ties H2j at `3 / 8` strict on H2m, improves executor-equivalence to `5 / 8`, preserves H2k/H2l/H2f transfer gates, records `3` value-bearing blocks, and motivates H2o canonical target synthesis. |
| [`H2o value-bearing target synthesis`](../../results/reports/h2o_value_bearing_target_synthesis/report.md) | Canonical value-bearing synthesis report showing H2o reaches `7 / 8` strict and executor-equivalent on H2m, preserves H2k/H2l/H2f transfer gates, records `4` value-bearing syntheses and `2` contextual-label rewrites, and isolates contextual surface-type alias routing as the remaining H2m residue. |
| [`H2p contextual surface alias routing synthesis`](../../results/reports/h2p_contextual_surface_alias_routing_synthesis/report.md) | Contextual surface-alias synthesis report showing H2p reaches `8 / 8` strict and executor-equivalent on H2m, preserves H2k/H2l/H2f transfer gates, records `1` contextual surface-alias intervention, and closes the current H2m non-exact set. |
| [`H2q composed surface/value/stale synthesis`](../../results/reports/h2q_composed_surface_value_stale_synthesis/report.md) | Composed post-H2p boundary report showing H2p remains best but reaches only `3 / 8` strict and executor-equivalent, with remaining failures split between same-value decoy argument mismatches and stale-selection wrong-tool calls. |
| [`H2r composed route-gating synthesis`](../../results/reports/h2r_composed_route_gating_synthesis/report.md) | Local H2q repair report showing H2r reaches `8 / 8` strict and executor-equivalent, records `5` composed-route interventions, and improves by `+0.625` exact and executor-equivalence rate versus H2p. |
| [`H2r transfer backtest synthesis`](../../results/reports/h2r_transfer_backtest_synthesis/report.md) | Transfer report showing H2r reaches `81 / 81` strict and executor-equivalent across transfer packets, `89 / 89` strict including H2q, ties H2j/H2e on H2b/H1x, and beats H2h on both regression gates. |
| [`H2s fresh composed holdout synthesis`](../../results/reports/h2s_fresh_composed_holdout_synthesis/report.md) | Fresh frozen-H2r holdout showing H2r reaches `10 / 10` strict and executor-equivalent, while H2p/H2o each reach `3 / 10` and H2j reaches `1 / 10`; this is the positive holdout that motivated H2t. |
| [`H2t overreach-independence synthesis`](../../results/reports/h2t_overreach_independence_synthesis/report.md) | Fresh independence holdout showing H2r/H2p/H2o/H2j tie at `8 / 10`, H2e reaches `6 / 10` strict and `9 / 10` executor-equivalent, and `2` H2r raw-exact calls are rewritten by target-query normalization into note/caption labels. |
| [`H2u negation guard synthesis`](../../results/reports/h2u_negation_guard_synthesis/report.md) | Transfer-positive negation guard report showing H2u repairs H2t to `10 / 10` and preserves `99 / 99` across the current same-family transfer subtotal. |
| [`H2v semantic negation synthesis`](../../results/reports/h2v_semantic_negation_synthesis/report.md) | Fresh semantic boundary report showing H2u is transfer-positive but not semantic-complete: stale-example context and genuine negated values still fail. |
| [`H2w semantic target preservation synthesis`](../../results/reports/h2w_semantic_target_preservation_synthesis/report.md) | Local semantic repair report showing H2w repairs H2v to `10 / 10`, fixing `6` H2u strict misses with attributable semantic-preservation and component-qualified value canonicalization. |
| [`H2w transfer backtest synthesis`](../../results/reports/h2w_transfer_backtest_synthesis/report.md) | Transfer/back-compat report showing H2w preserves `109 / 109` strict and executor-equivalent rows, ties H2u with zero regressions, and records the MLX low-concurrency runtime-posture lesson. |
| [`H2x CLI semantic pressure synthesis`](../../results/reports/h2x_cli_semantic_pressure_synthesis/report.md) | Packaged/CLI semantic-pressure gate showing H2w repairs H2x from H2u's `3 / 8` strict and `4 / 8` executor-equivalent to `8 / 8`, while no-fallback controls have `0.0` deltas. |
| [`H2y scaled CLI semantic pressure synthesis`](../../results/reports/h2y_scaled_cli_semantic_pressure_synthesis/report.md) | Scaled CLI semantic-pressure gate showing H2w improves H2y from H2u's `4 / 16` strict and `5 / 16` executor-equivalent to `12 / 16`, while no-fallback controls keep `0.0` deltas and the remaining boundary is stale-selection negation plus short-query collapse. |
| [`H1p component-value diagnostic`](../../results/reports/visual_h1p_component_value_diagnostic/diagnostic.md) | Matrix diagnostic showing component-value guard v9 as the local H1p upper bound over argument hints, hybrid guard, and no-call rescue. |
| [`H1p report table`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1p_live_replay_summary.csv) | Paper-facing summary table for H1p live replay comparisons against no-directive. |
| [`H1q component-label guard synthesis`](../../results/reports/h1q_component_label_guard_transfer_synthesis/report.md) | Aggregate transfer synthesis showing v11 as the strongest current profile across H1n/H1o/H1p. |
| [`H1q aggregate report table`](../../results/reports/mlx_tool_contract_harnessing/tables/h1q_component_label_guard_aggregate_summary.csv) | Paper-facing aggregate table comparing v11 against no-directive, v2, v8, v9, and v10. |
| [`H1s component-residual transfer synthesis`](../../results/reports/h1s_component_residual_transfer_synthesis/report.md) | Transfer gate showing v12 is a targeted residual patch, not a global replacement for v11. |
| [`H1s aggregate report table`](../../results/reports/mlx_tool_contract_harnessing/tables/h1s_component_residual_transfer_aggregate.csv) | Paper-facing aggregate table comparing v12 against v11 and no-directive across H1n/H1o/H1p. |
| [`H1x v11-breaker synthesis`](../../results/reports/h1x_v11_breaker_synthesis/report.md) | Focused replay gate showing v11 no longer saturates under oblique stale-field pressure, while v12 wins locally. |
| [`H1x v11-breaker report table`](../../results/reports/mlx_tool_contract_harnessing/tables/h1x_v11_breaker_packet_summary.csv) | Paper-facing packet table for no-directive, v11, v12, and v15 on H1x. |
| [`H1y/H2a routed-residual synthesis`](../../results/reports/h1y_routed_residual_synthesis/report.md) | Mixed ten-case prompt-vs-controller gate showing H2a as the current local winner. |
| [`H1y/H2a report table`](../../results/reports/mlx_tool_contract_harnessing/tables/h1y_routed_residual_packet_summary.csv) | Paper-facing packet table for no-directive, v11, v12, v16, v17, and H2a. |
| [`H2a stale-selection transfer synthesis`](../../results/reports/h2a_stale_selection_transfer_synthesis/report.md) | Held-out transfer synthesis showing H2a ties v12 strict transfer while beating v12 executor-equivalence. |
| [`H2a transfer report table`](../../results/reports/mlx_tool_contract_harnessing/tables/h2a_stale_selection_transfer_aggregate_summary.csv) | Paper-facing aggregate table for H2a versus no-directive, v11, and v12 across H1n/H1o/H1p/H1x. |
| [`H2b residual exactness synthesis`](../../results/reports/h2b_residual_exactness_synthesis/report.md) | Five-row residual gate showing v12 wins strict exactness but should be routed, not globalized. |
| [`H2b residual exactness table`](../../results/reports/mlx_tool_contract_harnessing/tables/h2b_residual_exactness_packet_summary.csv) | Paper-facing packet table for no-directive, v11, v12, v15, H2a, and v9 on the H2b residual rows. |
| [`H2b residual exactness case matrix`](../../results/reports/mlx_tool_contract_harnessing/tables/h2b_residual_exactness_case_matrix.csv) | Case-level strict/executor-equivalence matrix for the five residual exactness rows. |
| [`H2c scoped residual synthesis`](../../results/reports/h2c_scoped_residual_synthesis/report.md) | Local fit gate showing H2c reaches 5/5 on H2b while preserving the need for transfer validation. |
| [`H2c scoped residual table`](../../results/reports/mlx_tool_contract_harnessing/tables/h2c_scoped_residual_packet_summary.csv) | Paper-facing packet table for H2c versus no-directive, v9, v12, and H2a on H2b. |
| [`H2c scoped residual comparison table`](../../results/reports/mlx_tool_contract_harnessing/tables/h2c_scoped_residual_comparison_summary.csv) | Paper-facing comparison table quantifying H2c deltas over no-directive, H2a, v9, and v12. |

## H1s Research Read

H1s turns the H1r local win into a transfer-gated finding. V12 is real: it saturates H1r and improves H1p to `11 / 12` exact and executor-equivalent. But it is not the global visual-role default. Across H1n/H1o/H1p, v12 reaches `27 / 32` strict exactness, one case above v11, while executor-equivalence drops from v11's `29 / 32` to `27 / 32`. The strongest negative transfer is H1n, where v12 falls to `5 / 8` exact and executor-equivalent. The next publishable question is a conditional-route or prompt-factorial slice: keep v11 as the general component-label profile, and only add v12 residual wording for code-label or nonstandard component-class contexts.

## H1x Research Read

H1x is the first post-H1w packet that breaks v11 saturation. No-directive solves only the activation/no-call rows (`2 / 8`) and fails the oblique stale-field, surface-value, and nonstandard-class families. V11 improves strongly to `7 / 8`, but misses `h1x_responsible_party_field_old_owner_memo_decoy` with a wrong-tool call. V12 repairs that miss and reaches `8 / 8`; v15 reaches only `6 / 8` strict exactness and `7 / 8` executor-equivalent. Because H1s already showed v12 has broader negative transfer, H1x should be read as evidence for routed residual help rather than a global v12 promotion.

## H1y/H2a Research Read

H1y asks whether routed residual prompt help can keep v11's transfer stability while capturing v12's H1x stale-field gain. The answer is no for prompt-only variants: no-directive is `0 / 10`, v11 is `5 / 10`, v12 is `7 / 10`, v16 routed-residual prose is `5 / 10`, and v17 selection-origin prose is `5 / 10`. H2a keeps v11 but adds a controller-side stale-selection gate and reaches `8 / 10`. It fixes all three stale-field route rows and preserves both surface-value holdouts. This is now a publishable systems result: the stale user-mentioned `selection_id` problem belongs in runtime mediation.

## H2a Transfer Research Read

The H2a transfer gate answers the overfit concern. Across H1n/H1o/H1p/H1x, H2a reaches `35 / 40` strict exact and `38 / 40` executor-equivalent. That beats no-directive (`12 / 40`, `14 / 40`) and v11 (`33 / 40`, `36 / 40`), and it ties v12 strict exactness while beating v12 executor-equivalence (`35 / 40`). The residuals are no longer broad no-call or wrong-tool collapse; they are exact alias/code-label rows: `result pill`, `alert s92`, `badge c08`, `state tag`, and `mode toggle`. H2b is the packet that now isolates those rows.

## H2b Residual Exactness Research Read

H2b answers what H2a does not solve. On the five residual rows, no-directive reaches `1 / 5` strict and `2 / 5` executor-equivalent; v11 reaches `0 / 5` strict and `3 / 5` executor-equivalent; v12 reaches `4 / 5` strict and executor-equivalent; v15 reaches `3 / 5`; H2a reaches `0 / 5` strict and `3 / 5` executor-equivalent; and v9 reaches `3 / 5` strict with `4 / 5` executor-equivalence. This makes H2b a clean routing result: v12-like residual exactness is useful on alias/code-label pressure, but H1s blocks global v12 promotion and H2a remains the stale-selection controller helper. The next publishable slice is H2c, a scoped route/factor that chooses between stale-selection mediation and residual exactness without leaking expected calls.

## H2c Scoped Residual Research Read

H2c is the first route/factor to saturate the H2b residual packet. It reaches `5 / 5` strict exact and `5 / 5` executor-equivalent, fixing v12's remaining `result pill` miss while keeping the code-label and nonstandard-component rows exact. It also beats H2a by `+1.0` exact-rate on this residual packet, which reinforces the mechanism split: H2a is for stale-selection mediation, and H2c is for scoped residual exactness. The caution is the important part for publication quality: H2c is still a five-row fit packet. The next result must be a minimal transfer gate over H1n/H1o/H1p/H1x residual families before any global/default promotion.

## H2j-H2r Research Read

H2j is the current structural candidate because it turns H2h's prompt-side component-identity repair into controller-attributable target-query normalization. It repairs H2f to `10 / 10` while preserving H2b and H1x saturation. H2k then shows that this is not just transfer luck: on target/decoy overlap, H2j reaches `8 / 8` while H2e reaches `3 / 8` strict and H2h reaches `6 / 8`; the no-stale ablation ties full H2j and records the same `5` target-normalization interventions with `0` stale-selection events. H2l asks the opposing safety question under direct wording: does that normalizer strip labels when the longer label or alias is genuinely the target? On H2l, H2j and no-stale H2j both reach `8 / 8`; H2e reaches `7 / 8`; and the only intervention repairs `critical chip` into the regression-guard target `status badge`.

H2m is the important boundary result. Removing direct target-is phrasing drops H2j and no-stale H2j to `3 / 8` while H2e is `1 / 8` strict and `3 / 8` executor-equivalent. H2j still helps two exact contextual-label cases, but three target-normalization interventions over-strip value-bearing labels into shorter component labels.

H2n answers the scope question. It blocks those value-bearing over-strips when the local visual catalog contains a longer label and the prompt evidence asks for the displayed value. On H2m, H2n remains `3 / 8` strict but improves executor-equivalence to `5 / 8`; against H2e it improves both strict and executor-equivalence by `+0.25`. It also preserves the transfer gates: `8 / 8` on H2k, `8 / 8` on H2l, and `10 / 10` on H2f with zero exact-rate delta versus H2j.

H2o answers the construction question. When the longer label is recoverable, it synthesizes the canonical value-bearing `target_query`: `result badge Blocked`, `state tag Closed`, `mode toggle Manual`, and `priority badge Critical`. On H2m, H2o reaches `7 / 8` strict and executor-equivalent, improving exact-rate by `+0.50` versus H2n/H2j and `+0.75` versus H2e. H2k and H2l remain `8 / 8`, and H2f remains `10 / 10`.

H2p answers the remaining H2m surface-alias question. When the prompt asks for the surface class (`tile-style result surface`) and demotes badge/comment elements to nearby context, H2p rewrites the displayed value `Blocked` to the recoverable surface label `result tile`. H2m rises from H2o's `7 / 8` to `8 / 8` strict and executor-equivalent. Relative to H2n, H2p adds `+0.625` strict exact-rate and `+0.375` executor-equivalence-rate; relative to H2e it adds `+0.875` strict and `+0.625` executor-equivalence. H2k, H2l, and H2f remain saturated with zero exact-rate delta versus H2o.

H2q answers the composition question negatively. When value-bearing labels, surface aliases, stale-selection pressure, and decoys appear together, H2p remains the strongest current stack but reaches only `3 / 8` strict and executor-equivalent. It still beats H2o by `+0.125` strict, H2n by `+0.375`, and H2e by `+0.25`, but the unsolved rows are no longer one isolated alias. They are composed route-gating failures: stale `refine_selection` calls survive explicit ignore-old-selection prompts, and adjacent same-value comments, banners, and switches beat the requested surface classes.

H2r is now the local positive answer to H2q and transfer-positive on the current packet set. It adds composed route gating after H2p and reaches `8 / 8` strict and executor-equivalent on H2q, a `+0.625` exact-rate and executor-equivalence-rate gain versus H2p. The local mechanism is cleanly attributable: H2r records five composed-route interventions, exactly matching H2p's five non-exact H2q rows, split into two stale-selection rewrites and three requested-surface restorations. The transfer backtest then reaches `81 / 81` strict and executor-equivalent across H2m/H2k/H2l/H2f/H2b/H1x/H1y/H1o/H1p, and `89 / 89` strict when H2q is included. H2r ties H2j/H2e on the explicit H2b/H1x regression gates while beating H2h by `+0.40` and `+0.25` exact-rate respectively.

H2s is the first fresh frozen-H2r holdout after that transfer result. It confirms the H2r mechanism on unseen composed cases: H2r reaches `10 / 10` strict and executor-equivalent, while H2p and H2o each reach `3 / 10` and H2j reaches `1 / 10`. The H2r row records `7` composed route gates, `2` value-bearing syntheses, and `4` target-query normalizations. H2t then breaks the line in a useful way: H2r/H2p/H2o/H2j tie at `8 / 10`, H2e reaches `6 / 10` strict and `9 / 10` executor-equivalent, and the two H2r misses are raw-exact MLX calls overwritten by target-query normalization. H2u repairs that pipeline-order failure and now preserves the full current transfer pass: H2s/H2q/H2m remain `26 / 26`, H2k/H2l/H2f/H2b/H1x add `39 / 39`, H1y/H1o/H1p add `34 / 34`, and the broad subtotal is `99 / 99` with zero aggregate exact-rate delta versus H2r.

## H1o Research Read

H1o changes the immediate question. The next move is not to generalize the v10 no-call rescue. No-directive already solves the activation/no-call family at `4 / 4`, and v10 drops one of those cases. The remaining value is in component/value ambiguity: on H1o, argument hints v2 and component-value guard v9 tie the strict upper bound at `9 / 12`, but the best component/value-family rows only reach `2 / 4`. That is the right basis for H1p: a component-only holdout that tests selector copying and component label/value discrimination without adding activation wording unless the packet demands it.

## H1p Research Read

H1p answers the H1o follow-up with a split result. Component-value guard v9 is locally strong on a pure component/value domain: it reaches `10 / 12` exact and `11 / 12` executor-equivalent, compared with `6 / 12` for argument hints v2 and no-call rescue v10. The no-directive baseline at `0 / 12` confirms that H1p is now a useful hard slice rather than another saturated packaged-workflow row. The caution is equally important: H1n already showed broad component-value prose can hurt passable selector cases, so H1p promotes a transfer question, not v9 itself. The next packet should test narrower component-only wording across H1p, H1o, and H1n before changing the default visual catalog profile.

## Packet Summary

| Packet | Episodes | Contracted readiness | No-directive readiness | No-directive repair | No-directive fallback | No-directive arg repair | No-directive raw clean | No repair readiness | No fallback readiness | No arg repair readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1f compact | 5 | `0.97936` | `0.97936` | `0.70` | `0.20` | `0.50` | `0.30` | `0.73818` | `0.92104` | `0.82036` |
| H1h full | 10 | `0.96891` | `0.96891` | `0.70` | `0.25` | `0.45` | `0.30` | `0.73801` | `0.89598` | `0.83016` |
| H1i worst-family | 4 | `0.97710` | `0.97710` | `1.00` | `0.50` | `0.50` | `0.00` | `0.64697` | `0.83125` | `0.81220` |

## H1i System Read

H1i is now the best fast loop for local MLX prompt-contract work because it keeps the same causal ordering as H1h while making the gaps larger.

| System | Readiness | Strict interface | Recovered execution | Controller repair | Controller fallback | Arg repair | Raw clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| contracted | `0.97710` | `1.00000` | `1.00000` | `0.00` | `0.00` | `0.00` | `1.00` |
| no directive | `0.97710` | `1.00000` | `1.00000` | `1.00` | `0.50` | `0.50` | `0.00` |
| no directive + no repair | `0.64697` | `0.29688` | `0.00000` | `1.25` | `1.25` | `0.00` | `0.725` |
| no directive + no fallback | `0.83125` | `0.62500` | `0.50000` | `0.50` | `0.00` | `0.50` | `0.50` |
| no directive + no arg repair | `0.81220` | `0.71875` | `0.50000` | `0.50` | `0.50` | `0.00` | `0.50` |

The no-directive row is the important diagnostic: it is top-line clean but has zero raw-clean rate. That is a strong sign that readiness parity is controller-mediated, not raw model compliance.

## Probe Result

The focused tool probe explains the H1h/H1i behavior.

| Probe row | Exact match | Executable visual match |
| --- | ---: | ---: |
| contracted MLX | `7 / 8` | `1 / 1` |
| no-directive MLX | `0 / 8` | `0 / 1` |

Failure-mode split:

| Side | Failure mode | Case count | Interpretation |
| --- | --- | ---: | --- |
| candidate | `argument_mismatch` | 4 | The right tool family often appears, but canonical CLI/API arguments drift. |
| candidate | `no_tool_call` | 4 | Visual referent and parallel-tool cases collapse out of the tool protocol. |
| baseline non-exact | `executable_paraphrase` | 1 | Contracted MLX paraphrases one visual selector, but the executor can still resolve it. |

This is the strongest current reason to keep exact interface metrics, executable recovery metrics, and readiness metrics separate. A row can look solved at the task level while being fragile at the contract level.

## Prompt-Contract Candidate Queue

The prompt-contract queue now has six prompt-contract waves plus isolated tool-catalog profile probes. They deliberately do not include the exact planned tool call. That keeps the probe honest: a candidate should improve raw tool protocol behavior without simply leaking the oracle next call.

| Candidate system | Contract | Target |
| --- | --- | --- |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor` | `schema_anchor_v1` | Generic JSON/schema obedience for CLI/API canonicalization. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard` | `literal_argument_guard_v1` | Literal argument copying for path, query, record ids, visual selectors, and filters. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required` | `tool_required_parallel_v1` | No-tool-call and parallel/visual protocol collapse. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required` | `schema_literal_tool_required_v2` | Combined schema, literal-copy, and tool-required pressure. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_next_call_state` | `visual_next_call_state_v2` | Stateful visual next-call behavior after a visual result exists. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required` | `parallel_array_required_v2` | JSON-array shape for independent multi-source/parallel checks. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_canonical_json_copy` | `canonical_json_copy_v3` | Exact CLI/API JSON and literal argument token copying. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation` | `visual_tool_initiation_v3` | Visual no-call recovery through stateful visual tool initiation. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_two_call_array` | `parallel_two_call_array_v3` | Independent two-source checks as a two-call JSON array. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection` | `visual_state_tool_selection_v4` | Visual state-specific tool selection after wave three recovered tool initiation but missed a filter/refinement case. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_refine_selection` | `visual_refine_selection_v5` | Surgical latest-selection filtering pressure that prioritizes `refine_selection` when a current `selection_id` exists. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog` | `visual_role_catalog_v1` catalog profile | Tool-catalog-level visual routing roles, isolated from prompt-contract wording. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints` | `visual_role_catalog_argument_hints_v2` catalog profile | Tool-catalog-level visual argument semantics for selector fields after v1 fixed routing but left literal drift. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints` | `visual_role_catalog_schema_field_hints_v4` catalog profile | Schema-local visual field descriptions for `target_query`, `filter_query`, and `region_id` after v3 showed broad prose can destabilize JSON shape. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_literal_guard` | `literal_argument_guard_v1` + `visual_role_catalog_v1` | Composition test: preserve the catalog routing gain while attempting literal argument repair. |

Generated candidate table:

- [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv)

Dry-run candidate probe packet command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_probe_candidates
```

Real candidate probe packet command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_probe_candidates \
  --execute
```

Second-wave dry-run packet:

- [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1)

Second-wave execution command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_wave2_execute_v1 \
  --candidate-wave v2 \
  --execute
```

Third-wave executed packet:

- [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1)

Third-wave execution command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_wave3_execute_v1 \
  --candidate-wave v3 \
  --execute
```

Fourth-wave executed packet:

- [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1)

Fourth-wave execution command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_wave4_execute_v1 \
  --candidate-wave v4 \
  --execute
```

Fifth-wave executed packet:

- [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1)

Fifth-wave execution command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_wave5_execute_v1 \
  --candidate-wave v5 \
  --execute
```

## Executed Prompt-Contract Probe Gate

The first executed candidate packet is now recorded at:

- [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1)
- [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1/candidate_gate_summary.md)
- generated report table: [`prompt_contract_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_probe_gates.csv)
- generated failure table: [`prompt_contract_probe_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_probe_failure_modes.csv)

| Contract | Exact | Executable | Delta exact vs no-directive | Dominant failure | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| `schema_anchor_v1` | `0.125` | `0.0` | `+0.125` | `argument_mismatch` | weak exact gain |
| `literal_argument_guard_v1` | `0.0` | `1.0` | `0.0` | `no_tool_call` | visual executable gain only |
| `tool_required_parallel_v1` | `0.0` | `1.0` | `0.0` | `no_tool_call` | visual executable gain only |

The result is useful but not victory-shaped. All three candidates improve over the raw no-directive probe on one case, but none approaches the contracted MLX row's `7 / 8` exact-call rate. The main practical interpretation is:

- `schema_anchor_v1` is the only exact-copy gain, but it is a weak one-case visual-readback recovery.
- `literal_argument_guard_v1` and `tool_required_parallel_v1` recover the executable visual target, but they do not improve exact JSON copy rate.
- `tool_required_parallel_v1` still has the most `no_tool_call` failures, so its wording is not yet solving the no-call family it was meant to target.
- H1i can use these candidates as mechanism probes, but they are not replacements for the final tool-turn directive.

## Prompt-Contract Wave Two Probe Gate

The second executed candidate packet is now recorded at:

- [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1)
- [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1/candidate_gate_summary.md)
- generated report table: [`prompt_contract_wave2_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave2_probe_gates.csv)
- generated failure table: [`prompt_contract_wave2_probe_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave2_probe_failure_modes.csv)
- generated promotion table: [`prompt_contract_promotion_decisions.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv)

| Contract | Exact | Executable | Delta exact vs no-directive | Dominant failure | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| `schema_literal_tool_required_v2` | `0.125` | `0.0` | `+0.125` | `argument_mismatch` | weak exact gain |
| `visual_next_call_state_v2` | `0.0` | `1.0` | `0.0` | `no_tool_call` | visual executable gain only |
| `parallel_array_required_v2` | `0.0` | `0.0` | `0.0` | `no_tool_call` | no probe gain |

Wave two does not change the research direction. Combining schema anchoring, literal copying, and tool-required language still recovers only one exact case. The visual-state contract again repairs executable visual behavior without exact JSON fidelity. The parallel-array contract does not fix the no-call family. None of these candidates should be promoted as a replacement for the final tool-turn directive.

## Prompt-Contract Wave Three Probe Gate

The third executed candidate packet is now recorded at:

- [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1)
- [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1/candidate_gate_summary.md)
- generated report table: [`prompt_contract_wave3_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave3_probe_gates.csv)
- generated failure table: [`prompt_contract_wave3_probe_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave3_probe_failure_modes.csv)

| Contract | Exact | Executable | Delta exact vs no-directive | Dominant failure | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| `canonical_json_copy_v3` | `0.125` | `0.0` | `+0.125` | `no_tool_call` | weak exact gain |
| `visual_tool_initiation_v3` | `0.125` | `1.0` | `+0.125` | `no_tool_call` | weak exact gain |
| `parallel_two_call_array_v3` | `0.0` | `0.0` | `0.0` | `no_tool_call` | no probe gain |

Wave three makes the current boundary crisper. Canonical JSON wording still gives only the familiar one-case exact gain. Visual tool initiation is the best candidate so far because it combines the one-case exact gain with executable visual recovery. Parallel two-call wording still fails to move the parallel no-call case.

## Prompt-Contract Wave Four Probe Gate

The fourth executed candidate packet is now recorded at:

- [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1)
- [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1/candidate_gate_summary.md)
- generated report table: [`prompt_contract_wave4_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave4_probe_gates.csv)
- generated failure table: [`prompt_contract_wave4_probe_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave4_probe_failure_modes.csv)

| Contract | Exact | Executable | Delta exact vs no-directive | Dominant failure | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| `visual_state_tool_selection_v4` | `0.125` | `0.0` | `+0.125` | `no_tool_call` | weak exact gain |

Wave four is a useful negative result. It reproduces the familiar one-case exact gain, but does not improve executable visual recovery in the raw probe and does not reduce the dominant no-tool-call failure pattern enough to justify H1 spend on its own.

## Prompt-Contract Wave Five Probe Gate

The fifth executed candidate packet is now recorded at:

- [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1)
- [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1/candidate_gate_summary.md)
- generated report table: [`prompt_contract_wave5_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave5_probe_gates.csv)
- generated failure table: [`prompt_contract_wave5_probe_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave5_probe_failure_modes.csv)

| Contract | Exact | Executable | Delta exact vs no-directive | Dominant failure | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| `visual_refine_selection_v5` | `0.0` | `0.0` | `0.0` | `no_tool_call` | no probe gain |

Wave five is an even sharper negative result. It tried to make the remaining visual hypothesis more surgical by naming the latest-selection filtering transition and `refine_selection`, but raw probe behavior got worse rather than better: no exact cases, no executable visual recovery, and six no-tool-call failures. Per the current gate, it should not spend CLI-live replay or H1 budget.

## Tool-Catalog Profile Probe Gate

The catalog-profile packets are recorded at:

- [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe)
- [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe)
- [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe)
- [`results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe`](../../results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe)
- generated report table: [`tool_catalog_profile_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_profile_probe_gates.csv)

| Catalog profile | Exact | Executable | Delta exact vs no-directive | Probe gate |
| --- | ---: | ---: | ---: | --- |
| `visual_role_catalog_v1` | `0.125` | `1.0` | `+0.125` | improved vs no-directive |
| `visual_role_catalog_argument_hints_v2` | `0.25` | `0.0` | `+0.25` | improved vs no-directive |
| `visual_role_catalog_split_selector_hints_v3` | `0.125` | `0.0` | `+0.125` | improved vs no-directive |
| `visual_role_catalog_schema_field_hints_v4` | `0.25` | `0.0` | `+0.25` | improved vs no-directive |

This is the first visual intervention after wave five that changed the failure class in the desired direction. With the exact directive still disabled and no prompt contract attached, the tool catalog profile:

- moved raw exact from `0 / 8` to `1 / 8`
- restored the executable visual-form target to `1 / 1`
- changed `visual_latest_filter_literal` from no-call or wrong-tool behavior into the correct tool with an argument mismatch: `refine_selection(selection_id="sel-001", filter_query="latest issue")` instead of canonical `filter_query="latest"`
- preserved the exact readback case: `read_region_text(image_id="img-form-latest", region_id="form-err-202")`

Interpretation: visual role presentation inside the tool catalog is doing real routing work. It is not sufficient for exact-call fidelity, but it narrows the problem from "choose/enter the right visual tool" to "preserve literal selector arguments after choosing the right tool."

The v2 argument-hints profile is the first narrow follow-up that improves that literal selector problem. It raises raw exactness from `1 / 8` to `2 / 8` by making `visual_latest_filter_literal` exact and preserving exact readback. The tradeoff is also real: `visual_form_target_literal` falls from executable paraphrase to non-executable argument mismatch (`target_query="recruiter note"` instead of canonical `validation error`). The generated delta table is:

- [`tool_catalog_argument_hints_vs_role_catalog_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_argument_hints_vs_role_catalog_case_deltas.csv)

The v3 split-selector profile is a negative follow-up. It preserves the v2 latest-filter exact case, but raw exactness falls back to `1 / 8` and `visual_readback_region_literal` regresses because the model emits `tool_name` instead of `name`. Live replay was intentionally skipped:

- [`tool_catalog_split_selector_vs_argument_hints_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_split_selector_vs_argument_hints_case_deltas.csv)
- [`tool_catalog_split_selector_live_replay_decision.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_split_selector_live_replay_decision.csv)

The v4 schema-field profile is negative only on the original focused probe/replay slice. It avoids broad selector prose and restores the readback JSON shape, but it does not beat v2 there. Raw exactness stays `2 / 8`, executable visual-form recovery stays `0 / 1`, and `visual_form_target_literal` changes into a wrong-tool style failure: `refine_selection(selection_id="latest", filter_query="phone issue")`. Live replay was intentionally skipped for that focused slice:

- [`tool_catalog_schema_field_hints_vs_argument_hints_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_argument_hints_case_deltas.csv)
- [`tool_catalog_schema_field_hints_vs_split_selector_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_split_selector_case_deltas.csv)
- [`tool_catalog_schema_field_hints_vs_role_catalog_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_role_catalog_case_deltas.csv)
- [`tool_catalog_schema_field_hints_live_replay_decision.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_live_replay_decision.csv)

That negative focused-slice result is not the full story. The next visual surface was therefore not another tweak to the same three replay cases. The design packet at [`results/reports/visual_hard_slice_design/design.md`](../../results/reports/visual_hard_slice_design/design.md) defines eight fresh cases to separate visible-region targeting, valid selection carryover, compact filter copying, and readback protocol shape. Those cases are now executed in [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1), which includes the v5 target-literal repair attempt.

## Visual Hard-Slice Probe Gate

The executed hard slice is the cleanest current visual discriminator because it uses independently authored cases instead of repeatedly tuning against the original three visual replay failures:

- packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1)
- packet-local gate summary: [`candidate_gate_summary.md`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/candidate_gate_summary.md)
- v5-vs-v4 comparison: [`schema_literal_targets_vs_schema_field_hints`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints)
- generated gate table: [`visual_hard_slice_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_probe_gates.csv)
- generated family table: [`visual_hard_slice_family_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_family_summary.csv)
- generated case deltas: [`visual_hard_slice_case_deltas_vs_no_directive.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_case_deltas_vs_no_directive.csv) and [`visual_hard_slice_case_deltas_vs_contracted.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_case_deltas_vs_contracted.csv)
- exactness diagnostic: [`results/reports/visual_hard_slice_exactness_diagnostic`](../../results/reports/visual_hard_slice_exactness_diagnostic)

| System profile | Strict | Executable | Executor Eq | Dominant failure | Gate |
| --- | ---: | ---: | ---: | --- | --- |
| contracted MLX | `8 / 8` | `8 / 8` | `8 / 8` | exact | contracted reference |
| no-directive MLX | `1 / 8` | `1 / 8` | `1 / 8` | no tool call | no-directive reference |
| visual role catalog v1 | `3 / 8` | `3 / 8` | `3 / 8` | argument mismatch | improved vs no-directive |
| visual role catalog + argument hints v2 | `6 / 8` | `7 / 8` | `7 / 8` | exact | improved vs no-directive |
| visual role catalog + split selector v3 | `5 / 8` | `6 / 8` | `6 / 8` | exact | improved vs no-directive |
| visual role catalog + schema-field hints v4 | `6 / 8` | `8 / 8` | `8 / 8` | exact | improved vs no-directive |
| visual role catalog + schema target literals v5 | `5 / 8` | `7 / 8` | `7 / 8` | exact | improved vs no-directive |
| visual role catalog + literal guard v6 | `3 / 8` | `4 / 8` | `4 / 8` | exact | improved vs no-directive |

This changes the visual interpretation. Schema-field hints were not a focused-replay promotion candidate, but on the fresh hard slice they become the strongest no-directive harness profile because they preserve full executor-equivalent target success. The exact directive still matters: contracted MLX is the only row with `8 / 8` strict protocol fidelity. The exactness diagnostic resolves the immediate ambiguity: v4's two non-exact rows are executor-success selector aliases, while v5 keeps those same aliases and adds a true stale-selection wrong-tool failure. Executor-equivalence is now a first-class packet metric, so the next useful work is a packaged H1 visual workflow rather than another target-query wording patch.

## Visual Hard-Slice Exactness Diagnostic

| System | Exact | Executable | Non-exact executor success | Label-artifact candidates | True harness failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| visual role catalog + schema-field hints v4 | `6 / 8` | `8 / 8` | `2` | `2` | `0` |
| visual role catalog + schema target literals v5 | `5 / 8` | `7 / 8` | `2` | `2` | `1` |

The two v4 exact gaps are:

| Case | Expected target | Actual target | Interpretation |
| --- | --- | --- | --- |
| `visual_metric_panel_vs_table_selector` | `hard-metric-1001` | `hard-metric-1001` | benchmark-label artifact candidate |
| `visual_callout_warning_with_user_decoy` | `hard-callout-decoy-1102` | `hard-callout-decoy-1102` | benchmark-label artifact candidate |

This is an important research distinction. Strict JSON/canonical-label exactness is still useful because it measures interface fidelity, but it should not be confused with executor-visible success. For publication, this supports separating "protocol fidelity" from "local executor target success" in the metric table.

## Visual Hard-Slice CLI-Live Replay Matrix

The packaged H1l visual workflows saturated, so the live operator path now preserves the raw hard-slice replay shape directly:

- source replay packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1)
- generated summary table: [`visual_hard_slice_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_summary.csv)
- generated case table: [`visual_hard_slice_live_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_case_deltas.csv)
- generated figure: [`visual_hard_slice_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_live_replay_gate.svg)

| Live row vs no-directive | Exact delta | Executable delta | Executor-eq delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| contracted MLX | `+1.0` | `+1.0` | `+1.0` | upper bound, `2 / 2` strict and executor-equivalent |
| visual role catalog v1 | `+0.5` | `+0.5` | `+0.5` | recovers only the stale-selection decoy |
| argument hints v2 | `+0.5` | `+0.5` | `+0.5` | matches role catalog on this two-case slice |
| schema-field hints v4 | `+0.5` | `+1.0` | `+1.0` | strongest no-directive row; one strict hit plus one executor-equivalent selector alias |
| schema target literals v5 | `0.0` | `+0.5` | `+0.5` | negative strict result; stale-selection decoy becomes wrong-tool |

This is the cleanest live evidence so far that the schema-field catalog helps local MLX visual grounding without fully replacing the final tool-turn directive.

## Visual Hard-Slice Stress CLI-Live Replay Matrix

The follow-up stress packet repeats the same mechanisms with four fresh decoys:

- stress replay packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1)
- generated summary table: [`visual_hard_slice_stress_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_stress_live_replay_summary.csv)
- generated case table: [`visual_hard_slice_stress_live_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_stress_live_replay_case_deltas.csv)
- generated figure: [`visual_hard_slice_stress_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_stress_live_replay_gate.svg)

| Stress row vs no-directive | Exact delta | Executable delta | Executor-eq delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| contracted MLX | `+0.5` | `+0.25` | `+0.25` | strict upper bound, `4 / 4` |
| visual role catalog v1 | `-0.25` | `-0.25` | `-0.25` | regresses on a stale-selection warning decoy |
| argument hints v2 | `0.0` | `0.0` | `0.0` | tied with no-directive |
| schema-field hints v4 | `0.0` | `+0.25` | `+0.25` | recovers executor-equivalence on the hardest metric-panel decoy |
| schema target literals v5 | `0.0` | `+0.25` | `+0.25` | same executor-equivalence gain as v4, still no strict gain |

This makes the research conclusion more precise: schema-local visual catalog hints improve executor-visible grounding on alias-heavy metric panels, but strict JSON/canonical-label fidelity still belongs to the full contracted MLX profile. The next useful move is additional alias/decoy repetition before packaging an H1m workflow.

### Alias-Repeat Matrix

The alias-repeat follow-up expands the four-case stress slice to eight repeated metric-panel, callout, and stale-selection cases:

- alias-repeat packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1)
- generated summary table: [`visual_hard_slice_alias_repeat_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_repeat_live_replay_summary.csv)
- generated case table: [`visual_hard_slice_alias_repeat_live_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_repeat_live_replay_case_deltas.csv)
- generated figure: [`visual_hard_slice_alias_repeat_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_repeat_live_replay_gate.svg)
- diagnostic: [`results/reports/visual_alias_repeat_diagnostic/diagnostic.md`](../../results/reports/visual_alias_repeat_diagnostic/diagnostic.md)

| Row | Strict | Executable | Executor-equivalent |
| --- | ---: | ---: | ---: |
| no-directive MLX | `2 / 8` | `5 / 8` | `5 / 8` |
| contracted MLX | `7 / 8` | `8 / 8` | `8 / 8` |
| role catalog v1 | `1 / 8` | `6 / 8` | `6 / 8` |
| argument hints v2 | `2 / 8` | `6 / 8` | `6 / 8` |
| schema-field hints v4 | `2 / 8` | `7 / 8` | `7 / 8` |
| schema target literals v5 | `3 / 8` | `8 / 8` | `8 / 8` |

This repeats the same story under more alias pressure but adds nuance. Schema-field hints improve executor-visible visual grounding by `+0.25` without improving strict canonical-label fidelity. Schema target literals v5, which was negative on the preserved two-case hard replay, is positive on this repeated alias slice: it reaches full executor-equivalence and adds one strict exact win. The contracted row still owns strict fidelity, so the next step is not to declare v5 "fixed"; it is to repeat the alias packet or package the surviving metric-panel/callout mechanisms into a non-saturated H1m workflow.

## Prompt-Contract Wave Six Probe Gate

Wave six tests whether adding the existing literal-argument guard on top of the visual role catalog closes that remaining literal mismatch:

- [`results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe`](../../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe)
- generated report table: [`prompt_contract_wave6_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave6_probe_gates.csv)
- generated failure table: [`prompt_contract_wave6_probe_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave6_probe_failure_modes.csv)

| Contract | Catalog profile | Exact | Executable | Delta exact vs no-directive | Dominant failure | Recommendation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `literal_argument_guard_v1` | `visual_role_catalog_v1` | `0.125` | `0.0` | `+0.125` | `argument_mismatch` | weak exact gain |

This is a negative composition result. The combined profile keeps the single exact readback gain, but loses the catalog-only executable visual-form recovery and introduces no-tool-call regressions on two non-visual cases. It should not move to live replay or H1 as currently written.

## Prompt-Contract Promotion Decisions

The generated report now writes a machine-readable promotion table:

- [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv)

The current decisions are intentionally conservative:

- weak one-case exact gains are held for exact-probe live replay
- visual executable-only gains are held for visual replay, not general H1 promotion
- no-gain parallel/no-call candidates are rejected for H1 promotion

That table is the guardrail for the next loop: a prompt-contract candidate should not graduate because it sounds plausible; it should graduate only after it moves exact probe behavior or reduces controller burden on a harder live discriminator.

## Exact-Probe Replay Packet

The first replay packet is now recorded at:

- [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1)
- executed packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1)
- contracted replay packet: [`results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1)
- replay A/B comparison: [`results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1`](../../results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1)
- brief: [`docs/continuity/exact-probe-replay.md`](../continuity/exact-probe-replay.md)

It contains all `8` failed no-directive exact-call cases:

| Failure mode | Count |
| --- | ---: |
| `argument_mismatch` | `4` |
| `no_tool_call` | `4` |

The generated next-action table splits the implementation backlog into:

| Next action | Count |
| --- | ---: |
| `build_canonical_argument_replay` | `4` |
| `build_visual_state_replay_executor` | `3` |
| `build_parallel_array_replay_or_workflow` | `1` |

This packet is deliberately a dry-run replay artifact, not a packaged live workflow. It preserves each case's messages, media, allowed tool specs, expected calls, no-directive actual calls, and contracted baseline context. The next implementation should either turn the deferred parallel case into a faithful packaged workflow or make these exact probe cases operator-visible in a replay executor.

The executed replay packet reproduced the raw failure set exactly: exact match stayed `0 / 8`, the four argument-mismatch cases remained argument mismatches, and the four no-tool-call cases remained no-tool-call cases. That makes this a stable mechanism target rather than a one-off probe artifact.

The contracted replay packet restores `7 / 8` exact on the same cases, with the only non-exact case remaining executable through visual selector aliasing. This is now the cleanest current A/B artifact for the model-side tool-turn directive.

The generated replay comparison records a `-0.875` exact-rate delta for no-directive versus contracted MLX on the same replay case set.

Focused replay slices now isolate each `next_action` family:

| Slice | Contracted exact | No-directive exact | Delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| canonical arguments | `4 / 4` | `0 / 4` | `-1.0` | Final directive is carrying exact CLI/API argument canonicalization. |
| visual no-call | `2 / 3` plus one executable paraphrase | `0 / 3` | `-0.667` | Visual cases still need a stateful/executable replay path before another broad H1 run. |
| parallel array | `1 / 1` | `0 / 1` | `-1.0` | The missing faithful live workflow should preserve independent two-call array shape. |

Generated replay tables:

- [`exact_probe_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_case_deltas.csv)
- [`exact_probe_replay_family_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_family_deltas.csv)
- [`exact_probe_replay_focus_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_focus_summary.csv)
- [`live_parallel_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/live_parallel_replay_case_deltas.csv)
- [`live_visual_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/live_visual_replay_case_deltas.csv)
- [`live_canonical_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/live_canonical_replay_case_deltas.csv)

## CLI-Live Parallel Replay

The same `parallel_audit_array_literal` case has now been run through the live terminal replay entrypoint:

- contracted packet: [`results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1)
- no-directive packet: [`results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1)
- comparison: [`results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1)

Result:

| Row | Exact | Actual calls | Failure |
| --- | ---: | ---: | --- |
| contracted | `1 / 1` | `2` | `exact` |
| no directive | `0 / 1` | `0` | `no_tool_call` |

This is the key bridge from raw replay to live CLI testing. H1k packaged workflow execution saturated because it decomposed the parallel audit into staged tasks. The CLI-live replay keeps the exact one-turn shape and reproduces the same model-side contract gap under an operator-visible terminal surface.

The visual no-call family has the same live shape:

- contracted packet: [`results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1)
- no-directive packet: [`results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1)
- comparison: [`results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1)

| Row | Exact | Actual calls | Failure shape |
| --- | ---: | ---: | --- |
| contracted | `2 / 3` | `3` | two exact, one executable paraphrase |
| no directive | `0 / 3` | `0` | all no tool call |

Together, the parallel and visual live replay packets make the current conclusion more precise: packaged workflow completion can saturate, but exact live replay still shows the model-side directive carrying the raw tool-protocol behavior.

The live comparison tables also record executable-match deltas. That matters for `visual_form_target_literal`: contracted MLX is non-exact but executable, while no-directive MLX makes no call at all.

The canonical CLI/API cases complete the live replay set:

- contracted packet: [`results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1)
- no-directive packet: [`results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1)
- comparison: [`results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1)

| Row | Exact | Actual calls | Failure shape |
| --- | ---: | ---: | --- |
| contracted | `4 / 4` | `4` | all exact |
| no directive | `0 / 4` | `4` | all argument mismatch |

This is the cleanest distinction between protocol entry and argument fidelity: no-directive MLX calls tools, but it does not preserve canonical arguments.

## Wave Three CLI-Live Candidate Replay

Wave three was gated through live replay only where the raw probe showed movement:

- canonical candidate packet: [`results/tool_probe_replay_live/20260507T_canonical_argument_canonical_json_copy_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_canonical_json_copy_live_execute_v1)
- visual candidate packet: [`results/tool_probe_replay_live/20260507T_visual_state_visual_tool_initiation_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_visual_tool_initiation_live_execute_v1)
- generated summary table: [`wave3_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/wave3_live_candidate_replay_summary.csv)
- generated case table: [`wave3_live_candidate_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/wave3_live_candidate_case_deltas.csv)

| Comparison | Baseline exact | Candidate exact | Delta exact | Candidate executable | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| canonical JSON vs no-directive | `0.0` | `0.0` | `0.0` | n/a | no live gain; two cases regress to no-call |
| canonical JSON vs contracted | `1.0` | `0.0` | `-1.0` | n/a | still far below the directive |
| visual initiation vs no-directive | `0.0` | `0.3333333333333333` | `+0.3333333333333333` | `1.0` | first candidate with live visual-family movement |
| visual initiation vs contracted | `0.6666666666666666` | `0.3333333333333333` | `-0.3333333333333333` | `1.0` | still misses one visual referent case with the wrong tool |

This is the most useful wave-three finding. Generic canonical JSON copy is not enough; it can even reduce protocol entry on some canonical cases. Visual tool initiation is the first candidate that improves live exact replay and executable visual recovery over no-directive, but it is not yet a replacement for the final tool-turn directive.

## Wave Four CLI-Live Candidate Replay

Wave four was gated through the same visual live replay surface:

- visual state/tool-selection candidate packet: [`results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1)
- comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1)
- comparison vs contracted: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1)
- generated summary table: [`wave4_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/wave4_live_candidate_replay_summary.csv)
- generated case table: [`wave4_live_candidate_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/wave4_live_candidate_case_deltas.csv)

| Comparison | Baseline exact | Candidate exact | Delta exact | Candidate executable | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| visual state tool selection vs no-directive | `0.0` | `0.3333333333333333` | `+0.3333333333333333` | `0.0` | preserves one exact visual recovery over no-directive |
| visual state tool selection vs contracted | `0.6666666666666666` | `0.3333333333333333` | `-0.3333333333333333` | `0.0` | below directive and loses executable visual-form recovery |

Case read:

| Case | v4 replay failure | Interpretation |
| --- | --- | --- |
| `visual_form_target_literal` | `no_tool_call` | worse than wave-three visual initiation, which made an executable visual call |
| `visual_latest_filter_literal` | `wrong_tool` | the targeted remaining failure did not move |
| `visual_readback_region_literal` | `exact` | preserves the one exact visual referent recovery |

The conclusion is tighter now: visual-initiation wording was doing the useful work in wave three. Additional generic visual state/tool-selection rules did not solve the wrong-tool filter case and cost the executable form-target recovery.

The refreshed visual tool-choice diagnostic packet makes the transition explicit: wave three and wave four emit `extract_layout` when `visual_latest_filter_literal` expects `refine_selection`; the catalog profile emits `refine_selection` and only misses the literal `filter_query`. That means the catalog changed routing, and the next harness change should preserve that routing while repairing selector literalness.

## Visual Catalog CLI-Live Candidate Replay

The visual role catalog was promoted to live exact replay because its raw probe moved both exact and executable behavior:

- candidate packet: [`results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1)
- comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_no_directive_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_no_directive_v1)
- comparison vs visual initiation: [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_tool_initiation_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_tool_initiation_v1)
- comparison vs visual state/tool-selection: [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1)
- generated summary table: [`visual_catalog_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_catalog_live_candidate_replay_summary.csv)
- generated case table: [`visual_catalog_live_candidate_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_catalog_live_candidate_case_deltas.csv)

| Comparison | Baseline exact | Candidate exact | Delta exact | Candidate executable | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| visual role catalog vs no-directive | `0.0` | `0.3333333333333333` | `+0.3333333333333333` | `1.0` | enters the visual tool protocol in all three cases |
| visual role catalog vs visual initiation | `0.3333333333333333` | `0.3333333333333333` | `0.0` | `1.0` | same exact ceiling, but cleaner targeted tool choice on latest-filter |
| visual role catalog vs visual state tool | `0.3333333333333333` | `0.3333333333333333` | `0.0` | `1.0` | restores executable form-target recovery and changes wrong-tool to argument mismatch |

Case read:

| Case | Catalog replay failure | Interpretation |
| --- | --- | --- |
| `visual_form_target_literal` | `executable_paraphrase` | calls `extract_layout` and reaches the right executor target, but uses `target_query="phone issue"` instead of canonical `validation error` |
| `visual_latest_filter_literal` | `argument_mismatch` | now calls the right tool, `refine_selection`, but expands canonical `latest` into `latest issue` |
| `visual_readback_region_literal` | `exact` | preserves the exact readback behavior already recovered by the best visual candidates |

This is the stable routing baseline for the visual catalog path. The remaining problem is no longer broad visual tool initiation or state selection. It is literal visual argument preservation after the tool catalog has made the right role separable.

## Visual Catalog Argument-Hints CLI-Live Candidate Replay

The argument-hints catalog profile was promoted to live exact replay because its raw probe fixed the targeted latest-filter literal:

- candidate packet: [`results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1)
- comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1)
- comparison vs contracted: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1)
- comparison vs role catalog: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1)
- generated summary table: [`visual_catalog_argument_hints_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_catalog_argument_hints_live_candidate_replay_summary.csv)
- generated case table: [`visual_catalog_argument_hints_live_candidate_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_catalog_argument_hints_live_candidate_case_deltas.csv)

| Comparison | Baseline exact | Candidate exact | Delta exact | Candidate executable | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| argument hints vs no-directive | `0.0` | `0.6666666666666666` | `+0.6666666666666666` | `0.0` | recovers two exact visual follow-on calls |
| argument hints vs contracted | `0.6666666666666666` | `0.6666666666666666` | `0.0` | `0.0` | matches exact contracted MLX on this slice but loses executable rescue |
| argument hints vs role catalog | `0.3333333333333333` | `0.6666666666666666` | `+0.3333333333333333` | `0.0` | fixes latest-filter exactly, regresses form-target executability |

Case read:

| Case | Argument-hints replay failure | Interpretation |
| --- | --- | --- |
| `visual_form_target_literal` | `argument_mismatch` | calls `extract_layout`, but the selector no longer reaches the executable target |
| `visual_latest_filter_literal` | `exact` | preserves `refine_selection(selection_id="sel-001", filter_query="latest")` |
| `visual_readback_region_literal` | `exact` | preserves exact readback |

On the original focused visual replay, this is the best focused-replay exact no-directive candidate, but not a full harness replacement. It proves catalog-level argument semantics can fix selector literal drift after routing succeeds. It also proves that selector hints can overconstrain or misdirect the form-target case. The fresh hard-slice result below updates the broader recommendation: v2 remains the exact focused-replay reference, while v4 is now the stronger executable hard-slice profile.

## H1i Candidate Packet Result

The H1i mechanism-probe packet is now recorded at:

- [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet)
- [`tool_contract_summary.md`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet/tool_contract_summary.md)
- generated report table: [`h1i_prompt_contract_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_candidate_metrics.csv)

All five rows matched:

| Row | Readiness | Repair | Fallback | Arg repair | Raw clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| contracted | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |
| no directive | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |
| schema anchor | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |
| literal guard | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |
| tool required | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |

This means H1i did not discriminate after the probe gate. The earlier H1i no-directive packet showed controller burden; this candidate packet sampled clean no-directive tool calls across the same four workflow families. The practical next step is not to declare the prompt contracts solved. It is to define a harder second-stage packet with repeated no-directive trials or probe-derived live cases where exact protocol failure is stable.

The first concrete second-stage lever has now been executed:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1i_slice.yaml \
  --packet-id mlx_prompt_contract_candidates \
  --run-group-id 20260507T_h1i_prompt_contract_candidates_repeat3_v1 \
  --repeat 3
```

Result:

- packet: [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet)
- runs: `60` episode traces, `5` rows x `4` workflow families x `3` repeats
- all five rows matched readiness `0.97710`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`
- trace analysis found `0` note events and `0` failure candidates
- generated report table: [`h1i_prompt_contract_repeat3_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_repeat3_metrics.csv)

This is a useful negative result. Repeating the saturated H1i candidate packet did not recover the earlier H1i no-directive controller-burden signal. The current H1i packaged workflows are too deterministic to validate prompt-contract candidates after the probe gate. The next harder packet should be probe-derived live cases, especially visual/parallel no-call cases.

## H1j Probe-Derived Packet Result

H1j moved from worst-family attribution to exact probe-family attribution:

- config: [`configs/knowledge_work_h1j_slice.yaml`](../../configs/knowledge_work_h1j_slice.yaml)
- packet: [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet)
- generated table: [`h1j_probe_derived_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1j_probe_derived_candidate_metrics.csv)
- runs: `30` traces, `5` rows x `6` packaged live workflow families

Result:

| Row | Readiness | Repair | Fallback | Arg repair | Raw clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| contracted | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |
| no directive | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |
| schema anchor | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |
| literal guard | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |
| tool required | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |

Trace analysis again found `0` note events and `0` failure candidates. This makes the current research picture sharper: the raw probe remains a better discriminator than benchmark-style packaged workflow execution, even when the packaged workflow set is selected from the same probe failure families.

The paired helper-ablation packet is now recorded at:

- packet: [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet)
- generated table: [`h1j_probe_derived_helper_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1j_probe_derived_helper_metrics.csv)
- runs: `30` traces, `5` rows x `6` packaged live workflow families

It also saturated. Contracted, no-directive, no-controller-repair, no-controller-fallback, and no-argument-repair rows all matched readiness `0.96577`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, and raw clean `1.0`. Trace mining recorded `21` `controller_repair_disabled` markers on the disabled-repair row, but `0` failure candidates.

That matters: controller repair remains causal on H1h/H1i, but not on H1j. H1j is currently evidence that the packaged workflow route can wash out raw probe failures, not evidence that the model-side contract is fixed.

## H1k Parallel-Audit Packet Result

H1k closed the last packaged-workflow gap left by H1j: the deferred parallel no-call probe case.

- config: [`configs/knowledge_work_h1k_slice.yaml`](../../configs/knowledge_work_h1k_slice.yaml)
- workflow: `ops_parallel_audit_review`
- replay pressure: `parallel_audit_array_literal`
- candidate packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet)
- helper packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet)
- generated candidate table: [`h1k_parallel_audit_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1k_parallel_audit_candidate_metrics.csv)
- generated helper table: [`h1k_parallel_audit_helper_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1k_parallel_audit_helper_metrics.csv)

Both H1k packets saturated:

| Packet | Rows | Readiness | Strict/recovered | Repair/fallback/arg repair | Raw clean | Failure candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate | `5` | `0.91780` | `1.0 / 1.0` | `0.0 / 0.0 / 0.0` | `1.0` | `0` |
| helper | `5` | `0.91780` | `1.0 / 1.0` | `0.0 / 0.0 / 0.0` | `1.0` | `0` |

The negative result is important. H1k proves Moonie now has a safe packaged live scaffold for the parallel-audit family, but that staged workflow is easier than the raw one-turn replay case. It decomposes the pressure into known task steps, and no-directive MLX stays controller-clean even when controller repair, controller fallback, or argument repair are removed.

So the next move is not another packaged H1 derivative. The next discriminator should preserve exact-call replay shape, especially the independent two-call array behavior in `parallel_audit_array_literal`.

## H1l Visual Executor-Equivalence Candidate Packet

H1l tests whether the visual hard-slice executor-equivalence split survives current packaged visual workflows:

- config: [`configs/knowledge_work_h1l_slice.yaml`](../../configs/knowledge_work_h1l_slice.yaml)
- packet: [`results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet)
- generated table: [`h1l_visual_executor_equivalence_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1l_visual_executor_equivalence_candidate_metrics.csv)
- generated figure: [`h1l_visual_executor_equivalence_burden.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/h1l_visual_executor_equivalence_burden.svg)

Result:

| Rows | Workflows per row | Readiness | Strict/recovered | Repair/fallback/arg repair | Raw clean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `6` | `5` | `0.90406` | `0.85 / 0.8` | `0.0 / 0.0 / 0.0` | `1.0` |

The result is negative but useful. Schema-field hints v4 still matter on the hard slice because they separate strict exactness from executor-equivalent target success, but current packaged visual workflows are too staged to preserve that distinction. The H1l helper packet should wait until a visual live surface separates at least one candidate row.

## H1m Visual Alias-Repeat Candidate Packet

H1m tests whether the harder eight-case visual alias-repeat replay signal survives a narrower packaged visual workflow set:

- config: [`configs/knowledge_work_h1m_slice.yaml`](../../configs/knowledge_work_h1m_slice.yaml)
- packet: [`results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet)
- generated table: [`h1m_visual_alias_repeat_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1m_visual_alias_repeat_candidate_metrics.csv)
- generated figure: [`h1m_visual_alias_repeat_burden.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/h1m_visual_alias_repeat_burden.svg)

Result:

| Rows | Workflows per row | Readiness | Strict/recovered | Repair/fallback/arg repair | Raw clean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `6` | `3` | `0.87783` | `0.75 / 0.667` | `0.0 / 0.0 / 0.0` | `1.0` |

This is the second packaged visual saturation result after H1l. It matters because it prevents an overclaim: the alias-repeat replay matrix separates rows, but the current packaged workflows do not. The correct next move is not H1m helper ablation; there is no candidate separation to attribute. The next visual evidence should preserve replay shape, repeat alias cases, or use a less staged CLI live task.

## Packaged Replay Gap Diagnostic

The packaged replay gap diagnostic makes the design lesson explicit:

- diagnostic: [`results/reports/packaged_replay_gap_diagnostic/diagnostic.md`](../../results/reports/packaged_replay_gap_diagnostic/diagnostic.md)
- surface table: [`packaged_replay_gap_surfaces.csv`](../../results/reports/packaged_replay_gap_diagnostic/tables/packaged_replay_gap_surfaces.csv)

Result:

| Surface | Max replay executor-equivalence delta | Packaged readiness span | Packaged strict span | Classification |
| --- | ---: | ---: | ---: | --- |
| H1l visual executor-equivalence | `1.0` | `0.0` | `0.0` | positive replay, saturated packaged surface |
| H1m visual alias-repeat | `0.375` | `0.0` | `0.0` | positive replay, saturated packaged surface |

This is now a research finding about benchmark construction. Packaged workflows are still useful for safe live operation, attribution, and operator observability, but they can become too staged to preserve one-turn visual alias/decoy failures. The paper should report this as contract quality affecting measured capability, not as a Gemma-only behavior.

## H1n Visual Alias-Transfer Replay Matrix

H1n is the first post-packaging-gap visual replay result. It keeps the `moonie-agent replay-live` surface but changes the labels and decoys:

- source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1)
- oracle packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2)
- diagnostic: [`results/reports/visual_alias_transfer_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_diagnostic/diagnostic.md)
- oracle diagnostic: [`results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md)
- generated table: [`visual_hard_slice_alias_transfer_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_live_replay_summary.csv)
- oracle table: [`visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv)
- generated figure: [`visual_hard_slice_alias_transfer_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_transfer_live_replay_gate.svg)
- oracle figure: [`visual_hard_slice_alias_transfer_oracle_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_transfer_oracle_live_replay_gate.svg)

Legacy v1 result:

| Row | Strict | Executor-equivalent |
| --- | ---: | ---: |
| no-directive | `0 / 6` | `2 / 6` |
| contracted | `5 / 6` | `1 / 6` |
| role catalog v1 | `1 / 6` | `3 / 6` |
| argument hints v2 | `1 / 6` | `6 / 6` |
| schema-field hints v4 | `1 / 6` | `2 / 6` |
| schema target literals v5 | `1 / 6` | `4 / 6` |

The v1 result changed the visual story but exposed a benchmark-contract problem: `5 / 6` generated expected-call contracts did not satisfy the packet's own expected-execution oracle. Contracted MLX was strict-best against planner-derived expected calls, but it had `4` exact-but-not-executor rows. That means v1 strict exactness measured planner-call fidelity more than visual target success.

Oracle v2 result:

| Row | Strict | Executor-equivalent |
| --- | ---: | ---: |
| no-directive | `2 / 6` | `2 / 6` |
| contracted | `1 / 6` | `1 / 6` |
| role catalog v1 | `3 / 6` | `3 / 6` |
| argument hints v2 | `5 / 6` | `6 / 6` |
| schema-field hints v4 | `2 / 6` | `2 / 6` |
| schema target literals v5 | `4 / 6` | `4 / 6` |

The oracle replay fixes that scorer ambiguity by making the serialized expected calls execute to the target visual regions and by preserving those calls during replay-live scoring. Under this cleaner contract, argument hints v2 is the H1n winner, schema target literals v5 is second, and contracted prompting is not an upper bound. This is now one of the strongest evidence points for the paper thesis: benchmark contract quality changes what we think local Gemma is good at.

## Gemini CLI Baseline Status

The Gemini CLI adapter is currently a dry-run external baseline, not a replacement for Moonie. The packet uses the same H1h workflow families and records prompt/command artifacts without external side effects:

- [`results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1`](../../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1)
- `workflow_count = 10`
- `dry_run_count = 10`
- `available_count = 0` for the intentionally missing binary in that packet

Use it as:

- a CLI ergonomics reference
- a trust and sandboxing reference
- an external-baseline packet once a real binary run is explicitly desired

Do not use it as:

- a substitute for local Gemma harnessing
- evidence that Moonie's local MLX tool contract is solved
- a merged same-harness leaderboard row

## Interpretation

The current result has three layers:

1. Contracted MLX is strong on the H1 live workflow packets.
2. No-directive MLX can still be made to finish the work.
3. The ability to finish without the directive is mostly controller-mediated.
4. Visual tool-catalog role presentation can recover tool entry and correct visual tool choice, but exact argument fidelity remains unsolved.

That distinction matters because Moonie's research goal is not merely to produce good final artifacts. It is to understand what makes Gemma harnessable as a local operator. If final readiness hides controller dependence, then the harness is doing the work and the model-side contract remains weak.

The H1h -> H1i narrowing is also useful methodologically. H1h proves the phenomenon across the full ten-workflow live set. H1i turns the worst H1h workflow-family attribution into a cheap, repeatable packet for prompt-contract experiments.

## Next Experiments

Use this order before broad `32 / 26` reruns:

1. Treat `visual_role_catalog_argument_hints_v2` as the best exact candidate on the original focused visual replay, not as the overall visual answer.
2. Treat `visual_role_catalog_schema_field_hints_v4` as the best fresh hard-slice no-directive profile because it reaches `8 / 8` executor-equivalent target success, while still missing exact protocol on two cases.
3. Treat `visual_role_catalog_schema_literal_targets_v5` as negative evidence: it drops to `5 / 8` strict exactness and `7 / 8` executor-equivalent target success while adding a wrong-tool stale-selection regression.
4. Treat H1l and H1m as negative packaged-workflow results: current packaged visual workflows wash out hard-slice and alias-repeat row separation.
5. Treat H1n oracle v2 as the current non-packaged visual restart point: argument hints v2 is the clean winner at `5 / 6` strict and `6 / 6` executor-equivalent, with schema target literals v5 second at `4 / 6`.
6. Treat `visual_role_catalog_v1` as the stable routing baseline, `visual_state_tool_selection_v4` as a failed-to-improve live candidate, `visual_refine_selection_v5` as a raw-gate rejection, and the v6 catalog-plus-literal-guard composition as negative interference.
7. Stop iterating on standalone visual prompt rules unless the next idea changes tool-catalog role shape or generation-time argument copying without sacrificing protocol entry.
8. Keep canonical JSON copy and parallel two-call wording out of H1 as currently written; they did not earn live promotion.
9. H1h only after replay-live, raw probe, hard-slice, or less staged live evidence shows a mechanism-level change.
10. Gemini CLI real execution only when the binary/run environment is explicitly meant to be part of the comparison.
11. Treat H2v as the semantic boundary that broke H2u: H2u is transfer-positive but not semantic-complete because genuine negated values and stale-example context still fail.
12. Treat H2w as replay-transfer-clean and H2x/H2y-positive under CLI semantic pressure: it repairs H2v to `10 / 10`, preserves `109 / 109` strict/executor-equivalent on the current transfer/back-compat battery, reaches `8 / 8` on H2x, and reaches `12 / 16` on H2y.
13. Treat H2x and H2y as fallback-independent evidence: H2u/H2w no-fallback controls tie their full-controller rows on both gates, so the gain is semantic-preservation-causal rather than fallback-causal.
14. Treat H2y as the current boundary: the next helper should target stale-selection negation and short component-query collapse, not broader prompt prose.
15. Treat local MLX replay-live backtests as low-concurrency workloads unless proven otherwise; the H2w wave produced a Metal GPU timeout under four-way parallel replay but completed cleanly sequentially.
16. Runtime live-smoke packets after benchmark movement, to confirm the CLI operator path sees the same repair/fallback pattern.

Acceptance criteria for a useful candidate:

- improves no-directive exact-call or executable-call probe behavior
- improves H1i no-directive raw-clean rate or reduces repair/fallback burden
- does not hide failures by broadening controller rescue
- preserves approval-safe stop behavior and sandbox policy events
- leaves generated report artifacts attributable to workflow families

## Reporting Discipline

Generated reporting now has three layers:

- packet-local summaries under each H1/probe/Gemini output directory
- generated cross-packet report artifacts under [`results/reports/mlx_tool_contract_harnessing`](../../results/reports/mlx_tool_contract_harnessing)
- this curated narrative report under [`docs/reports`](.)
- dry-run or executed prompt-contract probe packets under [`results/tool_prompt_contract_probe_packets`](../../results/tool_prompt_contract_probe_packets)

When a new H1i/H1h/probe wave runs, update the report by rerunning:

```bash
uv run python scripts/build_visual_hard_slice_design.py
uv run python scripts/run_visual_hard_slice_probe_packet.py --run-group-id <timestamp>_visual_hard_slice_probe --execute
uv run python scripts/build_h2v_semantic_negation_synthesis.py
uv run python scripts/build_h2w_semantic_target_preservation_synthesis.py
uv run python scripts/build_h2w_transfer_backtest_synthesis.py
uv run python scripts/build_h2x_cli_semantic_pressure_synthesis.py
uv run python scripts/build_mlx_tool_contract_report.py
uv run python scripts/build_publication_evidence_ledger.py
uv run python scripts/audit_publication_readiness.py
uv run pytest tests/test_mlx_tool_contract_report.py -q
```

Then update this document only if the interpretation changes.
