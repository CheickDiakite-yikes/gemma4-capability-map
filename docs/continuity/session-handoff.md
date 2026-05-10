# Session Handoff

## Resume Here

The current restart point is now:

- [`docs/continuity/cli-live-harness-pivot.md`](./cli-live-harness-pivot.md)

Use that file first in a new chat.

It supersedes the old assumption that the next main workstream is React workspace refinement.

## Latest Overlay

The freshest research/reporting entrypoint is now:

- [`docs/reports/mlx-tool-contract-harnessing.md`](../reports/mlx-tool-contract-harnessing.md)
- generated packet: [`results/reports/mlx_tool_contract_harnessing/report.md`](../../results/reports/mlx_tool_contract_harnessing/report.md)
- CLI-live replay brief: [`docs/continuity/live-exact-replay-results.md`](./live-exact-replay-results.md)

Treat the older H1/HF and React notes below as historical context unless they are explicitly referenced by the current H1i or CLI-live workstream.

Current strongest MLX result:

- H1h proves the no-directive causal ordering across all ten live workflow families.
- H1i compresses the worst H1h workflow families into the current fast loop.
- contracted MLX on H1i is clean at readiness `0.97710`, strict/recovered `1.0 / 1.0`, raw clean `1.0`.
- no-directive MLX on H1i stays top-line clean only with controller repair/fallback/argument repair `1.00 / 0.50 / 0.50` and raw clean `0.00`.
- no-directive + no controller repair falls to readiness `0.64697`, strict/recovered `0.297 / 0.000`.
- the no-directive probe falls from contracted exact `7 / 8` to `0 / 8`.
- the first prompt-contract candidate probe gate is partial-gain only:
  - [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1)
  - `schema_anchor_v1` recovers one exact case (`0.125`)
  - `literal_argument_guard_v1` and `tool_required_parallel_v1` recover the executable visual target but not exact JSON copy
- the H1i candidate packet saturated:
  - [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet)
  - all five rows matched at readiness `0.97710`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`
  - trace analysis found `0` failure candidates
- the H1i repeat3 packet also saturated:
  - [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet)
  - `60` traces across five rows, four workflow families, and three repeats
  - all rows matched at readiness `0.97710`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`
  - trace analysis found `0` notes and `0` failure candidates
- the H1j probe-derived candidate packet also saturated:
  - [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet)
  - `30` traces across five rows and six probe-derived live workflow families
  - all rows matched at readiness `0.96577`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`
  - trace analysis found `0` notes and `0` failure candidates
- the H1j helper-ablation packet also saturated:
  - [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet)
  - no-repair, no-fallback, and no-argument-repair rows matched baseline at readiness `0.96577`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - trace analysis found `21` disabled-repair markers but `0` failure candidates
- the second prompt-contract wave is executed and remains only a partial-gain result:
  - [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1)
  - `schema_literal_tool_required_v2`: exact `0.125`, executable `0.0`, weak exact gain
  - `visual_next_call_state_v2`: exact `0.0`, executable `1.0`, visual executable gain only
  - `parallel_array_required_v2`: exact `0.0`, executable `0.0`, no probe gain
- the third prompt-contract wave is executed and live-gated:
  - probe packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1)
  - `canonical_json_copy_v3`: probe exact `0.125`, live canonical exact `0 / 4`; do not promote
  - `visual_tool_initiation_v3`: probe exact `0.125`, probe executable `1.0`, live visual exact `1 / 3`, live executable visual target recovered
  - `parallel_two_call_array_v3`: probe exact `0.0`, executable `0.0`; reject as written
  - candidate live summary: [`results/reports/mlx_tool_contract_harnessing/tables/wave3_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/wave3_live_candidate_replay_summary.csv)
  - interpretation: visual tool initiation is a real partial improvement, but not a directive replacement. It set up the wave-four wrong-tool visual referent test.
- the fourth prompt-contract wave is executed and live-gated:
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_dry_run_v1)
  - probe packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1)
  - `visual_state_tool_selection_v4`: probe exact `0.125`, probe executable `0.0`, recommendation `weak_exact_gain`
  - live visual packet: [`results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1), exact `1 / 3`, executable visual target not recovered
  - comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1)
  - comparison vs contracted: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1)
  - interpretation: v4 did not improve the wave-three ceiling. `visual_latest_filter_literal` still fails as `wrong_tool`, and `visual_form_target_literal` regressed to `no_tool_call`.
- the fifth prompt-contract wave is executed and rejected before live replay:
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_dry_run_v1)
  - probe packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1)
  - `visual_refine_selection_v5`: probe exact `0.0`, probe executable `0.0`, recommendation `no_probe_gain`
  - interpretation: surgical `refine_selection` wording did not preserve tool initiation and should not spend CLI-live replay or H1 budget.
- the visual role catalog profile is now the stable visual routing baseline:
  - isolated catalog probe: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe)
  - raw result: exact `0.125`, executable `1.0`, delta exact vs no-directive `+0.125`
  - live replay: [`results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1), exact `1 / 3`, executable visual target recovered
  - comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_no_directive_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_no_directive_v1)
  - comparison vs wave four: [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1)
  - interpretation: it converts the targeted remaining visual failure from `wrong_tool`/no-call into `argument_mismatch`, so the next visual problem is literal argument preservation after correct tool routing.
- the visual role catalog argument-hints profile is now the best focused-replay exact visual no-directive candidate:
  - profile: `visual_role_catalog_argument_hints_v2`
  - isolated probe: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe)
  - raw result: exact `0.25`, executable `0.0`, delta exact vs no-directive `+0.25`
  - live replay: [`results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1), exact `2 / 3`
  - comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1)
  - comparison vs contracted: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1)
  - comparison vs v1 catalog: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1)
  - interpretation: it fixes `visual_latest_filter_literal` exactly and preserves exact readback, matching contracted MLX at `2 / 3` exact on the focused visual replay. It is not solved because it loses the v1/contracted executable `visual_form_target_literal` recovery, and the fresh hard slice now gives a broader read.
- the visual split-selector profile is negative evidence:
  - profile: `visual_role_catalog_split_selector_hints_v3`
  - probe packet: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe)
  - comparison vs v2: [`results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2`](../../results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2)
  - skipped-live decision: [`results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1`](../../results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1)
  - interpretation: v3 stayed below v2 on exactness, did not recover executable form targeting, and regressed readback JSON shape
- the visual schema-field profile is split evidence:
  - profile: `visual_role_catalog_schema_field_hints_v4`
  - probe packet: [`results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe`](../../results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe)
  - comparison vs v2: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2)
  - comparison vs v3: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_split_selector_v3`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_split_selector_v3)
  - comparison vs v1: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_role_catalog_v1`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_role_catalog_v1)
  - skipped-live decision: [`results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1`](../../results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1)
  - focused-slice interpretation: v4 ties v2 at raw exact `2 / 8`, restores readback versus v3, but remains `0 / 1` executable and over-prefers `refine_selection` on the original form-target case
  - fresh hard-slice interpretation: v4 is the strongest no-directive candidate on independently authored visual cases at exact `6 / 8` and executable `8 / 8`, while still trailing contracted MLX exactness
- paper-facing artifacts now exist:
  - evidence ledger: [`results/reports/publication_evidence_ledger/ledger.md`](../../results/reports/publication_evidence_ledger/ledger.md)
  - publication readiness audit: [`results/reports/publication_readiness_audit/publication_readiness_audit.md`](../../results/reports/publication_readiness_audit/publication_readiness_audit.md)
  - paper outline: [`docs/paper/moonie-gemma-harnessing-paper-outline.md`](../paper/moonie-gemma-harnessing-paper-outline.md)
- executed visual hard-slice packet now exists:
  - design packet: [`results/reports/visual_hard_slice_design/design.md`](../../results/reports/visual_hard_slice_design/design.md)
  - runner: [`scripts/run_visual_hard_slice_probe_packet.py`](../../scripts/run_visual_hard_slice_probe_packet.py)
  - dry-run packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_dry_run_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_dry_run_v1)
  - first executed packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_execute_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_execute_v1)
  - latest executed packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1)
  - latest gate summary: [`candidate_gate_summary.md`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/candidate_gate_summary.md)
  - v5-vs-v4 comparison: [`schema_literal_targets_vs_schema_field_hints`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints)
  - exactness-vs-executor diagnostic: [`results/reports/visual_hard_slice_exactness_diagnostic`](../../results/reports/visual_hard_slice_exactness_diagnostic)
  - case count: `8`
  - result: contracted MLX `8 / 8` strict/executable/executor-equivalent; no-directive MLX `1 / 8` strict/executable/executor-equivalent; schema-field hints v4 `6 / 8` strict and `8 / 8` executor-equivalent; schema-target-literal v5 `5 / 8` strict and `7 / 8` executor-equivalent
  - exactness diagnostic: v4's two non-exact rows are executor-success selector aliases, while v5 adds a true stale-selection wrong-tool failure
  - status: current visual prompt-contract restart point, not a packaged H1 promotion yet
- replay-shaped visual hard-slice CLI-live matrix now preserves the same discriminator:
  - source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1)
  - summary table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_summary.csv)
  - result: no-directive `0 / 2`; contracted MLX `2 / 2`; role catalog v1 and argument hints v2 `1 / 2`; schema-field hints v4 `1 / 2` strict and `2 / 2` executor-equivalent; schema-target-literal v5 `0 / 2` strict and `1 / 2` executor-equivalent
  - stress follow-up: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1)
  - stress result: no-directive `2 / 4` strict and `3 / 4` executor-equivalent; contracted MLX `4 / 4`; schema-field hints v4 and schema-target-literal v5 `2 / 4` strict and `4 / 4` executor-equivalent
  - alias-repeat follow-up: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1)
  - alias-repeat diagnostic: [`results/reports/visual_alias_repeat_diagnostic/diagnostic.md`](../../results/reports/visual_alias_repeat_diagnostic/diagnostic.md)
  - alias-repeat result: no-directive `2 / 8` strict and `5 / 8` executor-equivalent; contracted MLX `7 / 8` strict and `8 / 8` executor-equivalent; schema-field hints v4 `2 / 8` strict and `7 / 8` executor-equivalent; schema-target-literal v5 `3 / 8` strict and `8 / 8` executor-equivalent
  - status: replay-shaped signal is positive; H1m packaged promotion below shows the current packaged surface washes it out
- H1m packaged promotion result:
  - config: [`configs/knowledge_work_h1m_slice.yaml`](../../configs/knowledge_work_h1m_slice.yaml)
  - brief: [`docs/continuity/h1m-slice.md`](./h1m-slice.md)
  - workflows: `executive_visual_dashboard_revision`, `jobs_visual_latest_issue_review`, `finance_visual_invoice_hold_review`
  - packet ids: `mlx_visual_alias_repeat_packaged_candidates`, `mlx_visual_alias_repeat_helper_ablation`
  - executed packet: [`results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet)
  - status: negative packaged-workflow result; all six rows tie at readiness `0.87783`, strict `0.75`, recovered `0.667`, raw clean `1.0`, and zero controller burden
  - next: skip helper ablation until a visual live surface separates rows; preserve alias-repeat replay shape or make less staged non-packaged live tasks
- packaged replay gap diagnostic:
  - diagnostic: [`results/reports/packaged_replay_gap_diagnostic/diagnostic.md`](../../results/reports/packaged_replay_gap_diagnostic/diagnostic.md)
  - result: `2 / 2` visual promotion surfaces have positive replay gains but zero packaged readiness/strict span
  - next: treat packaged workflow design as part of the benchmark contract, not just a neutral execution wrapper
- H1n alias-transfer replay design:
  - brief: [`docs/continuity/h1n-slice.md`](./h1n-slice.md)
  - packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1)
  - diagnostic: [`results/reports/visual_alias_transfer_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_diagnostic/diagnostic.md)
  - contract-split diagnostic: [`results/reports/h1n_alias_transfer_contract_split/diagnostic.md`](../../results/reports/h1n_alias_transfer_contract_split/diagnostic.md)
  - result: argument hints v2 reaches `1 / 6` strict and `6 / 6` executor-equivalent; schema target literals v5 reaches `1 / 6` strict and `4 / 6` executor-equivalent; no-directive is `0 / 6` strict and `2 / 6` executor-equivalent; contracted is `5 / 6` strict but `1 / 6` executor-equivalent
  - contract finding: `5 / 6` generated expected-call contracts miss the visual oracle, so strict exactness currently means planner-call fidelity; next rebuild H1n with oracle expected calls before promotion
- the sixth prompt-contract wave is a negative composition result:
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_dry_run`](../../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_dry_run)
  - probe packet: [`results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe`](../../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe)
  - candidate: `literal_argument_guard_v1` + `visual_role_catalog_v1`
  - raw result: exact `0.125`, executable `0.0`, delta exact vs no-directive `+0.125`
  - interpretation: broad literal-guard wording interfered with the catalog profile and should not move to live replay or H1.
- visual tool-choice diagnostics are now available:
  - script: [`scripts/analyze_visual_tool_choice_diagnostics.py`](../../scripts/analyze_visual_tool_choice_diagnostics.py)
  - packet: [`results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1`](../../results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1)
  - finding: wave-three and wave-four visual candidates choose `extract_layout` when `visual_latest_filter_literal` expects `refine_selection`; the catalog profile reaches `refine_selection` and only misses the literal selector.
- exact-probe replay is now scaffolded and recorded:
  - brief: [`docs/continuity/exact-probe-replay.md`](./exact-probe-replay.md)
  - packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1)
  - executed packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1)
  - contracted replay packet: [`results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1)
  - replay comparison: [`results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1`](../../results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1)
  - `8` failed no-directive probe cases, split into `4` argument mismatches and `4` no-tool-call cases
  - each case has messages, media, tool specs, expected calls, source actual calls, baseline context, and a runnable `run_tool_directive_probe.py --case-id <case>` command
  - execution reproduced the source failures exactly: `0 / 8` exact, same `4` argument mismatches and same `4` no-tool-call failures
  - contracted replay restored `7 / 8` exact, with the remaining visual paraphrase executable
  - comparison records no-directive exact-rate delta `-0.875` versus contracted
- CLI-live exact replay is now executed for all eight source failures:
  - canonical argument live comparison: [`results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1), contracted exact `4 / 4`, no-directive exact `0 / 4`, actual-call delta `0`
  - visual live comparison: [`results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1), contracted exact `2 / 3`, no-directive exact `0 / 3`, no-directive failures all `no_tool_call`
  - parallel live comparison: [`results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1), contracted exact `1 / 1`, no-directive exact `0 / 1`, actual-call delta `-2`
  - operator command: `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 --case-id <case_id> --execute`
- H1k parallel-audit packaged workflow is now a negative result:
  - candidate packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet)
  - helper packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet)
  - all rows stayed clean at readiness `0.91780`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`
  - interpretation: staged packaged workflows are safe and attributable, but still weaker than exact replay for exposing the one-turn parallel no-call failure

Current next loop:

1. Treat H1i, H1j, and H1k packaged packets as saturated or non-discriminating for current prompt-contract validation.
2. Treat waves one through six as partial-gain or negative/composition evidence, not fixes.
3. Treat `visual_role_catalog_argument_hints_v2` as the best exact candidate on the old focused visual replay, with the explicit caveat that it lost executable form-target recovery.
4. Treat `visual_role_catalog_schema_field_hints_v4` as the strongest fresh hard-slice no-directive candidate because it reaches `8 / 8` executor-equivalent, with the explicit caveat that it still misses exact protocol on two cases.
5. Treat `visual_role_catalog_schema_literal_targets_v5` as negative evidence: it did not fix the two v4 executable paraphrases and introduced a wrong-tool stale-selection regression.
6. Treat the two v4 exact misses as benchmark-label artifact candidates under the current local executor, not true executor-targeting failures.
7. Treat H1l as a negative packaged-workflow result: the current packaged visual workflows saturate and do not preserve v4's hard-slice executor-equivalence discriminator.
8. Treat the completed alias-repeat matrix as positive but still replay-shaped: schema-field hints improves executor-equivalence by `+0.25`, schema target literals reaches full executor-equivalence with a small strict gain, and contracted MLX remains the strict upper bound.
9. Return to H1h only after replay-live or raw/hard-slice evidence shows a mechanism-level change.
10. Keep Gemini CLI as an external baseline/reference, not a replacement for Moonie's local Gemma harness.
11. Regenerate the report, publication ledger, publication audit, and visual hard-slice packet summaries after any H1i/H1h/probe/Gemini/live-replay packet change.

H1l source:

- config: [`configs/knowledge_work_h1l_slice.yaml`](../../configs/knowledge_work_h1l_slice.yaml)
- brief: [`docs/continuity/h1l-slice.md`](./h1l-slice.md)
- candidate packet id: `mlx_visual_executor_equivalence_candidates`
- executed candidate packet: [`results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet)
- helper packet id: `mlx_visual_executor_equivalence_helper_ablation`

H1j source:

- config: [`configs/knowledge_work_h1j_slice.yaml`](../../configs/knowledge_work_h1j_slice.yaml)
- brief: [`docs/continuity/h1j-slice.md`](./h1j-slice.md)
- note: `parallel_audit_array_literal` remains deferred until a faithful live packaged workflow exists

Wave 2 source:

- contracts: `schema_literal_tool_required_v2`, `visual_next_call_state_v2`, `parallel_array_required_v2`
- dry-run packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1)
- executed packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1)

The current research seam is no longer “make the rows tie.”

That part is done on the aligned exploratory `32 / 26` surface.

The current seam is:

- reduce HF Gemma specialist controller burden further
- without losing the current aligned readiness tier
- shift the main live-testing surface to a CLI-first sandboxed operator harness
- use Gemini CLI as a design reference and external baseline, not a replacement

The latest live-harness gain is now CLI-first:

- sessions and runtime traces carry sandbox metadata
- packaged workflow runs get a per-session sandbox with copied workflow/episode inputs
- native artifacts and runtime summaries write under the sandbox output root
- live-web dry-run holds are emitted as `sandbox_policy_block` events and stored on sessions/traces
- `moonie-agent live` launches a packaged workflow and attaches a Rich terminal operator view
- `moonie-agent attach <session_id>` watches an existing run from the terminal
- `moonie-agent attach <session_id> --action approve|deny|resume|retry|quit` applies operator actions from the same terminal path
- `moonie-agent inspect <session_id>` inspects sandbox, artifact, policy-block, and summary metadata
- a real `mlx_gemma4_e2b_reasoner_only` CLI smoke completed on `executive_visual_dashboard_review`
- `moonie-agent gemini-baseline` prepares dry-run Gemini CLI baseline packets for packaged workflows
- `H1 v1` is defined as the next packaged-workflow-first harder slice:
  - [`configs/knowledge_work_h1_slice.yaml`](../../configs/knowledge_work_h1_slice.yaml)
  - [`docs/continuity/h1-slice.md`](./h1-slice.md)
- `scripts/run_knowledge_work_h1_slice.py` validates H1 and delegates filtered runs to the existing KWA arena runner
- second-wave ablation controls now exist for intent priority, argument repair, and deterministic visual follow-on
- H1 HF ablation should use [`scripts/run_knowledge_work_h1_ablation_packet.py`](../../scripts/run_knowledge_work_h1_ablation_packet.py) so the ablation rows share one warmed HF service-backed bundle
- an attempted in-process H1 ablation launch was stopped pre-child-manifest after roughly ten minutes; no episode results were produced from that attempt
- H1 primary replayable MLX Gemma completed cleanly:
  - [`results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1)
  - `real_world_readiness_avg = 0.9749800000000001`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- H1 service-backed HF Gemma ablation completed cleanly after the FunctionGemma prompt patch:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet)
  - baseline `hf_service_gemma4_specialists_cpu`: `real_world_readiness_avg = 0.9749800000000001`
  - baseline controller burden is now `controller_repair_avg = 0.8`, `controller_fallback_avg = 0.3`, `raw_planning_clean_rate_avg = 0.2`
  - `no_controller_repair = 0.7319`
  - `no_controller_fallback = 0.8606`
  - `no_visual_rescue = 0.9749800000000001`
  - `no_intent_priority = 0.9749800000000001`
  - `no_argument_repair = 0.9749800000000001`
  - `no_deterministic_visual_follow_on = 0.9749800000000001`
  - interpretation: H1 still confirms repair/fallback are causal on HF Gemma; the prompt patch materially softened fallback-disabled failures, while second-wave helper toggles still did not move readiness on this slice
- H1 trace-note mining is now reusable:
  - [`scripts/analyze_knowledge_work_h1_traces.py`](../../scripts/analyze_knowledge_work_h1_traces.py)
  - [`trace_note_counts.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet/trace_note_counts.csv)
  - [`trace_episode_failures.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet/trace_episode_failures.csv)
  - [`trace_failure_mode_counts.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet/trace_failure_mode_counts.csv)
  - current read: `93` controller-note events, `7` strict/recovered failure candidates, baseline `controller_fallback_planner` appears `3` times across `3` H1 episodes
  - aggregate failure modes: `raw_refusal = 5`, `repair_disabled = 4`, `fallback_disabled = 3`, `argument_repair = 2`, `fallback_planner = 2`
  - previous aggregate `generic_tool_name = 7` is now gone
  - next target: raw refusal/no-call and unrepaired real-tool placeholder arguments, not visual rescue
- FunctionGemma prompt canary after removing the literal `call:tool_name{arg:...}` hint:
  - [`results/knowledge_work_h1_slice/20260506T_h1_functiongemma_prompt_canary_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_functiongemma_prompt_canary_v1_knowledge_work_ablation_packet)
  - readiness stayed `0.9749800000000001`
  - `controller_fallback_avg` moved from `0.6` to `0.3`
  - `controller_repair_avg` moved from `0.9` to `0.8`
  - `raw_planning_clean_rate_avg` moved from `0.1` to `0.2`
  - `argument_repair_avg` rose from `0.1` to `0.5`
- FunctionGemma concrete request-specific hint canary:
  - [`results/knowledge_work_h1_slice/20260506T_h1_functiongemma_concrete_hint_canary_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_functiongemma_concrete_hint_canary_v1_knowledge_work_ablation_packet)
  - readiness stayed `0.9749800000000001`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
  - trace miner found `0` controller-note events and `0` failure candidates
  - next empirical run should be the full H1 service-backed ablation packet after this stronger prompt prior
- Full concrete-hint H1 ablation completed:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet)
  - baseline stayed `0.9749800000000001` readiness with `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - `no_controller_fallback` now matches baseline at `0.9749800000000001`
  - `no_controller_repair = 0.88748`
  - `no_deterministic_visual_follow_on = 0.88748`
  - trace miner found `42` controller-note events and `6` failure candidates
  - aggregate failure modes after the richer visual taxonomy are now `visual_readback_missing = 6`, `visual_stepwise_control = 6`, `fallback_planner = 4`, `argument_repair = 3`, `raw_refusal = 3`, `repair_disabled = 3`, `visual_follow_on = 3`, `visual_repeated_refinement = 3`
  - interpretation: fallback causality was prompt-artifact-heavy; repair and deterministic visual follow-on now expose the remaining stepwise visual-control seam

The prior React product-side gain remains useful context:

- the React workspace now runs against the real API in a live loop
- the shell uses backend health plus long-poll session streaming
- a fresh `mlx_gemma4_e2b_reasoner_only` session was launched from the UI and observed through completion
- the stream payload now wins over stale session-list snapshots so the rail settles correctly after completion

## Current Source Runs

Aligned comparison surface:

- HF Gemma controller-burden rerun:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26)
- oracle + MLX Gemma aligned reference:
  - [`results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26)
- MLX Qwen aligned reference:
  - [`results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26)

Focused replayable Gemma packet:

- [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet)

## Latest Headline Readout

Replayable `32`:

- oracle:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.578125`
  - `controller_fallback_avg = 0.0`
- HF Gemma specialists:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.71875`
  - `controller_fallback_avg = 0.28125`
  - `raw_planning_clean_rate_avg = 0.46875`
- MLX Qwen:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
- MLX Gemma:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`

Live `26`:

- oracle:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.7115384615384616`
- HF Gemma specialists:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.8076923076923077`
  - `controller_fallback_avg = 0.23076923076923078`
- MLX Qwen:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
- MLX Gemma:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.0`

## What Just Changed

The latest pass added the first CLI-first live harness slice.

Code path:

- [`src/gemma4_capability_map/runtime/sandbox.py`](../../src/gemma4_capability_map/runtime/sandbox.py)
- [`src/gemma4_capability_map/runtime/operator.py`](../../src/gemma4_capability_map/runtime/operator.py)
- [`src/gemma4_capability_map/runtime/cli.py`](../../src/gemma4_capability_map/runtime/cli.py)
- [`src/gemma4_capability_map/runtime/core.py`](../../src/gemma4_capability_map/runtime/core.py)
- [`tests/test_runtime_core.py`](../../tests/test_runtime_core.py)
- [`tests/test_runtime_cli.py`](../../tests/test_runtime_cli.py)
- [`tests/test_runtime_api.py`](../../tests/test_runtime_api.py)

What that means:

- `moonie-agent live` is now the active live-entry scaffold for packaged workflows
- `moonie-agent attach` provides a Rich terminal operator view
- new live runs are sandboxed by default with policy id `packaged_workflow_ephemeral_v1`
- runtime artifacts, summaries, and traces are attributable to the sandbox root
- live-web sandbox-only or approval-gated actions now produce explicit policy block metadata
- attach actions can approve, deny, resume, retry, or quit from the Rich operator path
- inspect commands can show sandbox roots, artifacts, policy blocks, and trace/summary paths as Rich output or JSON

Verification:

- `uv run pytest tests/test_runtime_core.py tests/test_runtime_cli.py tests/test_runtime_api.py`
- latest targeted run: `24 passed`
- `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py`
- latest operator inspect/action run: `21 passed`
- `uv run moonie-agent inspect <latest_session> --target sandbox --json`
- completed and showed the sandbox root plus manifest path
- `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --once --refresh-s 0.5 --timeout-s 1.0`
- completed session: `20260506T173247139289Z_executive_visual_dashboard_review`
- smoke metrics: `strict_interface_score = 1.0`, `role_readiness_score = 0.9942`, `controller_repair_count = 0.5`, `controller_fallback_count = 0.0`, `raw_planning_clean_rate = 0.5`
- `uv run pytest tests/test_runtime_gemini_cli.py tests/test_runtime_cli.py`
- Gemini adapter scaffold: `11 passed`
- `uv run moonie-agent gemini-baseline --workflow-id executive_visual_dashboard_review --lane replayable_core --output-dir tmp/gemini-baseline-smoke`
- completed as a dry-run packet with `/usr/local/bin/gemini` detected
- `uv run pytest tests/test_knowledge_work_h1.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py`
- H1 runner/config scaffold: `17 passed`
- `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set primary --lane replayable_core --output-root tmp/h1-dry-run-smoke --run-group-id 20260506T_h1_dry_run_smoke`
- completed and wrote a dry-run manifest for one primary replayable H1 run
- `uv run pytest tests/test_tool_planner.py tests/test_trace_metrics.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py tests/test_knowledge_work_h1.py`
- second-wave ablation control scaffold: `64 passed`
- `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set ablation --lane replayable_core --output-root tmp/h1-ablation-dry-run-smoke --run-group-id 20260506T_h1_ablation_dry_run_smoke`
- completed and wrote `7` replayable H1 ablation run specs
- `uv run pytest`
- full repo suite after H1 + second-wave controls: `260 passed`
- `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set all --lane replayable_core --output-root tmp/h1-all-dry-run-smoke --run-group-id 20260506T_h1_all_dry_run_smoke`
- completed and wrote `10` replayable H1 run specs
- `uv run python scripts/run_knowledge_work_h1_slice.py --run-set primary --lane replayable_core --system-id mlx_gemma4_e2b_reasoner_only --run-group-id 20260506T_h1_mlx_gemma_primary_v1`
- completed with `5 / 5` H1 replayable episodes and `failed_runs = 0`
- `uv run pytest tests/test_knowledge_work_h1.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py`
- service-backed specialist mapping after the v1 failure: `21 passed`
- `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_ablation_v2`
- completed with `7` service-backed H1 ablation rows and `5` replayable episodes each
- `uv run pytest tests/test_knowledge_work_trace_analysis.py`
- H1 trace analyzer: `2 passed`
- `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet`
- completed and wrote trace-note summary, note counts, and failure candidates
- `uv run pytest tests/test_functiongemma_runner.py tests/test_tool_parsing.py tests/test_knowledge_work_trace_analysis.py`
- FunctionGemma prompt patch: `4 passed`
- `uv run python scripts/run_knowledge_work_ablation_packet.py --lane replayable_core --bundle-system-id hf_service_gemma4_specialists_cpu --output-root results/knowledge_work_h1_slice --run-group-id 20260506T_h1_functiongemma_prompt_canary_v1 --run-intent exploratory --system-id hf_service_gemma4_specialists_cpu --episode-id kwa_exec_visual_dashboard_brief --episode-id kwa_exec_backlog_resume_hold_v5 --episode-id kwa_jobs_email_block_resume_hold_v5 --episode-id kwa_finance_diff_review_hold_v5 --episode-id kwa_finance_invoice_lock_direction_hold_v4`
- completed with `5 / 5` H1 replayable baseline episodes and `controller_fallback_avg = 0.3`
- `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_prompt_patch_ablation_v1`
- completed with `7` service-backed H1 ablation rows and `5` replayable episodes each after the FunctionGemma prompt patch
- `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet`
- completed and wrote post-prompt trace-note summary, failure candidates, and failure-mode counts
- `uv run pytest tests/test_functiongemma_runner.py tests/test_tool_planner.py tests/test_knowledge_work_h1.py tests/test_knowledge_work_trace_analysis.py`
- concrete FunctionGemma hint patch: `51 passed`
- `uv run python scripts/run_knowledge_work_ablation_packet.py --lane replayable_core --bundle-system-id hf_service_gemma4_specialists_cpu --output-root results/knowledge_work_h1_slice --run-group-id 20260506T_h1_functiongemma_concrete_hint_canary_v1 --run-intent exploratory --system-id hf_service_gemma4_specialists_cpu --episode-id kwa_exec_visual_dashboard_brief --episode-id kwa_exec_backlog_resume_hold_v5 --episode-id kwa_jobs_email_block_resume_hold_v5 --episode-id kwa_finance_diff_review_hold_v5 --episode-id kwa_finance_invoice_lock_direction_hold_v4`
- completed with `5 / 5` H1 replayable baseline episodes, `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
- `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_concrete_hint_ablation_v1`
- completed with `7` service-backed H1 ablation rows and `5` replayable episodes each after the concrete FunctionGemma prompt hint
- `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`
- completed and wrote post-concrete-hint trace-note summary, failure candidates, and failure-mode counts
- `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id oracle_gemma4_e2b --lane replayable_core --refresh-s 0.1 --timeout-s 0.5`
- completed through the Rich operator view with sandbox context visible
- `uv run pytest`
- earlier full live-harness suite: `244 passed`

## Prior Change

The latest pass added deterministic runtime execution for obvious visual follow-ons.

Code path:

- [`src/gemma4_capability_map/runtime/core.py`](../../src/gemma4_capability_map/runtime/core.py)
- [`src/gemma4_capability_map/tools/planner.py`](../../src/gemma4_capability_map/tools/planner.py)
- [`tests/test_tool_planner.py`](../../tests/test_tool_planner.py)
- [`tests/test_smoke_eval.py`](../../tests/test_smoke_eval.py)
- [`tests/test_trace_metrics.py`](../../tests/test_trace_metrics.py)

What that means:

- after a successful `extract_layout` or `refine_selection`, the runtime now auto-executes deterministic `refine_selection` / `read_region_text` follow-ons
- the runtime no longer asks the model again for those same obvious visual steps

## Measured Effect

Focused packet delta versus the prior packet:

- readiness unchanged at `0.9627777777777777`
- `controller_repair_avg` improved from `2.3333333333333335` to `0.8888888888888888`
- `feedback_prior:refine_selection` dropped from `16` to `0`
- `feedback_prior:read_region_text` dropped from `10` to `0`
- `controller_fallback_planner` stayed at `8`

Aligned full-lane delta for HF Gemma specialists:

- replayable:
  - `controller_repair_avg` improved from `1.296875` to `0.71875`
  - `controller_fallback_avg` stayed `0.28125`
  - readiness stayed `0.976853125`
- live:
  - `controller_repair_avg` improved from `1.5192307692307692` to `0.8076923076923077`
  - `controller_fallback_avg` stayed `0.23076923076923078`
  - readiness stayed `0.9791653846153847`

Interpretation:

- the old visual follow-on repairs were inflating controller burden
- removing them did not reduce the actual causal value of repair/fallback
- the remaining burden is now more honestly concentrated in fallback planner and non-visual repair families

## What Not To Re-Learn

Do not spend time re-proving:

- aligned top-line readiness parity exists
- MLX Gemma’s earlier executive-assistant judgment miss is closed
- MLX Qwen is a real same-surface comparator
- the direct in-process Gemma reasoner-only control is still materially weaker on the older reproduced surface

## Next Best Move

1. Follow the CLI live harness pivot file first.
Primary targets:
   - runtime sandbox model
   - `moonie-agent live`
   - `moonie-agent attach`
   - Rich terminal operator harness

2. Expand the H1 trace taxonomy around visual multi-call batches, future selection ids, and missing deterministic follow-ons before another broad same-surface rerun.

3. Attack the remaining HF Gemma specialist note families directly.
Primary targets:
   - stepwise visual-control failures when controller repair is disabled
   - missing deterministic visual follow-ons
   - `controller_fallback_planner`
   - `repaired_arguments:refine_selection`
Now scaffolded toggles:
   - `disable_intent_priority`
   - `disable_argument_repair`
   - `disable_deterministic_visual_follow_on`

4. Keep using the focused replayable packet first.
Only rerun the aligned `32 / 26` surface after the packet shifts again.

5. Use Gemini CLI as a wrapped reference/baseline after the CLI live harness exists.

6. If the next question becomes runtime posture instead of controller dependence, switch to installing the Gemma `31B` local `GGUF` artifact and run the first real `llama.cpp` row.

## Verification State

Current code-side verification from the latest CLI live harness patch:

- targeted runtime/API/CLI suite from the original live-harness slice: `22 passed`
- full suite after the CLI/H1/ablation-control pivot: `260 passed`
- H1 all-run-set dry-run: `10` replayable run specs

Benchmark outputs rebuilt:

- [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- [`results/history/knowledge_work_history.md`](../../results/history/knowledge_work_history.md)

## Operational Notes

- `output/` and `tmp/` remain untracked scratch dirs
- the Gemma `31B` lane is still blocked by local artifact availability:
  - `GEMMA4_31B_GGUF_PATH` unset
  - no local bundle under `/Users/cheickdiakite/models`
