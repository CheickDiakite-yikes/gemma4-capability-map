# Research Log

# Research Log

## 2026-05-12 - H2m Less-Direct Overreach Packet Scaffold

- Added H2m as the less-direct follow-up to H2l:
  - suite: `h2m_less_direct_target_normalization_overreach_v19`
  - builder: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
  - dry-run packet: [`results/tool_probe_replay_packets/20260512T_h2m_less_direct_target_normalization_overreach_dry_run_v1`](../results/tool_probe_replay_packets/20260512T_h2m_less_direct_target_normalization_overreach_dry_run_v1)
- Packet shape:
  - `8` cases
  - `4` less-direct value-bearing target cases, preserving longer labels such as `result badge Blocked`
  - `2` contextual alias-is-target cases (`error notice`, `result tile`)
  - `2` less-direct H2k regression guards (`status badge`, `mode field`)
- Research purpose:
  - H2l did not show over-normalization, but its prompts directly said "The target is ..."
  - H2m removes that phrasing and asks whether H2j can still preserve legitimate longer labels and aliases while retaining H2k-style short-label repair
  - this keeps the next work replay-shaped and controller-attributable instead of returning to broad prompt prose
- Next execution:
  - run H2j on H2m first
  - run H2j-without-stale-selection second
  - run H2e third as the no-target-normalizer control
  - compare H2j against H2e and no-stale H2j before adding any new profile
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py::test_visual_hard_slice_live_stress_packet_supports_h2m_less_direct_overreach_suite -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --suite h2m_less_direct_target_normalization_overreach_v19 --run-group-id 20260512T_h2m_less_direct_target_normalization_overreach_dry_run_v1`

## 2026-05-12 - H2l Overreach Holdout Supports Target-Normalization Scope

- Executed the H2l target-normalization overreach holdout across the three relevant profiles:
  - H2j target-query normalization: [`results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2j_execute_v1`](../results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2j_execute_v1)
  - H2j without stale-selection gate: [`results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2j_no_stale_gate_execute_v1`](../results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2j_no_stale_gate_execute_v1)
  - H2e route arbitration plus stale-selection gate: [`results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2e_execute_v1`](../results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2e_execute_v1)
- Result:
  - full H2j: `8 / 8` strict and `8 / 8` executor-equivalent
  - H2j without stale-selection: `8 / 8` strict and `8 / 8` executor-equivalent
  - H2e: `7 / 8` strict and `7 / 8` executor-equivalent
  - H2j-vs-H2e delta: `+0.125` strict exact-rate and `+0.125` executor-equivalence-rate
  - H2j-vs-no-stale delta: `0.0` strict exact-rate and `0.0` executor-equivalence-rate
- Mechanism read:
  - H2l did not expose over-stripping on value-bearing targets such as `result badge Blocked`, `state tag Closed`, `mode toggle Manual`, or `priority badge Critical`
  - H2l did not expose over-stripping on alias-is-target rows (`error notice`, `result tile`)
  - the only H2e miss is the regression guard `h2l_status_badge_short_label_regression_guard`, where H2e produced `critical chip` instead of `status badge`
  - full H2j and no-stale H2j both record one `visual_target_query_normalization` intervention on that case: `critical chip` -> `status badge`
  - both rows record `0` stale-selection interventions, so H2l continues the H2k finding that this slice is target-normalization evidence rather than stale rescue
- Reporting updates:
  - H2l synthesis: [`results/reports/h2l_target_normalization_overreach_synthesis/report.md`](../results/reports/h2l_target_normalization_overreach_synthesis/report.md)
  - H2l figure: [`results/reports/h2l_target_normalization_overreach_synthesis/figures/h2l_target_normalization_overreach_gate.svg`](../results/reports/h2l_target_normalization_overreach_synthesis/figures/h2l_target_normalization_overreach_gate.svg)
  - new claim: `C50_h2l_overreach_holdout_supports_target_normalization_scope`
  - publication evidence ledger now has `50` claims, `282` sources, and `0` missing sources
  - publication readiness audit now has `188` checks, `181` blocking checks, `0` blocking failures, and status `paper_draft_ready`
- Research decision:
  - treat H2l as positive scope evidence, not closure
  - build H2m with less direct target phrasing, ambiguous local context, and repeated variants before declaring over-normalization solved
  - keep H2a stale-selection globally for stale-origin packets; H2k/H2l only show it is not the active mechanism on these target-normalization slices
- Verification:
  - `uv run pytest tests/test_h2l_target_normalization_overreach_synthesis.py -q`
  - `uv run python scripts/build_h2l_target_normalization_overreach_synthesis.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-12 - H2l Target-Normalization Overreach Packet Scaffold

- Added H2l as the next post-H2k replay-shaped holdout:
  - suite: `h2l_target_normalization_overreach_v18`
  - builder: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
  - dry-run packet: [`results/tool_probe_replay_packets/20260512T_h2l_target_normalization_overreach_dry_run_v1`](../results/tool_probe_replay_packets/20260512T_h2l_target_normalization_overreach_dry_run_v1)
- Packet shape:
  - `8` cases
  - `4` value-bearing target cases where the expected `target_query` is intentionally longer, such as `result badge Blocked`
  - `2` alias-is-target cases where the alias that H2k treated as a decoy is now the correct target
  - `2` H2k regression guards where the shorter component label remains the correct target
- Research purpose:
  - H2k showed target-query normalization repairs target/decoy overlap and does not depend on stale-selection rescue
  - H2l asks the next paper-relevant question: does the normalizer over-strip labels when the longer/value-bearing label is truly canonical?
  - the packet keeps strict exactness and executor-equivalence separable and preserves replay-shaped pressure rather than returning to packaged workflows
- Next execution:
  - run H2j on H2l first
  - then run H2e and H2j-without-stale-selection on the same packet
  - compare whether failures, if any, come from target-query normalization overreach or from the underlying route-arbitration profile
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py::test_visual_hard_slice_live_stress_packet_supports_h2l_target_normalization_overreach_suite -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --suite h2l_target_normalization_overreach_v18 --run-group-id 20260512T_h2l_target_normalization_overreach_dry_run_v1`

## 2026-05-12 - H2k Matched Stale-Gate Ablation Supports Target Normalization

- Added and executed the matched H2j-without-stale-selection-gate registry row:
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_target_query_normalization_no_stale_selection_gate`
  - live packet: [`results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_no_stale_gate_execute_v1`](../results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_no_stale_gate_execute_v1)
  - comparison against full H2j: [`results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_no_stale_gate_v1`](../results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_no_stale_gate_v1)
- Result:
  - H2j without stale-selection gate: `8 / 8` strict and `8 / 8` executor-equivalent
  - full H2j versus no-stale ablation: `0.0` strict exact-rate delta and `0.0` executor-equivalence-rate delta
  - no-stale ablation metadata: `5` target-query-normalization interventions and `0` stale-selection interventions
- Mechanism read:
  - H2e-on-H2k remains the no-target-normalizer control at `3 / 8` strict and `6 / 8` executor-equivalent
  - H2j-no-stale ties full H2j at `8 / 8`
  - therefore H2k supports target-query normalization as the active mechanism and rejects stale-selection rescue as the explanation for this slice
- Reporting updates:
  - H2k synthesis now includes `4` packet rows and `3` comparisons
  - H2k figure now includes H2e, H2h, H2j-no-stale, and full H2j
  - publication evidence ledger keeps `49` claims, now with `274` sources and `0` missing
  - claim `C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization` is upgraded to `supported_current_packets_helper_ablation_passed`
- Research decision:
  - keep the stale-selection gate globally, because H2a/H2j still need it on stale-origin packets
  - do not write another prompt-profile candidate yet
  - build H2l around target-normalization overreach, where the correct target is sometimes the displayed value, alias, or longer label that H2j might strip away
- Verification:
  - `uv run pytest tests/test_knowledge_work_h1.py::test_h2k_target_query_normalization_without_stale_gate_registry_row_preserves_ablation_flags -q`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260512T_h2k_target_decoy_overlap_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_target_query_normalization_no_stale_selection_gate --output-dir results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_no_stale_gate_execute_v1 --execute --json`
  - `uv run python scripts/compare_tool_probe_replay_live_packets.py results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_no_stale_gate_execute_v1 results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_execute_v1 --output-dir results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_no_stale_gate_v1`
  - `uv run pytest tests/test_h2k_target_decoy_overlap_synthesis.py tests/test_publication_evidence_ledger.py -q`

## 2026-05-12 - H2k Target/Decoy Overlap Gate Separates H2j

- Executed the H2k target/decoy-overlap holdout across the three relevant profiles:
  - H2j target-query normalization: [`results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_execute_v1`](../results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_execute_v1)
  - H2e route arbitration plus stale-selection gate: [`results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2e_execute_v1`](../results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2e_execute_v1)
  - H2h component-identity negative examples plus stale-selection gate: [`results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2h_execute_v1`](../results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2h_execute_v1)
- Result:
  - H2j: `8 / 8` strict and `8 / 8` executor-equivalent
  - H2h: `6 / 8` strict and `6 / 8` executor-equivalent
  - H2e: `3 / 8` strict and `6 / 8` executor-equivalent
  - H2j-vs-H2e delta: `+0.625` strict exact-rate and `+0.25` executor-equivalence-rate
  - H2j-vs-H2h delta: `+0.25` strict exact-rate and `+0.25` executor-equivalence-rate
- Mechanism read:
  - H2j records `5` target-query-normalization interventions on H2k
  - H2j records `0` stale-selection interventions on H2k
  - the remaining H2e/H2h failures are target-query drift, not missing-tool or stale-selection-origin failures
  - this supports H2j as a structural controller-normalization result on a fresh post-H2j holdout
- Reporting updates:
  - H2k synthesis: [`results/reports/h2k_target_decoy_overlap_synthesis/report.md`](../results/reports/h2k_target_decoy_overlap_synthesis/report.md)
  - H2k figure: [`results/reports/h2k_target_decoy_overlap_synthesis/figures/h2k_target_decoy_overlap_gate.svg`](../results/reports/h2k_target_decoy_overlap_synthesis/figures/h2k_target_decoy_overlap_gate.svg)
  - publication evidence ledger now has `49` claims, `272` sources, and `0` missing sources
  - new claim: `C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization`
- Research decision:
  - do not write another prompt-profile candidate yet
  - treat H2e-on-H2k as the already-run target-normalizer-disabled ablation
  - add and run the matched H2j-without-stale-selection-gate row next
  - promote C49 only after helper ablation shows the H2k gain is actually target-query-normalization causal
- Verification:
  - `uv run pytest tests/test_h2k_target_decoy_overlap_synthesis.py -q`
  - `uv run python scripts/build_h2k_target_decoy_overlap_synthesis.py`
  - `uv run pytest tests/test_publication_evidence_ledger.py -q`
  - `uv run python scripts/build_publication_evidence_ledger.py`

## 2026-05-12 - H2k Target/Decoy Overlap Holdout Scaffold

- Added H2k as the next harder replay-shaped holdout for the H2j boundary:
  - suite: `h2k_target_decoy_overlap_v17`
  - builder: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
  - dry-run packet: [`results/tool_probe_replay_packets/20260512T_h2k_target_decoy_overlap_dry_run_v1`](../results/tool_probe_replay_packets/20260512T_h2k_target_decoy_overlap_dry_run_v1)
- Packet shape:
  - `8` cases
  - `3` negated same-component decoys
  - `2` before-reading decoys
  - `2` code-label overlap cases
  - `1` H2h transfer-regression guard
- Research purpose:
  - stress the exact risk left by H2j: labels that appear both as requested targets and as decoys
  - keep the executor oracle strict by giving each expected `extract_layout.target_query` a unique local visual region
  - create a stable packet for H2j/H2e/H2h comparison before any new global promotion claim
- Next execution:
  - run H2j on H2k first
  - then compare H2j against H2e and H2h
  - then run helper ablations only if H2j remains meaningfully above the H2e/H2h boundary
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --suite h2k_target_decoy_overlap_v17 --run-group-id 20260512T_h2k_target_decoy_overlap_dry_run_v1`

## 2026-05-12 - H2j Target-Query Normalization Repairs H2f and Preserves Transfer

- Added H2j as a controller-visible target-query normalization candidate layered on the H2e route-arbitration profile and the H2a stale-selection gate:
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization`
  - H2j transfer synthesis: [`results/reports/h2j_target_query_normalization_transfer_synthesis/report.md`](../results/reports/h2j_target_query_normalization_transfer_synthesis/report.md)
  - H2j transfer figure: [`results/reports/h2j_target_query_normalization_transfer_synthesis/figures/h2j_transfer_gate.svg`](../results/reports/h2j_target_query_normalization_transfer_synthesis/figures/h2j_transfer_gate.svg)
  - H2f live packet: [`results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2f_execute_v2`](../results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2f_execute_v2)
  - H2b live packet: [`results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2b_execute_v2`](../results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2b_execute_v2)
  - H1x live packet: [`results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h1x_execute_v1`](../results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h1x_execute_v1)
- H2j result:
  - H2f: `10 / 10` strict and executor-equivalent
  - H2b: `5 / 5` strict and executor-equivalent
  - H1x: `8 / 8` strict and executor-equivalent
  - H2f deltas: `+0.4` exact-rate versus H2e, `+0.1` versus H2h, and `+0.4` versus H2i
  - transfer deltas: ties H2e on H2b/H1x while beating H2h by `+0.4` on H2b and `+0.25` on H1x
- Mechanism read:
  - H2j is the first candidate in this line to repair H2f while preserving the prior H2b and H1x transfer gates
  - target-query normalization fired on `4` rows and stale-selection rescue fired on `4` rows across H2f/H2b/H1x
  - the repair is now recorded in per-case `runtime_metadata`, so controller dependence is attributable rather than inferred from top-line score
  - an initial H2b backtest exposed a transfer bug where `alert s92` could be lost to a negated `consent toggle` decoy; the label scorer was tightened to preserve action-verb target labels and penalize negated/before-reading decoys before the final H2b v2 run
- Research decision:
  - promote H2j to a harder post-H2j holdout, not to global default status yet
  - build H2k around labels that appear both as requested targets and as negated or before-reading decoys
  - run separate ablations for target-query normalization and stale-selection rescue after H2k, so the next claim measures controller dependence directly
- Reporting updates:
  - H2f synthesis now includes H2j and reports `10 / 10` H2j with `0` H2j non-exact rows
  - publication evidence ledger now has `48` claims, `264` sources, and `0` missing sources
  - new claim: `C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer`
  - publication readiness audit remains `paper_draft_ready` with `178` checks, `171` blocking checks, and `0` blocking failures
- Verification:
  - `uv run pytest tests/test_tool_directive_probe.py tests/test_knowledge_work_h1.py::test_h2j_target_query_normalization_registry_row_preserves_profile_and_controller_flags -q`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization --output-dir results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2f_execute_v2 --execute --json`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2b_residual_exactness_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization --output-dir results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2b_execute_v2 --execute --json`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_target_query_normalization --output-dir results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h1x_execute_v1 --execute --json`
  - `uv run pytest tests/test_h2f_route_arbitration_holdout_synthesis.py tests/test_h2j_target_query_normalization_transfer_synthesis.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py -q`
  - `uv run python scripts/build_h2f_route_arbitration_holdout_synthesis.py`
  - `uv run python scripts/build_h2j_target_query_normalization_transfer_synthesis.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-10 - H2i Conditional Arbitration Is Negative on H2f

- Added and executed H2i as a conditional component-identity arbitration candidate:
  - profile: `visual_role_catalog_conditional_component_identity_arbitration_v23`
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_component_identity_arbitration_visual_stale_selection_gate`
  - H2f live packet: [`results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1`](../results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1)
  - H2i-vs-H2h comparison: [`results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2h_on_h2f_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2h_on_h2f_v1)
  - H2i-vs-H2e comparison: [`results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2e_on_h2f_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2e_on_h2f_v1)
- H2i result:
  - H2f strict exactness: `6 / 10`, tying H2e and H2g
  - H2f executor-equivalence: `6 / 10`
  - delta versus H2h: `-0.3` exact-rate and executor-equivalence-rate
- Mechanism read:
  - H2i preserved the less aggressive route default, but did not preserve H2h's H2f repair
  - misses: `alert t47` -> `Escalated`, `result tile` -> `result tile for Blocked`, `resolution badge` -> `resolution badge for Deferred`, and `state marker` -> `lifecycle state marker`
  - because H2i failed the H2f gate, no H2b/H1x transfer budget was spent
- Research decision:
  - reject H2i as the transfer-safe conditionalization answer
  - do not write another softer conditional prompt paragraph
  - next candidate should be structurally different: route gate, query-normalization contract, or controller-visible argument canonicalization
- Reporting updates:
  - publication evidence ledger now has `47` claims, `253` sources, and `0` missing sources
  - new claim: `C47_h2i_conditional_component_arbitration_does_not_preserve_h2f_repair`
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py::test_visual_role_catalog_conditional_component_identity_arbitration_guards_h2h_regressions tests/test_knowledge_work_h1.py::test_h2i_conditional_component_identity_arbitration_registry_row_preserves_profile_and_controller_flag -q`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_component_identity_arbitration_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1 --execute --json`
  - `uv run python scripts/build_h2f_route_arbitration_holdout_synthesis.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-10 - H2h Repairs Fresh H2f but Fails Global Transfer

- Added H2h as an explicit negative-example component-identity contract:
  - profile: `visual_role_catalog_component_identity_negative_examples_v22`
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate`
  - H2f live packet: [`results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1`](../results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1)
  - H2b live packet: [`results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2b_execute_v1`](../results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2b_execute_v1)
  - H1x live packet: [`results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h1x_execute_v1`](../results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h1x_execute_v1)
  - tradeoff synthesis: [`results/reports/h2h_component_identity_tradeoff_synthesis/report.md`](../results/reports/h2h_component_identity_tradeoff_synthesis/report.md)
- H2h result:
  - H2f: `9 / 10` strict and executor-equivalent, a `+0.3` exact-rate lift over H2e/H2g
  - H2b: `3 / 5`, a `-0.4` exact-rate regression versus H2e/H2c
  - H1x: `6 / 8`, a `-0.25` exact-rate regression versus H2e/H2d
- Mechanism read:
  - explicit negative examples are causal on the fresh H2f displayed-value component-identity failures
  - the remaining H2f miss is `state marker` -> `lifecycle state marker`
  - the transfer regressions are component-class and code-label leakage: `result pill` -> `result tile`, `badge c08` -> `badge m31 c08`, `result chip` -> `result tile`, and `error banner` -> `error notice`
- Research decision:
  - reject global H2h promotion
  - keep H2e as the safest route-arbitration default
  - build H2i as conditional arbitration that activates H2h-style negative examples only for explicit displayed-value component-identity prompts
- Reporting updates:
  - publication evidence ledger now has `46` claims, `249` sources, and `0` missing sources
  - new claim: `C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer`
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py::test_visual_role_catalog_component_identity_negative_examples_targets_h2f_residuals tests/test_knowledge_work_h1.py::test_h2h_component_identity_negative_examples_registry_row_preserves_profile_and_controller_flag -q`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1 --execute --json`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2b_residual_exactness_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2b_execute_v1 --execute --json`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_negative_examples_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h1x_execute_v1 --execute --json`
  - `uv run python scripts/build_h2f_route_arbitration_holdout_synthesis.py`
  - `uv run python scripts/build_h2h_component_identity_tradeoff_synthesis.py`
  - `uv run pytest tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py tests/test_h2f_route_arbitration_holdout_synthesis.py tests/test_h2h_component_identity_tradeoff_synthesis.py -q`

## 2026-05-10 - H2g Component-Identity Contract Is Partial, Not Strict Repair

- Added H2g as the first component-identity query-contract candidate:
  - profile: `visual_role_catalog_component_identity_query_contract_v21`
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_query_contract_visual_stale_selection_gate`
  - H2f live packet: [`results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1`](../results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1)
  - H2g-vs-H2e comparison: [`results/tool_probe_replay_live_comparisons/20260510T_h2g_component_identity_query_contract_vs_h2e_on_h2f_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h2g_component_identity_query_contract_vs_h2e_on_h2f_v1)
  - updated synthesis: [`results/reports/h2f_route_arbitration_holdout_synthesis/report.md`](../results/reports/h2f_route_arbitration_holdout_synthesis/report.md)
- H2g result on H2f:
  - strict exactness: `6 / 10`, tied with H2e and H2c
  - executor-equivalence: `7 / 10`, improving over H2e by `+0.1`
  - no-directive remains `1 / 10`, so the controller/profile stack is still causal versus the floor
- Mechanism read:
  - H2g did not repair strict component-identity query fidelity
  - it made `resolution badge Deferred` executor-valid, but still failed exactness
  - remaining exact misses preserve the H2f pattern: `result tile` -> `Blocked`, `state marker` -> `lifecycle state marker`, and `mode switch` -> `mode toggle`
- Research decision:
  - reject H2g as a strict promotion candidate
  - build H2h with explicit negative examples for value substitution and alias expansion
  - defer H2b/H1x backtests until strict H2f exactness improves
- Reporting updates:
  - publication evidence ledger now has `45` claims, `243` sources, and `0` missing sources
  - new claim: `C45_h2g_component_identity_contract_is_partial_executor_gain`
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py::test_visual_role_catalog_component_identity_query_contract_preserves_h2f_labels tests/test_knowledge_work_h1.py::test_h2g_component_identity_query_contract_registry_row_preserves_profile_and_controller_flag -q`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_identity_query_contract_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1 --execute --json`
  - `uv run python scripts/build_h2f_route_arbitration_holdout_synthesis.py`
  - `uv run pytest tests/test_h2f_route_arbitration_holdout_synthesis.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py -q`

## 2026-05-10 - H2f Fresh Holdout Breaks H2e Global Promotion

- Built the fresh H2f route-arbitration holdout after H2e saturated H2b and H1x:
  - suite: `h2f_route_arbitration_v16`
  - packet: [`results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1)
  - synthesis: [`results/reports/h2f_route_arbitration_holdout_synthesis/report.md`](../results/reports/h2f_route_arbitration_holdout_synthesis/report.md)
  - figure: [`results/reports/h2f_route_arbitration_holdout_synthesis/figures/h2f_holdout_profile_bars.svg`](../results/reports/h2f_route_arbitration_holdout_synthesis/figures/h2f_holdout_profile_bars.svg)
- H2f row results:
  - no-directive: `1 / 10` strict and executor-equivalent
  - H2a component-label guard plus stale-selection gate: `4 / 10`
  - component-residual guard v12: `5 / 10` strict, `6 / 10` executor-equivalent
  - H2d class-preserving route: `5 / 10`
  - H2c scoped residual gate: `6 / 10`
  - H2e route arbitration: `6 / 10`
- Direct comparison read:
  - H2e ties H2c on H2f: `0.0` strict and executor-equivalence delta
  - H2e beats H2d by `+0.1`, H2a by `+0.2`, and no-directive by `+0.5`
  - H2e ties component-residual v12 on executor-equivalence while beating it by `+0.1` strict exactness
- Mechanism read:
  - H2f breaks the previous top-line saturation cleanly
  - controller/prompt helpers remain causal against the no-directive floor
  - route arbitration does not generalize beyond H2c on this fresh holdout
  - H2e's four misses all call the right tool but send the wrong `target_query`
  - the miss pattern is component-identity binding: `result tile` -> `Blocked`, `resolution badge` -> `Deferred`, `state marker` -> `lifecycle state marker`, and `mode switch` -> `mode toggle`
- Research decision:
  - reject global H2e promotion
  - build H2g around a component-identity query contract, not more broad route-arbitration wording
  - rerun H2g on H2f first, then backtest against H2b/H1x for regressions
- Reporting updates:
  - publication evidence ledger now has `44` claims, `240` sources, and `0` missing sources
  - new claim: `C44_h2f_holdout_breaks_h2e_global_promotion`
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1 --execute --json`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2c_execute_v1 --execute --json`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2f_route_arbitration_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive --output-dir results/tool_probe_replay_live/20260510T_h2f_route_arbitration_no_directive_execute_v1 --execute --json`
  - `uv run python scripts/build_h2f_route_arbitration_holdout_synthesis.py`
  - `uv run pytest tests/test_h2f_route_arbitration_holdout_synthesis.py tests/test_publication_evidence_ledger.py -q`
  - `uv run python scripts/build_publication_evidence_ledger.py`

## 2026-05-10 - H2d/H2e Route Arbitration Resolves the First Transfer Tradeoff

- Built and executed the class-preserving H2d profile after H2c's held-out H1x transfer miss:
  - profile: `visual_role_catalog_class_preserving_residual_route_v19`
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_class_preserving_residual_route_visual_stale_selection_gate`
  - H2b live packet: [`results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h2b_execute_v1`](../results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h2b_execute_v1)
  - H1x live packet: [`results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h1x_execute_v1`](../results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h1x_execute_v1)
  - synthesis: [`results/reports/h2d_transfer_tradeoff_synthesis/report.md`](../results/reports/h2d_transfer_tradeoff_synthesis/report.md)
- H2d result:
  - H2b: `4 / 5` strict, `5 / 5` executor-equivalent
  - H1x: `8 / 8` strict and executor-equivalent
  - interpretation: H2d fixed H2c's `result chip` -> `result pill` class-swap, but over-specified one H2b code-label row as `escalated badge c08` instead of `badge c08`
- Built and executed H2e as route arbitration over the H2c/H2d split:
  - profile: `visual_role_catalog_route_arbitration_residual_exactness_v20`
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate`
  - H2b live packet: [`results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h2b_execute_v1`](../results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h2b_execute_v1)
  - H1x live packet: [`results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h1x_execute_v1`](../results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h1x_execute_v1)
  - synthesis: [`results/reports/h2e_route_arbitration_synthesis/report.md`](../results/reports/h2e_route_arbitration_synthesis/report.md)
  - figure: [`results/reports/h2e_route_arbitration_synthesis/figures/h2e_route_arbitration_gate.svg`](../results/reports/h2e_route_arbitration_synthesis/figures/h2e_route_arbitration_gate.svg)
- H2e result:
  - H2b: `5 / 5` strict and executor-equivalent
  - H1x: `8 / 8` strict and executor-equivalent
  - zero non-exact rows across the two packets
  - delta versus H2c on H1x: `+0.125` strict and executor-equivalence
  - delta versus H2d on H2b: `+0.2` strict, with executor-equivalence tied
- Scientific read:
  - controller stale-selection mediation and prompt-level residual exactness are separable mechanisms
  - local saturation can hide transfer class-swap errors
  - class-preserving transfer can still lose compact code-label exactness
  - route arbitration is the first profile to preserve the observed maxima on both gates, but the result is not a population estimate
- Reporting updates:
  - publication evidence ledger now has `43` claims, `233` sources, and `0` missing sources
  - publication readiness audit remains `paper_draft_ready` with `178` checks, `171` blocking checks, and `0` failures
  - new claims: `C42_h2d_class_preserving_route_repairs_h2c_transfer_but_costs_h2b_exactness`, `C43_h2e_route_arbitration_reconciles_h2c_h2d_tradeoff`
- Next move:
  - build H2f as a fresh holdout, not a rerun of H2b/H1x
  - include unseen code suffixes, component classes, stale-id decoys, neighboring comments/logs/summaries, negated controls, and displayed-value distractors
  - run H2e against H2a/H2c/H2d/v12/no-directive controls and report strict exactness plus executor-equivalence
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py -q`
  - `uv run pytest tests/test_h2d_transfer_tradeoff_synthesis.py tests/test_h2e_route_arbitration_synthesis.py -q`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2b_residual_exactness_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h2b_execute_v1 --execute --json`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h1x_execute_v1 --execute --json`
  - `uv run python scripts/build_h2d_transfer_tradeoff_synthesis.py`
  - `uv run python scripts/build_h2e_route_arbitration_synthesis.py`
  - `uv run pytest tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py -q`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-10 - H2a Controller Stale-Selection Gate Becomes the Current Local Winner

- Executed the controller-side stale-selection gate on the same H1y mixed packet used for the prompt/catalog routed-residual tests.
- Core artifacts:
  - H2a live packet: [`results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1y_execute_v1`](../results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1y_execute_v1)
  - synthesis: [`results/reports/h1y_routed_residual_synthesis/report.md`](../results/reports/h1y_routed_residual_synthesis/report.md)
  - main MLX report table: [`results/reports/mlx_tool_contract_harnessing/tables/h1y_routed_residual_packet_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/h1y_routed_residual_packet_summary.csv)
  - main MLX report figure: [`results/reports/mlx_tool_contract_harnessing/figures/h1y_routed_residual_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/h1y_routed_residual_gate.svg)
- Live result on H1y:
  - no-directive: `0 / 10` exact and executor-equivalent
  - v11 component-label guard: `5 / 10`
  - v12 component-residual guard: `7 / 10`
  - v16 routed-residual guard: `5 / 10`
  - v17 selection-origin guard: `5 / 10`
  - H2a v11 + controller stale-selection gate: `8 / 10`
- Mechanism read:
  - catalog prose alone did not solve stale user-mentioned `selection_id` hazards
  - H2a fixed all three stale-field route rows without using expected calls or benchmark answers
  - H2a preserved both surface-value holdouts, avoiding the v16 regression
  - the two remaining H2a misses are `h1y_lifecycle_state_tag_audit_value_decoy` and `h1y_alert_s92_negated_toggle_decoy`, both argument-alias/code-label style failures rather than stale-selection failures
- Research decision:
  - promote H2a to transfer retest, not to global default yet
  - stop adding broad catalog prose for this failure family until transfer data says otherwise
  - next slice should test the stale-selection gate across H1n/H1o/H1p/H1x and then isolate the remaining argument-alias/code-label residue
- Reporting updates:
  - H2a is now publication claim `C38_h2a_controller_stale_selection_gate_is_causal`
  - MLX tool-contract report now has `97` tables and `42` figures
  - publication evidence ledger now has `38` claims and `197` evidence sources with `0` missing sources
  - publication readiness audit remains `paper_draft_ready` with `0` blocking failures
- Verification:
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h1y_routed_residual_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1y_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate --registry configs/model_registry.yaml --execute --json`
  - `uv run python scripts/build_h1y_routed_residual_synthesis.py`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-10 - H2b Residual Exactness Gate and H2c Pivot

- Added and reported the H2b residual exactness synthesis:
  - synthesis script: [`scripts/build_h2b_residual_exactness_synthesis.py`](../scripts/build_h2b_residual_exactness_synthesis.py)
  - synthesis report: [`results/reports/h2b_residual_exactness_synthesis/report.md`](../results/reports/h2b_residual_exactness_synthesis/report.md)
  - synthesis payload: [`results/reports/h2b_residual_exactness_synthesis/synthesis.json`](../results/reports/h2b_residual_exactness_synthesis/synthesis.json)
  - packet summary table: [`results/reports/h2b_residual_exactness_synthesis/tables/h2b_residual_exactness_packet_summary.csv`](../results/reports/h2b_residual_exactness_synthesis/tables/h2b_residual_exactness_packet_summary.csv)
  - case matrix: [`results/reports/h2b_residual_exactness_synthesis/tables/h2b_residual_exactness_case_matrix.csv`](../results/reports/h2b_residual_exactness_synthesis/tables/h2b_residual_exactness_case_matrix.csv)
- H2b composes the five exactness residuals left by H2a transfer:
  - `component_value_result_pill_log_decoy`
  - `h1o_code_alert_s92_negated_toggle_decoy`
  - `h1o_code_badge_c08_note_decoy`
  - `h1p_compact_state_tag_log_value_decoy`
  - `h1p_surface_mode_toggle_note_value_decoy`
- Result:
  - no-directive: `1 / 5` strict, `2 / 5` executor-equivalent
  - v11 component-label guard: `0 / 5` strict, `3 / 5` executor-equivalent
  - v12 component-residual guard: `4 / 5` strict, `4 / 5` executor-equivalent
  - v15 code-label exact guard: `3 / 5` strict, `3 / 5` executor-equivalent
  - H2a stale-selection gate: `0 / 5` strict, `3 / 5` executor-equivalent
  - v9 component-value guard: `3 / 5` strict, `4 / 5` executor-equivalent
- Interpretation:
  - H2a is causal and transferable for stale-selection mediation, but it is not an alias/code-label exactness solution.
  - V12 is the strict H2b winner, but H1s still blocks global v12 promotion because v12 reduced executor-equivalence on transfer.
  - V9 tying v12 on executor-equivalence keeps the distinction between strict canonical-label fidelity and executor-visible usefulness alive.
  - The next scientific move is H2c: a scoped residual route/factor that chooses v12-like residual exactness only for alias/code-label risk while preserving H2a for stale or missing `selection_id` repair.
- Reporting updates:
  - generated MLX tool-contract report now has `106` tables and `44` figures
  - publication evidence ledger now has `40` claims, `214` evidence sources, and `0` missing sources
  - new claim: `C40_h2b_residual_exactness_favors_scoped_v12_not_global_h2a`
  - publication readiness audit now has `170` checks, `164` blocking checks, `0` blocking failures, and status `paper_draft_ready`
- Verification:
  - `uv run pytest tests/test_h2b_residual_exactness_synthesis.py -q`
  - `uv run pytest tests/test_mlx_tool_contract_report.py -q`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-10 - H2c Scoped Residual Local Gate

- Added the H2c scoped residual profile and controller-gated registry row:
  - profile id: `visual_role_catalog_scoped_residual_exactness_v18`
  - system id: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate`
  - mechanism: preserve H2a's stale-selection controller gate while adding scoped exactness for code-label, tag/toggle/switch, stale-field, and easy-to-swap role-plus-component cases
- Ran H2c live on the H2b residual packet:
  - live packet: [`results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1`](../results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1)
  - synthesis: [`results/reports/h2c_scoped_residual_synthesis/report.md`](../results/reports/h2c_scoped_residual_synthesis/report.md)
  - comparison versus v12: [`20260510T_h2c_scoped_residual_gate_vs_component_residual_guard_on_h2b_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_component_residual_guard_on_h2b_v1)
  - comparison versus H2a: [`20260510T_h2c_scoped_residual_gate_vs_h2a_on_h2b_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_h2a_on_h2b_v1)
- Result:
  - H2c: `5 / 5` strict, `5 / 5` executor-equivalent
  - v12: `4 / 5` strict, `4 / 5` executor-equivalent
  - H2a: `0 / 5` strict, `3 / 5` executor-equivalent
  - v9: `3 / 5` strict, `4 / 5` executor-equivalent
  - no-directive: `1 / 5` strict, `2 / 5` executor-equivalent
- Interpretation:
  - H2c is the strongest local residual-exactness result so far.
  - The mechanism split is now clean: H2a handles stale selection-origin mediation; H2c handles alias/code-label/nonstandard-component exactness.
  - This is not enough for a default promotion because the packet is selected from H2a residuals and H1s already showed residual wording can hurt transfer.
  - The next publishable test is a minimal H2c transfer gate over H1n/H1o/H1p/H1x residual families.
- Reporting updates:
  - generated MLX tool-contract report now has `110` tables and `45` figures
  - publication evidence ledger now has `41` claims, `222` evidence sources, and `0` missing sources
  - new claim: `C41_h2c_scoped_residual_gate_saturates_h2b_but_needs_transfer`
  - publication readiness audit now has `178` checks, `171` blocking checks, `0` blocking failures, and status `paper_draft_ready`
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py -q`
  - `uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h2b_residual_exactness_dry_run_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_scoped_residual_exactness_visual_stale_selection_gate --output-dir results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1 --execute --json`
  - `uv run pytest tests/test_h2c_scoped_residual_synthesis.py -q`
  - `uv run python scripts/build_h2c_scoped_residual_synthesis.py`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-10 - H2a Stale-Selection Transfer Gate

- Completed the H2a transfer test that was queued after the local H1y result.
- H2a profile:
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard_visual_stale_selection_gate`
  - interpretation: keep v11's component-label prompt contract, add only a controller-side stale-selection mediation path
- Transfer live packets:
  - H1n component-value residual: [`20260510T_h2a_visual_stale_selection_gate_on_h1n_component_value_execute_v1`](../results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1n_component_value_execute_v1)
  - H1o control-factorial: [`20260510T_h2a_visual_stale_selection_gate_on_h1o_execute_v1`](../results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1o_execute_v1)
  - H1p component-value holdout: [`20260510T_h2a_visual_stale_selection_gate_on_h1p_execute_v1`](../results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1p_execute_v1)
  - H1x v11-breaker: [`20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1`](../results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1)
- Transfer synthesis:
  - report: [`results/reports/h2a_stale_selection_transfer_synthesis/report.md`](../results/reports/h2a_stale_selection_transfer_synthesis/report.md)
  - aggregate table: [`h2a_stale_selection_transfer_aggregate_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/h2a_stale_selection_transfer_aggregate_summary.csv)
  - residual table: [`h2a_stale_selection_transfer_residual_rows.csv`](../results/reports/mlx_tool_contract_harnessing/tables/h2a_stale_selection_transfer_residual_rows.csv)
  - figure: [`h2a_stale_selection_transfer_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/h2a_stale_selection_transfer_gate.svg)
- Result:
  - no-directive transfer aggregate: `12 / 40` strict, `14 / 40` executor-equivalent
  - v11 component-label guard: `33 / 40` strict, `36 / 40` executor-equivalent
  - v12 component-residual guard: `35 / 40` strict, `35 / 40` executor-equivalent
  - H2a stale-selection gate: `35 / 40` strict, `38 / 40` executor-equivalent
- Interpretation:
  - H2a is not only a local H1y fix. It transfers across the older held-out replay-shaped visual packets.
  - The useful promotion claim is scoped: controller-side stale-selection mediation is causal when the model emits a missing/stale `selection_id` and live visual state can supply the current region. It is not a license for the controller to read expected calls or benchmark labels.
  - H2a ties v12 strict transfer while improving executor-equivalence by three rows, which makes it the cleanest current visual helper profile.
  - The residual problem has shifted: remaining failures are exact alias/code-label fidelity, especially `result pill`, `alert s92`, `badge c08`, `state tag`, and `mode toggle`.
- Reporting updates:
  - generated MLX report now has `102` tables and `43` figures
  - publication evidence ledger now has `39` claims, `205` evidence sources, and `0` missing sources
  - publication readiness audit has `157` checks, `152` blocking checks, `0` blocking failures, and status `paper_draft_ready`
  - new publication claim: `C39_h2a_stale_selection_gate_transfers_with_better_executor_profile`
- Verification:
  - `uv run pytest tests/test_h2a_stale_selection_transfer_synthesis.py tests/test_h1y_routed_residual_synthesis.py -q`
  - `uv run pytest tests/test_mlx_tool_contract_report.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py -q`
  - `uv run python scripts/build_h2a_stale_selection_transfer_synthesis.py`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
  - `uv run pytest tests/test_mlx_tool_contract_report.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py tests/test_h1y_routed_residual_synthesis.py -q`

## 2026-05-10 - H1y/H1z Routed Residual Prompt-Only Negative Result

- Built and executed the H1y mixed routed-residual packet to test whether prompt/catalog wording could keep v11's transfer stability while capturing v12's H1x stale-field gain.
- Packet: [`results/tool_probe_replay_packets/20260510T_h1y_routed_residual_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_h1y_routed_residual_oracle_dry_run_v1)
- Results:
  - no-directive: `0 / 10`
  - v11 component-label guard: `5 / 10`
  - v12 component-residual guard: `7 / 10`
  - v16 routed-residual guard: `5 / 10`
  - v17 selection-origin guard: `5 / 10`
- Interpretation:
  - v16's routed residual prose regressed surface-value holdouts while failing to beat v11
  - v17 restored surface-value holdouts but still used stale user-written selection ids on all three stale-field rows
  - this is negative evidence against solving stale selection-origin errors through more catalog prose alone
- Decision:
  - move the stale-selection hypothesis into the runtime/controller layer
  - keep v11 as the prompt default during the controller-gate test

## 2026-05-10 - H1x V11-Breaker Live Gate

- Executed the H1x packet after the scaffolded v11-breaker design:
  - packet: [`results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1)
  - synthesis: [`results/reports/h1x_v11_breaker_synthesis/report.md`](../results/reports/h1x_v11_breaker_synthesis/report.md)
  - main MLX report table: [`results/reports/mlx_tool_contract_harnessing/tables/h1x_v11_breaker_packet_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/h1x_v11_breaker_packet_summary.csv)
  - main MLX report figure: [`results/reports/mlx_tool_contract_harnessing/figures/h1x_v11_breaker_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/h1x_v11_breaker_gate.svg)
- Live result:
  - no-directive: `2 / 8` exact and executor-equivalent
  - v11 component-label guard: `7 / 8` exact and executor-equivalent
  - v12 component-residual guard: `8 / 8` exact and executor-equivalent
  - v15 code-label exact guard: `6 / 8` exact and `7 / 8` executor-equivalent
- Mechanism read:
  - H1x is the first focused post-H1w replay packet that breaks v11 saturation.
  - The v11 miss is concentrated in oblique stale-field routing: `h1x_responsible_party_field_old_owner_memo_decoy` becomes a wrong-tool call.
  - V12 repairs that stale-field miss and saturates the local packet, so residual wording is still a real local intervention.
  - V15 over-narrows again: it preserves one executor-equivalent surface-value paraphrase, but strict exactness falls below v11 and v12.
- Research decision:
  - do not globally promote v12 from H1x alone, because H1s already showed v12's broader transfer cost
  - keep v11 as the transfer-stable default
  - treat v12 as the routed residual-helper candidate for a future conditional or classifier-gated harness
  - next slice should test whether the H1x v12 win transfers to a mixed H1y packet without reintroducing H1s negative transfer
- Reporting updates:
  - H1x is now publication claim `C37_h1x_breaks_v11_saturation_but_supports_routing`
  - MLX tool-contract report now has `92` tables and `41` figures
  - publication evidence ledger now has `37` claims and `191` evidence sources with `0` missing sources
  - publication readiness audit remains `paper_draft_ready` with `0` blocking failures
- Verification:
  - `uv run python scripts/build_h1x_v11_breaker_synthesis.py`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
  - `uv run pytest tests/test_mlx_tool_contract_report.py tests/test_h1x_v11_breaker_synthesis.py -q`

## 2026-05-10 - H1x V11-Breaker Packet Scaffold

- Added `h1x_v11_breaker_v14` to the visual hard-slice packet builder.
- Generated dry-run packet: [`results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1)
- Packet shape:
  - `2` oblique stale-field cases
  - `2` oblique surface-value cases
  - `2` oblique nonstandard-class cases
  - `2` oblique activation/no-call cases
- Purpose:
  - follow H1w's v11 saturation with a packet that stresses v11 more directly
  - combine old selections, repeated values, and user-facing paraphrases with canonical layout labels in one case
- Status:
  - completed by the H1x live gate above
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`

## 2026-05-10 - H1w Residual-Overlap Packet Scaffold

- Added `h1w_residual_overlap_v13` to the visual hard-slice packet builder.
- Generated dry-run packet: [`results/tool_probe_replay_packets/20260510T_h1w_residual_overlap_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_h1w_residual_overlap_oracle_dry_run_v1)
- Packet shape:
  - `2` stale field-routing cases
  - `2` nonstandard component-class cases
  - `2` surface component-value cases
  - `2` activation/no-call cases
- Purpose:
  - turn the H1v rejection into a harder residual benchmark, not another broad prompt tweak
  - test whether v11's transfer stability, v12's residual repair, or v15's code-label exactness helps on the remaining overlap
- Live result:
  - no-directive: `0 / 8` exact and executor-equivalent
  - v11 component-label guard: `8 / 8` exact and executor-equivalent
  - v12 component-residual guard: `7 / 8` exact and executor-equivalent
  - v15 code-label exact guard: `6 / 8` exact and executor-equivalent
- Interpretation:
  - H1w is a useful controller-dependence probe: raw no-directive collapses completely
  - it is not a v11 breaker; v11 remains the transfer-stable default
  - v12 and v15 regress on surface component-value rows, especially status/result badge or pill cases with repeated values nearby
- Artifacts:
  - synthesis: [`results/reports/h1w_residual_overlap_synthesis/report.md`](../results/reports/h1w_residual_overlap_synthesis/report.md)
  - v11 replay: [`results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_label_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_label_guard_execute_v1)
  - v12 replay: [`results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_residual_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_residual_guard_execute_v1)
  - v15 replay: [`results/tool_probe_replay_live/20260510T_h1w_residual_overlap_code_label_exact_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1w_residual_overlap_code_label_exact_guard_execute_v1)
- Next execution:
  - design H1x as a v11 breaker, combining oblique labels, stale selections, and repeated values within single cases
- Verification:
  - `uv run pytest tests/test_h1w_residual_overlap_synthesis.py tests/test_visual_hard_slice_live_stress_packet.py tests/test_tool_probe_replay_live_comparison.py -q`

## 2026-05-10 - H1v Code-Label Exact Transfer Gate

- Transfer-tested `visual_role_catalog_code_label_exact_guard_v15` across H1n/H1o/H1p after its H1r local saturation.
- Live results:
  - H1n: `5 / 8` exact and executor-equivalent, matching v12 but below v11 (`6 / 8` exact, `7 / 8` executor-equivalent)
  - H1o: `11 / 12` exact and executor-equivalent, matching v12 and improving strict exactness over v11 while losing one executor-equivalent case
  - H1p: `9 / 12` exact and executor-equivalent, below v11 (`10 / 12`) and v12 (`11 / 12`)
- Aggregate transfer verdict:
  - v15: `25 / 32` exact and `25 / 32` executor-equivalent
  - v11: `26 / 32` exact and `29 / 32` executor-equivalent
  - v12: `27 / 32` exact and `27 / 32` executor-equivalent
- Interpretation:
  - v15 is a real local code-label repair, but not a global prompt-contract promotion
  - the remaining bottleneck is still component/value and stale-selection routing, not code-label exactness alone
- Artifacts:
  - synthesis: [`results/reports/h1v_code_label_exact_transfer_synthesis/report.md`](../results/reports/h1v_code_label_exact_transfer_synthesis/report.md)
  - H1n replay: [`results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1`](../results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1)
  - H1o replay: [`results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1o_control_factorial_execute_v1`](../results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1o_control_factorial_execute_v1)
  - H1p replay: [`results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1`](../results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1)
- Next execution:
  - keep v11 as the transfer-stable default
  - build H1w around the remaining shared residuals: owner-field stale selection, nonstandard `tag`/`toggle`, and surface/result badge value collapse
- Verification:
  - `uv run pytest tests/test_h1v_code_label_exact_transfer_synthesis.py tests/test_h1u_split_factor_synthesis.py tests/test_h1s_component_residual_transfer_synthesis.py tests/test_tool_probe_replay_live_comparison.py -q`

## 2026-05-10 - H1u Split-Factor Route Gate

- Split the failed H1t compact conditional route into two independent prompt factors:
  - `visual_role_catalog_nonstandard_component_class_guard_v14`
  - `visual_role_catalog_code_label_exact_guard_v15`
- Added registry systems:
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_nonstandard_component_class_guard`
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard`
- Motivation from H1t failures:
  - `state tag` and `mode toggle` collapsed into displayed values (`Closed`, `Manual`)
  - `alert s92` collapsed into a negated neighboring `consent toggle`
- Live result:
  - v14 reaches `5 / 6` exact and executor-equivalent on H1r, tying v11 while fixing the nonstandard class value-collapse cases
  - v15 reaches `6 / 6` exact and executor-equivalent on H1r, tying v12 and beating v11 by one strict/executor case
- Interpretation:
  - nonstandard component-class wording is real but incomplete; it does not repair the `alert s92` code-label miss
  - code-label exactness wording is the stronger local factor and may preserve the H1r repair with less broad component-residual wording than v12
- Artifacts:
  - synthesis: [`results/reports/h1u_split_factor_synthesis/report.md`](../results/reports/h1u_split_factor_synthesis/report.md)
  - v14 replay: [`results/tool_probe_replay_live/20260510T_h1u_nonstandard_component_class_guard_on_h1r_component_residual_execute_v1`](../results/tool_probe_replay_live/20260510T_h1u_nonstandard_component_class_guard_on_h1r_component_residual_execute_v1)
  - v15 replay: [`results/tool_probe_replay_live/20260510T_h1u_code_label_exact_guard_on_h1r_component_residual_execute_v1`](../results/tool_probe_replay_live/20260510T_h1u_code_label_exact_guard_on_h1r_component_residual_execute_v1)
- Next execution:
  - transfer-test v15 across H1n/H1o/H1p before any promotion
  - compare aggregate exactness and executor-equivalence against v11 and v12
- Verification:
  - `uv run pytest tests/test_h1u_split_factor_synthesis.py tests/test_h1t_conditional_residual_route_synthesis.py tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1u_split_factor_registry_rows_preserve_catalog_profiles tests/test_tool_probe_replay_live_comparison.py -q`

## 2026-05-10 - H1t Conditional Residual-Route Rejected at H1r Gate

- Added `visual_role_catalog_conditional_residual_route_v13` as the direct follow-up to H1s:
  - defaults to v11's narrow component-label guard
  - activates v12-style residual handling only for code suffixes, nonstandard component classes (`tag`, `toggle`, `switch`), or stale-field contexts
  - explicitly avoids residual handling for ordinary `pill`, `badge`, `chip`, or `tile` cases unless a route condition is present
- Added registry system:
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route`
- Research purpose:
  - test whether conditional prompt routing can keep H1r/H1p gains without the H1n/H1o executor-equivalence loss that blocked v12 global promotion
- Live result:
  - v13 H1r replay: [`20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1`](../results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1)
  - synthesis: [`h1t_conditional_residual_route_synthesis`](../results/reports/h1t_conditional_residual_route_synthesis/report.md)
  - v13 reaches `3 / 6` exact and executor-equivalent on H1r
  - v13 is below v11 (`5 / 6`) and v12 (`6 / 6`)
  - non-exact failures are `h1r_state_tag_log_value_decoy`, `h1r_mode_toggle_note_value_decoy`, and `h1r_alert_s92_toggle_negation_decoy`
- Decision:
  - reject v13 before H1n/H1o/H1p transfer
  - compact conditional wording did not preserve the local residual win, so the next attempt should split route factors more explicitly
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1t_conditional_residual_route_registry_row_preserves_catalog_profile -q`
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h1r_component_label_residual_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route --registry configs/model_registry.yaml --execute --json`
  - `uv run python scripts/build_h1t_conditional_residual_route_synthesis.py`
  - `uv run pytest tests/test_h1t_conditional_residual_route_synthesis.py tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1t_conditional_residual_route_registry_row_preserves_catalog_profile tests/test_tool_probe_replay_live_comparison.py -q`

## 2026-05-10 - H1s Transfer Gate Rejects v12 as a Global Default

- Transfer-tested `visual_role_catalog_component_residual_guard_v12` back across the active H1n/H1o/H1p surfaces before promotion:
  - H1n v12 replay: [`20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1`](../results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1)
  - H1o v12 replay: [`20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1`](../results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1)
  - H1p v12 replay: [`20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1`](../results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1)
  - synthesis: [`h1s_component_residual_transfer_synthesis`](../results/reports/h1s_component_residual_transfer_synthesis/report.md)
- Transfer results:
  - H1r local residual: v12 remains positive at `6 / 6` exact and executor-equivalent
  - H1n component-value: v12 falls to `5 / 8` exact and executor-equivalent, below v11 at `6 / 8` exact and `7 / 8` executor-equivalent
  - H1o control-factorial: v12 reaches `11 / 12` exact and executor-equivalent, improving strict exactness over v11 but losing v11's `12 / 12` executor-equivalence ceiling
  - H1p component-value: v12 reaches `11 / 12` exact and executor-equivalent, improving over v11's `10 / 12`
  - aggregate across H1n/H1o/H1p: v12 is `27 / 32` exact and `27 / 32` executor-equivalent; v11 is `26 / 32` exact and `29 / 32` executor-equivalent
- Interpretation:
  - v12 is a real targeted patch for the H1r/H1p residuals
  - v12 is not globally promoted because the H1n/H1o executor-equivalence regression is material
  - the next prompt-contract move should test conditional routing or factor isolation: v11 as default, v12 wording only when code-label or nonstandard-component evidence is present
- Verification:
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_visual_hard_slice_component_value_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard --registry configs/model_registry.yaml --execute --json`
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h1o_control_factorial_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard --registry configs/model_registry.yaml --execute --json`
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h1p_component_value_holdout_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard --registry configs/model_registry.yaml --execute --json`
  - `uv run python scripts/build_h1s_component_residual_transfer_synthesis.py`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
  - `uv run pytest tests/test_h1s_component_residual_transfer_synthesis.py tests/test_h1r_component_residual_synthesis.py tests/test_h1q_component_label_guard_transfer_synthesis.py tests/test_tool_probe_replay_live_comparison.py -q`
- Formal reporting:
  - H1s is now publication claim `C36`
  - publication evidence ledger now has `36` claims and `184` evidence sources with `0` missing sources
  - publication readiness audit now has `133` checks, `131` blocking checks, `0` blocking failures, and status `paper_draft_ready`
  - MLX tool-contract report now has `87` tables and `40` figures

## 2026-05-10 - H1r Residual Component-Label Packet Saturates Under v12

- Added `visual_role_catalog_component_residual_guard_v12`, a narrow follow-up to v11:
  - keeps v11's role-plus-component copying discipline
  - adds explicit residual coverage for `tag`, `toggle`, `switch`, field-style stale-selection cases, and code labels such as `alert s92` / `badge c08`
  - keeps the guard scoped to H1q miss families instead of reviving broad v9 component-value prose
- Added a six-case H1r oracle dry-run packet:
  - packet: [`20260510T_h1r_component_label_residual_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_h1r_component_label_residual_oracle_dry_run_v1)
  - families: `2` stale-selection component-label cases, `2` nonstandard component-class cases, and `2` code-label exactness cases
  - all expected calls are oracle `extract_layout` calls that reach the expected executor region
- Next execution move:
  - completed live replay on no-directive, v11, and v12
  - no-directive reaches `0 / 6` exact and `1 / 6` executor-equivalent
  - v11 reaches `5 / 6` exact and executor-equivalent; its remaining miss is `h1r_alert_s92_toggle_negation_decoy`
  - v12 reaches `6 / 6` exact and executor-equivalent
  - synthesis: [`h1r_component_residual_synthesis`](../results/reports/h1r_component_residual_synthesis/report.md)
  - next: transfer-test v12 back across H1n/H1o/H1p before promotion or main-report integration
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1r_component_residual_guard_registry_row_preserves_catalog_profile tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260510T_h1r_component_label_residual_oracle_dry_run_v1 --suite h1r_component_label_residual_v12 --replay-system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard`
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_h1r_component_label_residual_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_residual_guard_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard --registry configs/model_registry.yaml --execute --json`
  - `uv run python scripts/build_h1r_component_residual_synthesis.py`
  - `uv run pytest tests/test_h1r_component_residual_synthesis.py tests/test_tool_probe_replay_live_comparison.py tests/test_prompt_contracts.py tests/test_visual_hard_slice_live_stress_packet.py tests/test_knowledge_work_h1.py::test_h1r_component_residual_guard_registry_row_preserves_catalog_profile -q`

## 2026-05-10 - H1q Narrow Component-Label Guard Becomes the Best Transfer Candidate

- Added `visual_role_catalog_component_label_guard_v11`, a narrower profile than v9:
  - copies requested role-plus-component labels such as `state pill`, `status badge`, `priority chip`, `owner field`, `lane tile`, `queue badge`, or `stage chip`
  - strips wrapper words like `component` and `itself`
  - avoids replacing the requested component label with the displayed value inside it
- Executed v11 across the three active transfer surfaces:
  - H1n component-value replay: [`20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1`](../results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1)
  - H1o control-factorial replay: [`20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1`](../results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1)
  - H1p component-value replay: [`20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1`](../results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1)
  - synthesis: [`h1q_component_label_guard_transfer_synthesis`](../results/reports/h1q_component_label_guard_transfer_synthesis/report.md)
- Transfer results:
  - H1n component-value: v11 `6 / 8` exact and `7 / 8` executor-equivalent, versus v9 at `4 / 8` and `4 / 8`
  - H1o control-factorial: v11 `10 / 12` exact and `12 / 12` executor-equivalent, setting a new H1o executor-equivalence ceiling
  - H1p component-value: v11 `10 / 12` exact and `10 / 12` executor-equivalent, tying v9 strict exactness but trailing v9 executor-equivalence by one case
  - aggregate: v11 `26 / 32` exact and `29 / 32` executor-equivalent, versus v9 at `23 / 32` and `25 / 32`
- Interpretation:
  - v11 is the strongest transfer candidate so far.
  - H1q validates the hypothesis that v9's H1p win was not purely an artifact, but broad v9 prose was too blunt.
  - v11 should not become the global default yet; residual failures remain on `owner field` stale selection, `state tag`, `mode toggle`, and H1o code-label exact paraphrases.
- Next research move:
  - build H1r/v12 around the v11 residuals, especially nonstandard component classes (`tag`, `toggle`) and stale owner-field routing
- Reporting:
  - H1q is now integrated into the formal MLX tool-contract report as packet, aggregate, failure, and finding tables plus a transfer-gate SVG
  - publication evidence ledger now has `35` claims and `178` evidence sources with `0` missing sources
  - publication readiness audit now has `127` checks, `125` blocking checks, `0` blocking failures, and status `paper_draft_ready`
  - MLX tool-contract report now has `82` tables and `39` figures
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1q_component_label_guard_registry_row_preserves_catalog_profile -q`
  - `uv run pytest tests/test_h1q_component_label_guard_transfer_synthesis.py -q`
  - `uv run python scripts/build_h1q_component_label_guard_transfer_synthesis.py`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
  - `uv run pytest tests/test_mlx_tool_contract_report.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py tests/test_h1q_component_label_guard_transfer_synthesis.py tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1q_component_label_guard_registry_row_preserves_catalog_profile -q`

## 2026-05-10 - H1p Component-Only Holdout Identifies a Component-Value Activation Domain

- Built and executed H1p as the fresh component-only holdout motivated by H1o's component/value residue:
  - packet: [`20260510T_h1p_component_value_holdout_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_h1p_component_value_holdout_oracle_dry_run_v1)
  - no-directive baseline: [`20260510T_h1p_component_value_no_directive_execute_v1`](../results/tool_probe_replay_live/20260510T_h1p_component_value_no_directive_execute_v1)
  - argument-hints row: [`20260510T_h1p_component_value_argument_hints_execute_v1`](../results/tool_probe_replay_live/20260510T_h1p_component_value_argument_hints_execute_v1)
  - hybrid-label row: [`20260510T_h1p_component_value_hybrid_label_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1p_component_value_hybrid_label_guard_execute_v1)
  - no-call rescue row: [`20260510T_h1p_component_value_no_call_control_rescue_execute_v1`](../results/tool_probe_replay_live/20260510T_h1p_component_value_no_call_control_rescue_execute_v1)
  - component-value row: [`20260510T_h1p_component_value_component_value_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1p_component_value_component_value_guard_execute_v1)
  - diagnostic: [`visual_h1p_component_value_diagnostic`](../results/reports/visual_h1p_component_value_diagnostic)
  - report table: [`visual_hard_slice_h1p_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1p_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_h1p_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1p_live_replay_gate.svg)
- Live replay matrix:
  - no-directive MLX: `0 / 12` exact and executor-equivalent
  - argument hints v2: `6 / 12` exact and executor-equivalent
  - no-call control rescue v10: `6 / 12` exact and executor-equivalent
  - hybrid label guard v8: `9 / 12` exact and `10 / 12` executor-equivalent
  - component-value guard v9: `10 / 12` exact and `11 / 12` executor-equivalent
- Interpretation:
  - H1p successfully breaks the saturated top-line surface: when the packet is pure component/value ambiguity, no-directive has no exact or executor-equivalent successes.
  - Component-value guard v9 is not globally bad; it has a real activation domain on component-only surfaces.
  - v9 is still not globally promotable because H1n showed broad component-value prose can regress already-passable selector cases, while H1o only tied argument hints on mixed mechanisms.
  - The next research move is H1q: split component-value guidance into narrower component-only wording and transfer-test it across H1p, H1o, and the H1n component-value cases.
- Reporting:
  - publication evidence ledger now has `34` claims and `173` evidence sources with `0` missing sources
  - publication readiness audit now has `121` checks, `119` blocking checks, `0` blocking failures, and status `paper_draft_ready`
  - MLX tool-contract report now has `78` tables and `38` figures
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py tests/test_visual_live_stress_diagnostic.py -q`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-10 - H1o Control-Factorial Slice Identifies Component/Value as the Residue

- Built and executed the H1o control-factorial replay slice to separate the mechanisms that H1n had started to entangle:
  - packet: [`20260510T_h1o_control_factorial_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_h1o_control_factorial_oracle_dry_run_v1)
  - no-directive baseline: [`20260510T_h1o_control_factorial_no_directive_execute_v1`](../results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_directive_execute_v1)
  - argument-hints row: [`20260510T_h1o_control_factorial_argument_hints_execute_v1`](../results/tool_probe_replay_live/20260510T_h1o_control_factorial_argument_hints_execute_v1)
  - hybrid-label row: [`20260510T_h1o_control_factorial_hybrid_label_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1o_control_factorial_hybrid_label_guard_execute_v1)
  - no-call rescue row: [`20260510T_h1o_control_factorial_no_call_control_rescue_execute_v1`](../results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_call_control_rescue_execute_v1)
  - oblique-code row: [`20260510T_h1o_control_factorial_oblique_code_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1o_control_factorial_oblique_code_guard_execute_v1)
  - component-value row: [`20260510T_h1o_control_factorial_component_value_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1o_control_factorial_component_value_guard_execute_v1)
  - diagnostic: [`visual_h1o_control_factorial_diagnostic`](../results/reports/visual_h1o_control_factorial_diagnostic)
  - synthesis: [`h1o_control_factorial_synthesis`](../results/reports/h1o_control_factorial_synthesis/report.md)
  - report table: [`visual_hard_slice_h1o_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1o_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_h1o_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1o_live_replay_gate.svg)
- Live replay matrix:
  - no-directive MLX: `5 / 12` exact and `6 / 12` executor-equivalent
  - argument hints v2: `9 / 12` exact and `10 / 12` executor-equivalent
  - component-value guard v9: `9 / 12` exact and `10 / 12` executor-equivalent
  - hybrid label guard v8: `8 / 12` exact and `10 / 12` executor-equivalent
  - oblique code guard v7: `8 / 12` exact and `9 / 12` executor-equivalent
  - no-call control rescue v10: `7 / 12` exact and `8 / 12` executor-equivalent
- Mechanism-family finding:
  - activation/no-call is not the remaining global bottleneck: no-directive is already `4 / 4` exact and executor-equivalent on that family
  - v10 no-call rescue is scoped and can regress activation cases; it loses `h1o_activation_error_banner_previous_region_decoy`
  - code/negation preservation is repairable: best rows reach `3 / 4` exact and `4 / 4` executor-equivalent
  - component/value remains the hard residue: best rows reach only `2 / 4` exact and executor-equivalent
  - argument hints remains the conservative default; component-value guard ties it on H1o but has not earned promotion because H1n already showed broad component-value wording can regress passable selector cases
- Reporting:
  - publication evidence ledger now has `33` claims and `165` evidence sources with `0` missing sources
  - publication readiness audit now has `115` checks, `113` blocking checks, `0` blocking failures, and status `paper_draft_ready`
  - MLX tool-contract report now has `76` tables and `37` figures
- Completed follow-up:
  - H1p now expands the component/value residue with more component/value surface diversity
  - activation/no-call wording stayed out of the default H1p packet
  - component-value guard v9 beat argument hints on H1p, but only as a local component-domain signal pending transfer tests
- Verification:
  - `uv run pytest tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py tests/test_mlx_tool_contract_report.py tests/test_h1o_control_factorial_synthesis.py tests/test_visual_live_stress_diagnostic.py -q`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
  - `uv run python scripts/build_mlx_tool_contract_report.py`

## 2026-05-10 - v10 No-Call Control Rescue Becomes the Component-Value Upper Bound

- Implemented `visual_role_catalog_no_call_control_rescue_v10` as a lighter follow-up to the failed v9 component-value guard:
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue`
  - live packet: [`20260510T_h1n_component_value_no_call_control_rescue_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_component_value_no_call_control_rescue_execute_v1)
  - no-directive comparison: [`20260510T_h1n_component_value_no_call_control_rescue_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h1n_component_value_no_call_control_rescue_vs_no_directive_v1)
  - diagnostic: [`visual_component_value_diagnostic`](../results/reports/visual_component_value_diagnostic)
- Live replay result:
  - v10 no-call control rescue: `7 / 8` exact and `8 / 8` executor-equivalent
  - argument hints v2 and hybrid label guard v8: `6 / 8` exact and `7 / 8` executor-equivalent
  - no-directive MLX: `5 / 8` exact and `6 / 8` executor-equivalent
  - v9 component-value guard: `4 / 8` exact and executor-equivalent
- Interpretation:
  - The useful mechanism is narrower than "component-role/value disambiguation." A generic current-image visual activation guard fixed the two no-call cases while preserving the already-passable pill/badge cases.
  - The remaining miss is not a failed execution; it is an exactness miss where the priority-chip selector is executor-equivalent.
  - Transfer synthesis: [`h1n_no_call_rescue_transfer_synthesis`](../results/reports/h1n_no_call_rescue_transfer_synthesis/report.md) shows v10 at `22 / 30` exact and `25 / 30` executor-equivalent across component-value, residual, post-repair, and oblique packets. That is a large gain over no-directive (`11 / 30`, `12 / 30`) but still behind per-packet incumbents (`25 / 30`, `26 / 30`).
  - This motivated H1o as a factorial slice instead of treating v10 as a durable replacement profile.
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1n_no_call_control_rescue_registry_row_preserves_catalog_profile -q`
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_visual_hard_slice_component_value_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1n_component_value_no_call_control_rescue_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_no_call_control_rescue --registry configs/model_registry.yaml --execute --json`
  - `uv run pytest tests/test_visual_live_stress_diagnostic.py -q`

## 2026-05-10 - Component-Value Holdout Rejects the Broad v9 Guard

- Built the focused component-role/value micro-slice that was motivated by the residual `state pill` miss:
  - packet: [`20260510T_visual_hard_slice_component_value_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_visual_hard_slice_component_value_oracle_dry_run_v1)
  - diagnostic: [`visual_component_value_diagnostic`](../results/reports/visual_component_value_diagnostic)
  - report table: [`visual_hard_slice_component_value_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_component_value_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_component_value_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_component_value_live_replay_gate.svg)
- Live replay matrix:
  - contracted/default MLX: `1 / 8` exact and executor-equivalent
  - no-directive MLX: `5 / 8` exact and `6 / 8` executor-equivalent
  - argument hints v2: `6 / 8` exact and `7 / 8` executor-equivalent
  - hybrid label guard v8: `6 / 8` exact and `7 / 8` executor-equivalent
  - oblique code guard v7: `5 / 8` exact and executor-equivalent
  - oblique code hints v6: `2 / 8` exact and executor-equivalent
  - schema-field hints v4: `3 / 8` exact and `4 / 8` executor-equivalent
  - component-value guard v9: `4 / 8` exact and executor-equivalent
- Interpretation:
  - v9 is negative evidence, not a promotion candidate. It fixes the `status badge` no-call case but regresses `state pill`, `priority chip`, and `result pill` into argument mismatches.
  - This result narrowed the research question: avoid broad component-role/value prose, and test a lighter no-call rescue that does not disturb argument-hints selector behavior.
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1n_component_value_guard_registry_row_preserves_catalog_profile -q`
  - `uv run pytest tests/test_visual_live_stress_diagnostic.py -q`
  - `uv run pytest tests/test_mlx_tool_contract_report.py -q`
- Reporting:
  - MLX tool-contract report now has `74` tables and `36` figures
  - publication evidence ledger now has `32` claims, `159` evidence sources, and `0` missing sources
  - publication readiness audit now has `108` checks, `106` blocking checks, `0` blocking failures, and status `paper_draft_ready`

## 2026-05-10 - H1n Residual Hybrid Label Guard Becomes the Strict Upper Bound

- Built and executed the residual holdout that followed the post-repair `chip l90` / `status pill` misses:
  - packet: [`20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1)
  - hybrid live packet: [`20260510T_h1n_residual_hybrid_label_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_residual_hybrid_label_guard_execute_v1)
  - diagnostic: [`visual_alias_transfer_residual_diagnostic`](../results/reports/visual_alias_transfer_residual_diagnostic)
  - report table: [`visual_hard_slice_residual_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_residual_live_replay_summary.csv)
- Result:
  - contracted/default MLX: `2 / 8`
  - no-directive MLX: `4 / 8`
  - argument hints v2: `5 / 8` exact and `7 / 8` executor-equivalent
  - v6 code hints: `6 / 8`
  - v7 code guard: `6 / 8` exact and `7 / 8` executor-equivalent
  - v8 hybrid label guard: `7 / 8` exact and executor-equivalent
- Interpretation:
  - v8 is the current residual strict-selector upper bound, but its advantage over argument hints and v7 is mostly exactness rather than executor-equivalent reach.
  - The persistent miss is now `state pill`, which isolates a component-role/value ambiguity: the model still prefers the visible state/content value over the named component label in that case.
  - The next research move is a focused component-role/value micro-slice before promoting v8 into packaged workflows.
- Reporting snapshot at the time of this residual result:
  - MLX tool-contract report had `72` tables and `35` figures
  - publication evidence ledger had `29` claims, `137` evidence sources, and `0` missing sources
  - publication readiness audit had `100` checks, `98` blocking checks, `0` blocking failures, and status `paper_draft_ready`

## 2026-05-10 - H1n Post-Repair Holdout Favors the Code Guard

- Built a fresh post-repair H1n holdout to test whether the v7 activation-gated oblique-code repair transfers beyond the packet that motivated it:
  - builder: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
  - suite flag: `--suite alias_transfer_post_repair_v6`
  - source packet: [`20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1)
  - no-directive packet: [`20260510T_h1n_post_repair_no_directive_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_post_repair_no_directive_execute_v1)
  - contracted/default packet: [`20260510T_h1n_post_repair_contracted_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_post_repair_contracted_execute_v1)
  - argument-hints packet: [`20260510T_h1n_post_repair_argument_hints_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_post_repair_argument_hints_execute_v1)
  - v6 code-hints packet: [`20260510T_h1n_post_repair_code_hints_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_post_repair_code_hints_execute_v1)
  - v7 code-guard packet: [`20260510T_h1n_post_repair_code_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_post_repair_code_guard_execute_v1)
  - diagnostic: [`results/reports/visual_alias_transfer_post_repair_diagnostic`](../results/reports/visual_alias_transfer_post_repair_diagnostic)
  - report table: [`visual_hard_slice_post_repair_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_post_repair_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_post_repair_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_post_repair_live_replay_gate.svg)
- Result:
  - no-directive MLX: strict/executor-equivalent `2 / 8`
  - contracted/default MLX: strict/executor-equivalent `3 / 8`
  - argument hints v2: strict/executor-equivalent `5 / 8`
  - oblique code hints v6: strict/executor-equivalent `5 / 8`
  - oblique code guard v7: strict/executor-equivalent `6 / 8`
- Mechanism shape:
  - argument hints remains better on some non-code labels, especially `status pill`
  - v6 code hints helps code-like labels and stale-selection routing but regresses `review tile` into no-tool-call
  - v7 code guard preserves the v6 code/stale gains, recovers `review tile`, and becomes the current strict/executor-equivalence upper bound on this fresh packet
  - remaining v7 misses are `chip l90` and `status pill`
- Interpretation:
  - This reverses the earlier transfer caution in an important but bounded way. Across the first oracle/repeat/oblique packets, argument hints still had the better executor-equivalence aggregate; on a fresh post-repair holdout, the activation-gated code guard is now the best row.
  - The useful research answer is not "one prompt profile wins forever." It is that visual catalog profiles have activation domains: broad argument hints help ordinary visible-region labels, while the code guard helps code-like suffixes and stale-selection traps.
  - The next best experiment is a hybrid or activation-gated profile that keeps argument-hints behavior for non-code labels while applying code-guard behavior only when code-like suffixes or stale-selection hazards appear.
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1 --suite alias_transfer_post_repair_v6`
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1n_post_repair_code_guard_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard --registry configs/model_registry.yaml --execute --json`
  - `uv run pytest tests/test_visual_live_stress_diagnostic.py tests/test_mlx_tool_contract_report.py -q`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-transfer-post-repair`
  - `uv run python scripts/build_mlx_tool_contract_report.py`

# 2026-05-06

### CLI-first live harness pivot now has a sandboxed runtime scaffold

- Runtime implementation:
  - [`src/gemma4_capability_map/runtime/sandbox.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/sandbox.py)
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
  - [`src/gemma4_capability_map/runtime/schemas.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/schemas.py)
- API/test support:
  - [`src/gemma4_capability_map/api/app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/api/app.py)
  - [`tests/test_runtime_core.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_core.py)
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)
  - [`tests/test_runtime_api.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_api.py)

- What changed:
  - frontend refinement is no longer the active workstream
  - live runs now default to `ephemeral_copy` sandbox metadata with policy id `packaged_workflow_ephemeral_v1`
  - packaged workflow and episode inputs are copied into each session sandbox
  - runtime summaries, traces, manifests, and native artifacts now write under the sandbox output root
  - `moonie-agent live` launches a packaged workflow and immediately attaches a Rich terminal operator view
  - `moonie-agent attach <session_id>` watches an existing session from the terminal
  - the live CLI defaults to `mlx_gemma4_e2b_reasoner_only`, while tests use oracle rows to keep verification local and fast

- Verification:
  - `uv lock`
  - `uv run pytest tests/test_runtime_core.py tests/test_runtime_cli.py tests/test_runtime_api.py`
  - `22 passed`
  - `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id oracle_gemma4_e2b --lane replayable_core --refresh-s 0.1 --timeout-s 0.5`
  - completed through the Rich operator view with sandbox context visible
  - `uv run pytest`
  - `244 passed`

- Research interpretation:
  - the repo now has a safer first live-operator path for studying local Gemma execution without adding more UI surface area
  - the next useful slice is not visual polish; it is operator actions, stricter live-web dry-run policy, Gemini CLI as a wrapped baseline, and a harder `H1` slice that breaks the current saturated top-line readiness

### Live-web sandbox policy blocks are now runtime-visible

- Runtime implementation:
  - [`src/gemma4_capability_map/runtime/sandbox.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/sandbox.py)
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/runtime/schemas.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/schemas.py)
- Regression coverage:
  - [`tests/test_runtime_core.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_core.py)

- What changed:
  - live-web `sandbox_only`, `approval_required`, and `blocked` actions now produce `sandbox_policy_block` events
  - sessions and runtime traces persist `sandbox_policy_blocks`
  - the runtime manifest includes the policy block payloads alongside the sandbox root and policy id

- Verification:
  - `uv run pytest tests/test_runtime_core.py tests/test_runtime_cli.py tests/test_runtime_api.py`
  - `24 passed`

### Rich operator path can now apply session actions

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
- Regression coverage:
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - `moonie-agent attach <session_id> --action approve`
  - `moonie-agent attach <session_id> --action deny`
  - `moonie-agent attach <session_id> --action resume`
  - `moonie-agent attach <session_id> --action retry`
  - `moonie-agent attach <session_id> --action quit`
  - the Rich side panel now prints the exact approval/resume commands for blocked sessions

- Verification:
  - `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py`
  - `20 passed`

### CLI inspection now exposes sandbox and artifact evidence

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
- Regression coverage:
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - `moonie-agent inspect <session_id> --target sandbox`
  - `moonie-agent inspect <session_id> --target artifacts`
  - `moonie-agent inspect <session_id> --target policy`
  - `moonie-agent inspect <session_id> --target summary`
  - `--json` makes the inspection machine-readable for harness scripts

- Verification:
  - `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py`
  - `21 passed`
  - `uv run moonie-agent inspect <latest_session> --target sandbox --json`
  - completed and showed the sandbox root plus manifest path

### CLI inspection now exposes scorecards and controller findings

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
- Regression coverage:
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - `moonie-agent inspect <session_id> --target scorecard`
  - default `--target all` now includes scorecard metrics and per-task controller findings
  - the Rich live side panel shows readiness plus repair/raw-clean at completion
  - controller findings include repair notes, raw planning outputs, and per-task repair/fallback metrics

- Verification:
  - `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py -q`
  - `23 passed`
  - `uv run moonie-agent inspect 20260506T220039380037Z_executive_visual_dashboard_review --target scorecard --json`
  - surfaced `repaired_arguments:extract_layout` on `visual_013_dashboard_stale_selection_recovery`

- Research interpretation:
  - live CLI runs are now directly useful for Gemma harnessing research: an operator can see not only that the run completed, but which controller intervention made it clean
  - the latest MLX smoke shows a concrete remaining live-path signal: the model chose the right tool, but used a semantically natural visual query that needed benchmark-canonical argument repair

### MLX Gemma live CLI smoke completed through the sandbox harness

- Smoke command:
  - `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --once --refresh-s 0.5 --timeout-s 1.0`
- Session:
  - `20260506T173247139289Z_executive_visual_dashboard_review`
- Inspection:
  - `uv run moonie-agent inspect 20260506T173247139289Z_executive_visual_dashboard_review --target all --json`

- Result:
  - status: `completed`
  - system: `mlx_gemma4_e2b_reasoner_only`
  - lane: `replayable_core`
  - sandbox manifest exists
  - artifacts: `3` `.docx` revisions, all under the sandbox output root
  - `strict_interface_score = 1.0`
  - `role_readiness_score = 0.9942`
  - `controller_repair_count = 0.5`
  - `controller_fallback_count = 0.0`
  - `raw_planning_clean_rate = 0.5`

- Interpretation:
  - the CLI-first harness can now execute a real local MLX Gemma run, persist sandboxed artifacts, and make the run inspectable from terminal commands
  - this is a smoke, not a new benchmark row; benchmark claims still need packet or aligned matrix reruns

### Second MLX Gemma live CLI smoke confirms the harness and exposes the repair cause

- Smoke command:
  - `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --once --refresh-s 0.5 --timeout-s 1.0`
- Session:
  - `20260506T220039380037Z_executive_visual_dashboard_review`
- Result:
  - status: `completed`
  - system: `mlx_gemma4_e2b_reasoner_only`
  - sandbox mode: `ephemeral_copy`
  - sandbox policy: `packaged_workflow_ephemeral_v1`
  - artifacts: `3` `.docx` revisions, all under `sandbox/output`
  - `role_readiness_score = 0.9942`
  - `strict_interface_score = 1.0`
  - `recovered_execution_score = 1.0`
  - `controller_repair_count = 0.5`
  - `argument_repair_count = 0.5`
  - `controller_fallback_count = 0.0`
  - `raw_planning_clean_rate = 0.5`
- Controller finding:
  - task: `visual_013_dashboard_stale_selection_recovery`
  - repair note: `repaired_arguments:extract_layout`
  - raw call: `{"name": "extract_layout", "arguments": {"image_id": "img-dashboard-stale", "target_query": "metric panels that need review"}}`

- Interpretation:
  - the live operator path is now better than a success/failure dashboard; it exposes the actual harness intervention
  - this repair is not a catastrophic model failure, but it is exactly the kind of local Gemma harnessing signal worth measuring: semantically reasonable arguments still need canonicalization on the benchmark contract

### Runtime live-smoke packets are now replayable and commit-friendly

- Implementation:
  - [`scripts/run_runtime_live_smoke_packet.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_runtime_live_smoke_packet.py)
- Regression coverage:
  - [`tests/test_runtime_live_smoke_packet.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_live_smoke_packet.py)

- What changed:
  - added a small packet runner over packaged workflows that uses the same `LocalAgentRuntime` path as `moonie-agent live`
  - writes `manifest.json`, `summary.json`, `sessions.json`, and `leaderboard.csv`
  - stores compact packet outputs under tracked `results/runtime_live_smoke_packets`
  - keeps raw per-session runtime state under ignored `results/runtime/sessions`

- Verification:
  - `uv run pytest tests/test_runtime_live_smoke_packet.py tests/test_runtime_cli.py tests/test_runtime_core.py -q`
  - `25 passed`
  - dry-run command:
    - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --run-group-id 20260506T_runtime_live_smoke_dry_run_v2 --dry-run`
  - real MLX packet command:
    - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --run-group-id 20260506T_runtime_live_smoke_mlx_v2`
- Packet output:
  - [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet)
  - `workflow_count = 1`
  - `failed_sessions = 0`
  - `status_counts.completed = 1`
  - `role_readiness_avg = 0.9942`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.5`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.5`
  - `controller_finding_count = 1`

- Research interpretation:
  - live CLI validation now has a durable packet shape that can be committed and compared over time
  - the next packet should include an approval-hold workflow and a live-web sandbox-policy workflow so the operator harness is tested beyond the dashboard happy path

### Runtime approval/smoke trio covers approval holds and controller findings

- Packet command:
  - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id executive_visual_dashboard_review --workflow-id finance_visual_invoice_review --workflow-id jobs_visual_form_hold --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --run-group-id 20260506T_runtime_live_smoke_mlx_trio_v2`
- Packet output:
  - [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet)
  - `workflow_count = 3`
  - `status_counts.completed = 1`
  - `status_counts.awaiting_approval = 2`
  - `failed_sessions = 0`
  - `role_readiness_avg = 0.9800333333333334`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.6666666666666666`
  - `argument_repair_avg = 0.6666666666666666`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.3333333333333333`
  - `approval_count = 2`
  - `policy_block_count = 0`
  - `controller_finding_count = 4`
- Controller finding families:
  - dashboard visual `extract_layout` argument repair
  - finance `api_update_record` argument repair
  - jobs `cli_apply_patch` argument repair
  - jobs visual `extract_layout` argument repair

- Research interpretation:
  - the replayable live-harness packet now exercises successful completion plus terminal approval holds
  - local MLX Gemma continues to complete the workflows, but this packet makes clear that controller argument repair remains materially present on live CLI execution
  - replayable approval holds do not trigger sandbox policy blocks; the next packet should use `live_web_stress` or a dedicated side-effect-gated workflow to test the sandbox policy stream

### Live-web policy packet exercises sandbox policy blocks

- Packet command:
  - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id jobs_visual_form_hold --system-id mlx_gemma4_e2b_reasoner_only --lane live_web_stress --run-group-id 20260506T_runtime_live_web_policy_mlx_v2`
- Packet output:
  - [`results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet)
  - `workflow_count = 1`
  - `status_counts.awaiting_approval = 1`
  - `failed_sessions = 0`
  - `role_readiness_avg = 0.9826`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 1.0`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.5`
  - `raw_planning_clean_rate_avg = 0.0`
  - `approval_count = 1`
  - `policy_block_count = 3`
  - `controller_finding_count = 2`
- Policy blocks:
  - two `sandbox_only` blocks for live jobs form rehydration and repair actions
  - one `approval_required` block for the live jobs resume submission attempt
  - all three blocks carry `sandbox://` targets plus `https://sandbox.local/...` endpoints
- Controller findings:
  - `cli_apply_patch` argument repair in stage 1
  - `controller_fallback_planner` on the live visual form refinement in stage 2

- Research interpretation:
  - the live CLI harness now has committed evidence for three operator-critical states: completed, awaiting approval, and sandbox policy blocked
  - this also reintroduces a real controller-fallback signal on live local MLX, so the next H1c design should include live-web visual/form pressure rather than only replayable visual readback
  - next runtime UX slice: make policy blocks easier to read directly in `moonie-agent attach` and `moonie-agent inspect --target policy`

### Repeated live-web CLI packet confirms stable MLX controller signal

- Packet command:
  - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id executive_visual_dashboard_review --workflow-id finance_visual_invoice_review --workflow-id jobs_visual_form_hold --system-id mlx_gemma4_e2b_reasoner_only --lane live_web_stress --run-group-id 20260506T_runtime_live_repeat_mlx_h1c_overlap_v2 --repeat 3`
- Packet output:
  - [`results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet)
  - `workflow_count = 3`
  - `repeat_count = 3`
  - `session_count = 9`
  - `failed_sessions = 0`
  - `status_counts.completed = 3`
  - `status_counts.awaiting_approval = 6`
  - `role_readiness_avg = 0.9818333333333334`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.6666666666666666`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.16666666666666666`
  - `raw_planning_clean_rate_avg = 0.3333333333333333`
  - `controller_finding_count = 12`
  - `policy_block_count = 21`
  - `approval_count = 6`
- Stable repair families across all three repeats:
  - executive dashboard: `repaired_arguments:extract_layout`
  - finance invoice: `repaired_arguments:cli_search_logs`
  - jobs form: `repaired_arguments:cli_apply_patch`
  - jobs live visual form: `controller_fallback_planner`
- Analyzer outputs:
  - [`runtime_packet_analysis.json`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_packet_analysis.json)
  - [`runtime_repair_family_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_repair_family_counts.csv)
  - [`runtime_policy_block_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_policy_block_counts.csv)
  - [`runtime_workflow_stability.csv`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_workflow_stability.csv)
  - `stable_repair_family_count = 4`
  - `stable_policy_block_family_count = 7`

- Research interpretation:
  - H1c benchmark execution is clean, but CLI live execution reproducibly needs controller help on overlapping packaged workflows
  - the discrepancy is now a concrete runtime-vs-benchmark question rather than a single-run anomaly
  - next slice should compare the repeated CLI findings against clean H1c traces before encoding H1d

### H1c MLX monolith rerun aligns benchmark with CLI live signal

- Harness correction:
  - H1 primary run specs now pass `--pipeline-name monolith` for `local_reasoner` systems
  - `run_knowledge_work_arena.py` now accepts `--pipeline-name`
  - this makes `mlx_gemma4_e2b_reasoner_only` match the `moonie-agent live` posture instead of silently using a modular heuristic router
- Corrected packet command:
  - `uv run python scripts/run_knowledge_work_h1_slice.py --config configs/knowledge_work_h1c_slice.yaml --run-set primary --lane live_web_stress --system-id mlx_gemma4_e2b_reasoner_only --run-group-id 20260506T_h1c_mlx_live_primary_monolith_v1`
- Corrected packet output:
  - [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
  - `real_world_readiness_avg = 0.97936`
  - `artifact_quality_avg = 0.95`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.7`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.2`
  - `raw_planning_clean_rate_avg = 0.3`
- Controller-dependent task families:
  - `visual_016_live_dashboard_stale_selection_recovery`
  - `tool_018_jobs_api_latest_form_issue`
  - `visual_022_live_form_latest_issue_referent_carryover`
  - `tool_019_finance_cli_log_search_latest_lock`
  - `tool_021_jobs_cli_patch_only_latest_email_fix`
  - `visual_030_live_form_latest_blocked_email_refinement`
  - `tool_016_finance_api_invoice_lock_update`
- Research interpretation:
  - the earlier clean H1c MLX row was a harness artifact, not evidence that local MLX Gemma was controller-clean on live-policy workflows
  - the corrected monolith benchmark now agrees with repeated CLI live smoke: local MLX Gemma finishes the workflows, but controller repair/fallback remains material
  - next slice should add local MLX monolith helper-ablation profiles and run a compact H1c ablation over the repeated families

### H1c MLX monolith helper ablation shows causal controller dependence

- Packet command:
  - `uv run python scripts/run_knowledge_work_h1_slice.py --config configs/knowledge_work_h1c_slice.yaml --run-set primary --lane live_web_stress --system-id mlx_gemma4_e2b_reasoner_only --system-id mlx_gemma4_e2b_reasoner_only_no_controller_repair --system-id mlx_gemma4_e2b_reasoner_only_no_controller_fallback --system-id mlx_gemma4_e2b_reasoner_only_no_argument_repair --run-group-id 20260506T_h1c_mlx_monolith_helpers_v1`
- Packet output:
  - [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- Row summary:
  - baseline: readiness `0.97936`, strict/recovered `1.0 / 1.0`, repair/fallback `0.7 / 0.2`, raw clean `0.3`
  - `no_controller_repair`: readiness `0.7381800000000001`, strict/recovered `0.475 / 0.3`, raw clean `0.89`
  - `no_controller_fallback`: readiness `0.92104`, strict/recovered `0.85 / 0.8`, raw clean `0.5`
  - `no_argument_repair`: readiness `0.82036`, strict/recovered `0.7125 / 0.5`, raw clean `0.8`
- Trace mining:
  - `note_count = 41`
  - `failure_candidate_count = 12`
  - dominant failure modes: `visual_stepwise_control = 6`, `repair_disabled = 5`, `fallback_planner = 4`, `argument_repair = 2`, `fallback_disabled = 2`
- Research interpretation:
  - controller repair is strongly causal for local MLX monolith on H1c
  - argument repair is also causal, especially where raw calls are semantically close but contract-wrong
  - fallback is narrower but real, concentrated in the jobs visual/form chains
  - disabled rows can show higher `raw_planning_clean_rate` while performing worse, so raw-clean must be interpreted with repair controls in mind
- H1d candidate brief:
  - [`docs/continuity/h1d-candidates.md`](/Users/cheickdiakite/Codex/moonie/docs/continuity/h1d-candidates.md)
  - [`configs/knowledge_work_h1d_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1d_slice.yaml)
  - proposed stress families: visual stepwise control, API/CLI canonicalization, fallback boundary, and approval-safe stop under repair pressure
- First named H1d packet:
  - [`results/knowledge_work_h1_slice/20260506T_h1d_mlx_controller_stress_v1_knowledge_work_h1d_mlx_monolith_controller_stress_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1d_mlx_controller_stress_v1_knowledge_work_h1d_mlx_monolith_controller_stress_v1)
  - reproduced the H1c monolith helper-ablation row values exactly
  - trace mining found `41` notes and `12` failure candidates

### CLI policy rendering now surfaces live-web block details

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
- Regression coverage:
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - the Rich side panel now shows `policy blocks` when a session has sandbox policy holds
  - `moonie-agent inspect --target policy` renders severity, gate, action, sandbox target, sandbox endpoint, and reason in a readable table
  - JSON policy inspection remains machine-readable for packet scripts

- Verification:
  - `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py tests/test_runtime_live_smoke_packet.py -q`
  - `26 passed`
  - `uv run moonie-agent inspect 20260506T221455478537Z_jobs_visual_form_hold --target policy`
  - rendered the two `sandbox_only` holds and one `approval_required` hold from the live-web packet

### H1c live-policy controller slice is scaffolded

- Config and note:
  - [`configs/knowledge_work_h1c_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1c_slice.yaml)
  - [`docs/continuity/h1c-slice.md`](/Users/cheickdiakite/Codex/moonie/docs/continuity/h1c-slice.md)
- What changed:
  - defined H1c around live-web sandbox policy gates, approval-safe stops, and the visual/API/CLI repairs surfaced by live CLI packets
  - kept packaged workflows as the only live entrypoint
  - selected five replayable/live workflow mirrors:
    - `executive_visual_dashboard_review`
    - `finance_visual_invoice_review`
    - `jobs_visual_form_hold`
    - `jobs_phone_patch_resume`
    - `finance_billing_patch_hold`
  - added compact packet `live_policy_controller_helpers` over `live_web_stress`
  - packet systems: baseline HF service specialists, `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`

- Verification:
  - `uv run pytest tests/test_knowledge_work_h1.py tests/test_runtime_cli.py tests/test_runtime_live_smoke_packet.py -q`
  - `20 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --config configs/knowledge_work_h1c_slice.yaml --dry-run --run-set primary --lane live_web_stress --output-root tmp/h1c-dry-run-smoke --run-group-id 20260506T_h1c_live_primary_dry_run_v1`
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1c_slice.yaml --packet-id live_policy_controller_helpers --run-group-id 20260506T_h1c_live_policy_packet_dry_run_v1 --dry-run`

- Research interpretation:
  - H1c is the next clean empirical target because it is tied to the fresh CLI live evidence rather than another same-shape H1/H1b rerun
  - the first execution should be the compact live-policy helper packet before any full five-episode live ablation

### H1c compact live-policy helper packet is HF-clean

- Packet command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1c_slice.yaml --packet-id live_policy_controller_helpers --run-group-id 20260506T_h1c_live_policy_packet_v1`
- Packet output:
  - [`results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet)
  - systems: baseline HF service specialists, `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`
  - episodes: `kwa_jobs_live_email_block_resume_hold_v5`, `kwa_finance_live_invoice_lock_direction_hold_v4`, `kwa_jobs_live_phone_patch_resume_hold_v4`
  - all four rows matched at `real_world_readiness_avg = 0.9779666666666667`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- Trace mining:
  - `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet`
  - `failure_candidate_count = 0`
  - note counts are only `controller_repair_disabled = 14` markers in the disabled-repair row

- Research interpretation:
  - the H1c live-policy helper packet does not restore HF service specialist helper dependence
  - this is a useful negative result because the local MLX CLI packets still show repair/fallback signal on adjacent workflows
  - next check: run the H1c MLX primary live path to separate local MLX runtime behavior from HF specialist packet behavior

### Gemini CLI baseline adapter scaffold exists

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/gemini_cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/gemini_cli.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
- Regression coverage:
  - [`tests/test_runtime_gemini_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_gemini_cli.py)
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - `moonie-agent gemini-baseline --workflow-id <id>` prepares a dry-run Gemini CLI baseline packet
  - the adapter detects `GEMINI_CLI_BIN` or `gemini` on `PATH`
  - `--execute` is explicit; default behavior writes the prompt and command packet without calling the external baseline
  - the prompt frames Gemini CLI as an external baseline, not Moonie's controller, and preserves no-public-side-effects constraints

- Verification:
  - `uv run pytest tests/test_runtime_gemini_cli.py tests/test_runtime_cli.py`
  - `11 passed`
  - `uv run moonie-agent gemini-baseline --workflow-id executive_visual_dashboard_review --lane replayable_core --output-dir tmp/gemini-baseline-smoke`
  - completed as a dry-run packet with `/usr/local/bin/gemini` detected

### H1 harder slice is defined around packaged workflow families

- Config and note:
  - [`configs/knowledge_work_h1_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1_slice.yaml)
  - [`docs/continuity/h1-slice.md`](/Users/cheickdiakite/Codex/moonie/docs/continuity/h1-slice.md)

- What changed:
  - defined `H1 v1` as the next harder slice before another broad aligned `32 / 26` rerun
  - kept the slice packaged-workflow-first so live CLI runs remain attributable to workflow families
  - selected `5` replayable and `5` live mirror episodes across executive dashboard review, executive stale brief packet, jobs visual form hold, finance billing patch hold, and finance visual invoice review
  - made controller repair, fallback, raw planning cleanliness, approval-safe stop behavior, and sandbox policy blocks the primary read fields

- Research interpretation:
  - H1 is designed to break the current top-line readiness saturation by concentrating resume, latest-instruction, stale-override, CLI/API/function-call, artifact-revision, and approval pressure
  - the next implementation slice should add a config-backed H1 runner/validator so packet execution does not depend on manually copying episode ids

### H1 slice has a config-backed runner scaffold

- Implementation:
  - [`src/gemma4_capability_map/knowledge_work/h1.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/h1.py)
  - [`scripts/run_knowledge_work_h1_slice.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_h1_slice.py)
  - [`tests/test_knowledge_work_h1.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_h1.py)

- What changed:
  - added typed H1 config loading and validation against packaged workflows and KWA episode ids
  - added run-spec construction for `primary`, `comparison`, `ablation`, and `all`
  - added a dry-run/execute script that writes an H1 manifest and delegates real execution to `scripts/run_knowledge_work_arena.py`
  - kept H1 execution exploratory and `--no-update-latest` by default

- Verification:
  - `uv run pytest tests/test_knowledge_work_h1.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py`
  - `17 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set primary --lane replayable_core --output-root tmp/h1-dry-run-smoke --run-group-id 20260506T_h1_dry_run_smoke`
  - completed and wrote one primary replayable H1 dry-run manifest

### Second-wave controller ablation controls are scaffolded

- Implementation:
  - [`src/gemma4_capability_map/research_controls.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/research_controls.py)
  - [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py)
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - [`configs/knowledge_work_h1_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1_slice.yaml)
  - [`configs/knowledge_work_matrix_ablation_32_replayable.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_ablation_32_replayable.yaml)

- What changed:
  - added `disable_intent_priority`
  - added `disable_argument_repair`
  - added `disable_deterministic_visual_follow_on`
  - exposed matching HF Gemma specialist registry rows
  - propagated flags through the arena, matrix, and H1 runner command builders
  - kept ablation-disabled marker notes out of controller repair counts

- Verification:
  - `uv run pytest tests/test_tool_planner.py tests/test_trace_metrics.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py tests/test_knowledge_work_h1.py`
  - `64 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set ablation --lane replayable_core --output-root tmp/h1-ablation-dry-run-smoke --run-group-id 20260506T_h1_ablation_dry_run_smoke`
  - completed and wrote `7` H1 replayable ablation run specs

### Final verification for the CLI/H1 pivot pass is clean

- Verification:
  - `uv run pytest`
  - `260 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set all --lane replayable_core --output-root tmp/h1-all-dry-run-smoke --run-group-id 20260506T_h1_all_dry_run_smoke`
  - completed and wrote `10` replayable H1 run specs

- Next empirical move:
  - run H1 primary on `mlx_gemma4_e2b_reasoner_only`
  - then run the H1 replayable ablation set before any broad aligned `32 / 26` rerun

### H1 primary replayable MLX Gemma run completed

- Command:
  - `uv run python scripts/run_knowledge_work_h1_slice.py --run-set primary --lane replayable_core --system-id mlx_gemma4_e2b_reasoner_only --run-group-id 20260506T_h1_mlx_gemma_primary_v1`
- Output:
  - [`results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1)
  - [`summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json)
  - [`episode_leaderboard.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1/mlx_gemma4_e2b_reasoner_only__replayable_core/episode_leaderboard.csv)

- Result:
  - runs: `5`
  - `real_world_readiness_avg = 0.9749800000000001`
  - `artifact_quality_avg = 0.9277799999999999`
  - `browser_workflow_avg = 0.9800000000000001`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `escalation_correctness_avg = 1.0`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

- Per-episode read:
  - strongest rows: `kwa_exec_visual_dashboard_brief` and `kwa_jobs_email_block_resume_hold_v5`
  - lower artifact-quality rows: `kwa_finance_invoice_lock_direction_hold_v4`, `kwa_exec_backlog_resume_hold_v5`, and `kwa_finance_diff_review_hold_v5`

- Interpretation:
  - H1 replayable is slightly harder for artifact quality than the broad aligned read, but it does not yet break MLX Gemma's controller-clean posture
  - the next useful experiment is the H1 replayable HF Gemma ablation set, not another broad aligned rerun

### H1 HF ablation packet wrapper added

- Implementation:
  - [`scripts/run_knowledge_work_h1_ablation_packet.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_h1_ablation_packet.py)
  - [`tests/test_knowledge_work_h1.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_h1.py)

- What changed:
  - added a config-backed wrapper for running the H1 ablation wave through the existing shared-bundle ablation packet runner
  - avoids warming the HF Gemma specialist bundle once per ablation row
  - preserves H1 episode filters and configured ablation row attribution

- Next command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_ablation_v1`

### H1 ablation posture switched to service-backed HF rows

- What happened:
  - attempted the H1 replayable ablation packet with the in-process HF specialist bundle
  - the process stayed pre-child-manifest after roughly ten minutes and was stopped
  - no episode results were produced from that attempt

- What changed:
  - added service-backed HF specialist ablation rows to [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - updated [`configs/knowledge_work_h1_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1_slice.yaml) so H1 ablation uses `hf_service_gemma4_specialists_cpu` as the shared bundle and service-backed ablation row ids

- Interpretation:
  - this is an execution-posture finding, not a model-quality result
  - H1 ablation should run through the service-backed HF primitive on this machine before spending more time on in-process HF warmup behavior

### H1 service-backed HF ablation packet completed

- Command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_ablation_v2`
- Output:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet)
  - [`results.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/results.json)
  - [`manifest.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/manifest.json)

- Runtime posture:
  - shared reasoner: `hf_service` on `google/gemma-4-E2B-it`, `mps`, service id `google__gemma_4_E2B_it_auto`
  - specialists: in-process `hf` FunctionGemma router on CPU and in-process `hf` EmbeddingGemma retriever on CPU
  - the earlier v1 failure showed why this split matters: the reasoner can be service-backed, but the specialist adapters currently need `hf`, not `hf_service`

- Result:
  - baseline `hf_service_gemma4_specialists_cpu`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`, controller repair `0.9`, fallback `0.6`, clean rate `0.1`
  - `no_controller_repair`: readiness `0.7194`, strict `0.4`, recovered `0.4`
  - `no_controller_fallback`: readiness `0.7596999999999999`, strict `0.475`, recovered `0.4`
  - `no_visual_rescue`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`
  - `no_intent_priority`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`
  - `no_argument_repair`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`
  - `no_deterministic_visual_follow_on`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`, controller repair rose to `2.1`

- Interpretation:
  - H1 successfully breaks the saturated top-line read for HF Gemma controller ablations
  - controller repair and controller fallback are causal on the H1 packaged-workflow slice
  - visual rescue, intent priority, argument repair, and deterministic visual follow-on do not move readiness on this H1 slice
  - the best next slice is trace mining for the repair/fallback rows, then a targeted H1b packet around `controller_fallback_planner` and malformed/raw planning spillover

### H1 trace-note miner added

- Implementation:
  - [`src/gemma4_capability_map/knowledge_work/trace_analysis.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/trace_analysis.py)
  - [`scripts/analyze_knowledge_work_h1_traces.py`](/Users/cheickdiakite/Codex/moonie/scripts/analyze_knowledge_work_h1_traces.py)
  - [`tests/test_knowledge_work_trace_analysis.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_trace_analysis.py)
- Command:
  - `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet`
- Output:
  - [`trace_note_summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_note_summary.json)
  - [`trace_note_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_note_counts.csv)
  - [`trace_episode_failures.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_episode_failures.csv)
  - [`trace_failure_mode_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_failure_mode_counts.csv)

- Verification:
  - `uv run pytest tests/test_knowledge_work_trace_analysis.py`
  - `2 passed`

- Trace read:
  - mined `102` controller-note events across `35` H1 episode rows
  - found `10` strict/recovered failure candidates
  - failure candidates now carry coarse `failure_modes` labels such as `raw_refusal`, `generic_tool_name`, `fallback_disabled`, `repair_disabled`, `argument_repair`, and `intent_prior`
  - aggregate mode counts: `raw_refusal = 10`, `generic_tool_name = 7`, `fallback_disabled = 5`, `fallback_planner = 5`, `repair_disabled = 5`
  - baseline still uses `controller_fallback_planner` `6` times across all `5` H1 episodes
  - `no_controller_fallback` fails all `5` H1 episodes, mostly raw refusal/no-call cases
  - `no_controller_repair` fails all `5` H1 episodes, mostly repeated generic `tool_name` hallucinations and unprojected arguments
  - disabling deterministic visual follow-on reintroduces `feedback_prior:refine_selection` and `feedback_prior:read_region_text`, but readiness still does not move

- Interpretation:
  - the next useful controller slice is now a trace-mined regression packet for raw refusal and generic `tool_name` planning failures
  - visual rescue remains deprioritized for this H1 family

### FunctionGemma prompt no longer seeds literal `tool_name` placeholders

- Implementation:
  - [`src/gemma4_capability_map/models/functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/functiongemma_runner.py)
  - [`tests/test_functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_functiongemma_runner.py)
- What changed:
  - replaced the router prompt's literal `call:tool_name{arg:...}` format hint with a catalog-specific example using a real allowed tool and schema field
  - added an explicit instruction not to emit placeholder names such as `tool_name` or `arg`

- Verification:
  - `uv run pytest tests/test_functiongemma_runner.py tests/test_tool_parsing.py tests/test_knowledge_work_trace_analysis.py`
  - `4 passed`

- H1 baseline canary:
  - command: `uv run python scripts/run_knowledge_work_ablation_packet.py --lane replayable_core --bundle-system-id hf_service_gemma4_specialists_cpu --output-root results/knowledge_work_h1_slice --run-group-id 20260506T_h1_functiongemma_prompt_canary_v1 --run-intent exploratory --system-id hf_service_gemma4_specialists_cpu --episode-id kwa_exec_visual_dashboard_brief --episode-id kwa_exec_backlog_resume_hold_v5 --episode-id kwa_jobs_email_block_resume_hold_v5 --episode-id kwa_finance_diff_review_hold_v5 --episode-id kwa_finance_invoice_lock_direction_hold_v4`
  - output: [`results/knowledge_work_h1_slice/20260506T_h1_functiongemma_prompt_canary_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_functiongemma_prompt_canary_v1_knowledge_work_ablation_packet)
  - readiness stayed `0.9749800000000001`
  - strict/recovered stayed `1.0 / 1.0`
  - `controller_fallback_avg` moved from `0.6` to `0.3`
  - `controller_repair_avg` moved from `0.9` to `0.8`
  - `raw_planning_clean_rate_avg` moved from `0.1` to `0.2`
  - `argument_repair_avg` rose from `0.1` to `0.5`
  - trace miner found `0` failure candidates and `controller_fallback_planner = 3`

- Interpretation:
  - this appears to remove part of the generic `tool_name` failure pressure while preserving readiness
  - it shifts some remaining burden into argument repair, so the next run should be the full H1 ablation packet with this prompt patch before a broader aligned rerun

### Full H1 packet after the FunctionGemma prompt patch

- Command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_prompt_patch_ablation_v1`
- Output:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet)
  - [`results.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet/results.json)
  - [`trace_failure_mode_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet/trace_failure_mode_counts.csv)

- Result:
  - baseline stayed saturated: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`
  - baseline controller burden improved versus pre-patch H1: repair `0.9 -> 0.8`, fallback `0.6 -> 0.3`, clean rate `0.1 -> 0.2`
  - `no_controller_repair`: readiness `0.7194 -> 0.7319`, strict `0.4 -> 0.475`, recovered stayed `0.4`
  - `no_controller_fallback`: readiness `0.7596999999999999 -> 0.8606`, strict `0.475 -> 0.725`, recovered `0.4 -> 0.7`
  - `no_visual_rescue`, `no_intent_priority`, `no_argument_repair`, and `no_deterministic_visual_follow_on` all stayed at readiness `0.9749800000000001`

- Trace read:
  - command: `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet`
  - controller-note events dropped from `102` to `93`
  - strict/recovered failure candidates dropped from `10` to `7`
  - aggregate `generic_tool_name` failures dropped from `7` to `0`
  - aggregate raw refusals dropped from `10` to `5`
  - post-patch aggregate mode counts are now `raw_refusal = 5`, `repair_disabled = 4`, `fallback_disabled = 3`, `argument_repair = 2`, `fallback_planner = 2`
  - baseline `controller_fallback_planner` appears `3` times across `3` H1 episodes instead of `6` times across all `5`

- Interpretation:
  - the literal placeholder format hint was a real harness-induced failure source
  - controller fallback dependence is lower after removing that prompt seed, but not gone
  - controller repair remains the stronger causal helper on H1
  - the next controller patch should target refusal-to-tool-contract behavior and unrepaired real-tool placeholder arguments, not visual rescue or broad UI work

### FunctionGemma concrete request-specific hint canary

- Implementation:
  - [`src/gemma4_capability_map/models/functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/functiongemma_runner.py)
  - [`tests/test_functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_functiongemma_runner.py)
- What changed:
  - the router prompt now uses a request-specific example generated from the existing planner rather than a schema-only example with `<escape>value<escape>`
  - the system prompt explicitly bans placeholder values such as `value`, `example`, and `placeholder`

- Verification:
  - `uv run pytest tests/test_functiongemma_runner.py tests/test_tool_planner.py tests/test_tool_parsing.py tests/test_knowledge_work_trace_analysis.py`
  - `47 passed`
  - `uv run pytest tests/test_functiongemma_runner.py tests/test_tool_planner.py tests/test_knowledge_work_h1.py tests/test_knowledge_work_trace_analysis.py`
  - `51 passed`

- H1 baseline canary:
  - command: `uv run python scripts/run_knowledge_work_ablation_packet.py --lane replayable_core --bundle-system-id hf_service_gemma4_specialists_cpu --output-root results/knowledge_work_h1_slice --run-group-id 20260506T_h1_functiongemma_concrete_hint_canary_v1 --run-intent exploratory --system-id hf_service_gemma4_specialists_cpu --episode-id kwa_exec_visual_dashboard_brief --episode-id kwa_exec_backlog_resume_hold_v5 --episode-id kwa_jobs_email_block_resume_hold_v5 --episode-id kwa_finance_diff_review_hold_v5 --episode-id kwa_finance_invoice_lock_direction_hold_v4`
  - output: [`results/knowledge_work_h1_slice/20260506T_h1_functiongemma_concrete_hint_canary_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_functiongemma_concrete_hint_canary_v1_knowledge_work_ablation_packet)
  - readiness stayed `0.9749800000000001`
  - strict/recovered stayed `1.0 / 1.0`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
  - trace miner found `0` controller-note events and `0` failure candidates

- Interpretation:
  - the remaining baseline H1 controller burden was prompt-shape-induced: concrete request-specific examples remove both the copied `value` argument placeholders and raw refusal/no-call fallback cases on the baseline canary
  - the next required evidence is the full H1 ablation after this stronger prompt prior, because it may reduce or erase apparent repair/fallback causality on this narrow slice

### Full H1 ablation after the concrete FunctionGemma hint

- Command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_concrete_hint_ablation_v1`
- Output:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet)
  - [`results.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet/results.json)
  - [`trace_failure_mode_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet/trace_failure_mode_counts.csv)

- Result:
  - baseline stayed saturated and controller-clean: readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, repair `0.0`, fallback `0.0`, raw clean `1.0`
  - `no_controller_fallback` now matches baseline: readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - `no_controller_repair` improved substantially but still lags: readiness `0.88748`, strict/recovered `0.775 / 0.7`, raw clean `0.89`
  - `no_deterministic_visual_follow_on` now also lags: readiness `0.88748`, strict/recovered `0.775 / 0.7`, repair `0.8`, fallback `0.4`
  - `no_visual_rescue`, `no_intent_priority`, and `no_argument_repair` all stayed at readiness `0.9749800000000001`

- Trace read:
  - command: `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`
  - controller-note events dropped to `42`
  - strict/recovered failure candidates dropped to `6`
  - the richer visual taxonomy now labels `visual_readback_missing = 6`, `visual_stepwise_control = 6`, `fallback_planner = 4`, `argument_repair = 3`, `raw_refusal = 3`, `repair_disabled = 3`, `visual_follow_on = 3`, and `visual_repeated_refinement = 3`
  - generic placeholder modes are absent from the aggregate failure taxonomy

- Interpretation:
  - fallback dependence on H1 was mostly a prompt-shape artifact
  - repair dependence remains, but its current family is stepwise visual control and future-state visual calls rather than generic placeholders
  - deterministic visual follow-on has become causally visible again once the router prompt is clean
  - the next controller slice should expand failure taxonomy around visual multi-call batches, future selection ids, and missing deterministic follow-ons

# 2026-04-14

### The React Gemma MLX workspace now runs a real end-to-end local session loop

- Product/frontend implementation:
  - [`frontend/src/App.tsx`](/Users/cheickdiakite/Codex/moonie/frontend/src/App.tsx)
  - [`frontend/src/api.ts`](/Users/cheickdiakite/Codex/moonie/frontend/src/api.ts)
  - [`frontend/src/styles.css`](/Users/cheickdiakite/Codex/moonie/frontend/src/styles.css)
- Verification:
  - `npm run build`
  - `uv run pytest tests/test_runtime_api.py tests/test_runtime_core.py`
  - live browser verification against:
    - `uv run moonie-agent-api --host 127.0.0.1 --port 8765`
    - `npm run dev -- --host 127.0.0.1 --port 5174`

- What changed:
  - the desktop shell now behaves like a real local agent workspace instead of a styled snapshot viewer
  - the frontend API client now has backend health checks and abortable requests
  - the workspace now long-polls the real session stream endpoint while a session is active
  - the status strip exposes backend state, runtime posture, and session-loop state directly in the shell
  - the stream payload now overrides stale list responses locally so completed sessions settle correctly in the rail
  - the desktop shell was tightened visually toward the reference:
    - quieter project selection
    - slimmer composer
    - fewer heavy nested boxes in the browser pane
    - mobile-only controls hidden on desktop

- Live product verification:
  - launched a fresh `mlx_gemma4_e2b_reasoner_only` `Dashboard Visual Review` session from the React UI
  - observed `created -> instruction_updated -> warming -> running -> artifacts_ready -> completed` in the center timeline
  - verified browser preview and browser-state events in the right pane
  - confirmed the project rail updated from `running` to `completed` after the stream/list race fix

- Product interpretation:
  - Moonie now has a real showable local harness loop for Gemma on MLX, not just a design shell
  - the next frontend question is no longer “can the UI talk to the backend?”; it is how far to push the desktop host and artifact/browser fidelity

# 2026-04-13

### The main Gemma MLX product harness is now a real React frontend over the local API

- Product/frontend implementation:
  - [`frontend/src/App.tsx`](/Users/cheickdiakite/Codex/moonie/frontend/src/App.tsx)
  - [`frontend/src/api.ts`](/Users/cheickdiakite/Codex/moonie/frontend/src/api.ts)
  - [`frontend/src/types.ts`](/Users/cheickdiakite/Codex/moonie/frontend/src/types.ts)
  - [`frontend/src/styles.css`](/Users/cheickdiakite/Codex/moonie/frontend/src/styles.css)
  - [`frontend/package.json`](/Users/cheickdiakite/Codex/moonie/frontend/package.json)
  - [`frontend/vite.config.ts`](/Users/cheickdiakite/Codex/moonie/frontend/vite.config.ts)
- API support:
  - [`src/gemma4_capability_map/api/app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/api/app.py)
- Regression coverage:
  - [`tests/test_runtime_api.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_api.py)

- What changed:
  - the main product harness direction is no longer Streamlit
  - the repo now has a proper React desktop shell backed by `moonie-agent-api`
  - the shell implements the intended three-pane structure directly:
    - left project/thread rail
    - center conversation/composer workspace
    - right `Summary` / `Review` / `Browser` context
  - the browser pane now uses real runtime/API data, local file preview, and browser-state events rather than a benchmark-only placeholder
  - the API now supports browser-frontend needs directly via CORS preflight and safe local file serving

- Verification:
  - frontend build: `npm run build`
  - API/runtime regressions: `15 passed`
  - live browser smoke:
    - `uv run moonie-agent-api --host 127.0.0.1 --port 8765`
    - `npm run dev -- --host 127.0.0.1 --port 5173`
    - verified in-browser at `http://127.0.0.1:5173`

- Research/product interpretation:
  - Moonie now has a stronger answer to “what can people actually use?” than “open Streamlit”
  - the screenshot reference turned out to be architectural guidance, not a styling exercise
  - a proper local Gemma harness needs a real frontend shell over the runtime API; the right next question is the eventual desktop host for embedded browser behavior

### A dedicated Gemma MLX workspace now exists as a real product harness surface

- Product/runtime wiring:
  - [`src/gemma4_capability_map/app/views/gemma_mlx_workspace.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/views/gemma_mlx_workspace.py)
  - [`src/gemma4_capability_map/app/streamlit_app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/streamlit_app.py)
  - [`src/gemma4_capability_map/app/view_models.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/view_models.py)
  - [`src/gemma4_capability_map/app/assets/console.css`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/assets/console.css)
  - [`src/gemma4_capability_map/app/theme.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/theme.py)
- Regression coverage:
  - [`tests/test_view_models.py`](/Users/cheickdiakite/Codex/moonie/tests/test_view_models.py)

- What changed:
  - the Streamlit router now exposes `gemma_mlx_workspace` as the primary desktop harness surface
  - the workspace defaults to `mlx_gemma4_e2b_reasoner_only`
  - sessions are grouped by `project_id` in a left rail
  - the center pane is now a conversation/composer surface instead of a benchmark dashboard
  - the right pane keeps `Summary`, `Review`, and `Browser` context attached to the selected runtime session

- Research/product interpretation:
  - Moonie now has a stronger answer to “what can people actually use?” than “open the board”
  - the benchmark/runtime substrate is now materially closer to a real local agent shell
  - the next product questions shift from “can we render the data?” to “how live and browser-native can this workspace become while staying benchmark-backed?”

### Deterministic visual follow-ons removed a real controller-burden artifact without changing outcomes

- Runtime/controller patch:
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py)
  - the runtime now auto-executes deterministic `refine_selection` / `read_region_text` follow-ons after successful visual tool feedback instead of asking the model to plan those same steps again
- Regression updates:
  - [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py)
  - [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)
  - [`tests/test_trace_metrics.py`](/Users/cheickdiakite/Codex/moonie/tests/test_trace_metrics.py)

- Focused replayable packet rerun:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet)
- Aligned HF Gemma rerun:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26)

- Focused packet baseline after the patch:
  - `real_world_readiness_avg = 0.9627777777777777`
  - `controller_repair_avg = 0.8888888888888888`
  - `controller_fallback_avg = 0.4444444444444444`
- Focused helper ranking still shows real controller leverage:
  - `no_controller_repair = 0.6551777777777779`
  - `no_controller_fallback = 0.8182333333333333`
  - `no_visual_rescue = 0.9627777777777777`

- Direct packet comparison versus the prior focused baseline:
  - readiness unchanged
  - `controller_repair_avg` dropped from `2.3333333333333335` to `0.8888888888888888`
  - `feedback_prior:refine_selection` dropped from `16` to `0`
  - `feedback_prior:read_region_text` dropped from `10` to `0`
  - `controller_fallback_planner` remained at `8`

- HF Gemma specialist delta on the aligned `32 / 26` surface:
  - replayable:
    - `controller_repair_avg` improved from `1.296875` to `0.71875`
    - `controller_fallback_avg` stayed `0.28125`
    - readiness stayed `0.976853125`
  - live:
    - `controller_repair_avg` improved from `1.5192307692307692` to `0.8076923076923077`
    - `controller_fallback_avg` stayed `0.23076923076923078`
    - readiness stayed `0.9791653846153847`

- Research interpretation:
  - the old visual follow-on repair families were real controller-burden artifacts
  - removing them did not change outcomes, which means they were not the causal value in the controller
  - repair and fallback are still clearly causal, but the remaining burden is now more honestly concentrated in fallback-planner and non-visual repair families

### Controller-burden cleanup improved the HF Gemma specialist row without moving top-line readiness

- Planner/controller patch:
  - [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py)
  - the planner now synthesizes obvious priority replacement calls directly instead of falling through to broad fallback behavior in several follow-on repair cases
- Regression updates:
  - [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py)

- Focused replayable packet rerun:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet)
- Aligned full-lane rerun:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26)

- Focused packet baseline:
  - `real_world_readiness_avg = 0.9627777777777777`
  - `controller_repair_avg = 2.3333333333333335`
  - `controller_fallback_avg = 0.4444444444444444`
- Focused helper ranking:
  - `no_controller_repair = 0.6551777777777779`
  - `no_controller_fallback = 0.8182333333333333`
  - `no_visual_rescue = 0.9627777777777777`
- Direct packet comparison versus the older baseline packet:
  - readiness unchanged
  - `controller_fallback_avg` dropped from `2.0555555555555554` to `0.4444444444444444`
  - dominant `controller_fallback_planner` notes dropped from `37` to `8`

- Current aligned replayable `32` headline rows after the patch:
  - oracle:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.578125`
    - `controller_fallback_avg = 0.0`
  - HF Gemma specialists:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 1.296875`
    - `controller_fallback_avg = 0.28125`
    - `raw_planning_clean_rate_avg = 0.46875`
  - MLX Qwen:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
  - MLX Gemma:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`

- HF Gemma specialist delta versus the prior aligned run:
  - replayable:
    - `controller_repair_avg` improved from `2.046875` to `1.296875`
    - `controller_fallback_avg` improved from `1.03125` to `0.28125`
    - readiness stayed `0.976853125`
  - live:
    - `controller_repair_avg` improved from `2.3653846153846154` to `1.5192307692307692`
    - `controller_fallback_avg` improved from `1.0769230769230769` to `0.23076923076923078`
    - readiness stayed `0.9791653846153847`

- Research interpretation:
  - the current aligned surface is no longer about top-line parity; that is already solved
  - the real Gemma question is now how much controller burden can be removed while holding the same readiness tier
  - the latest planner/controller patch proved that this burden is reducible by controller design, not only by model change
  - the next best targets are now explicit:
    - `feedback_prior:refine_selection`
    - `feedback_prior:read_region_text`

### The MLX Gemma executive-assistant judgment seam is closed, and the remaining headline gap is HF controller dependence

- Narrow runtime patch:
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - added a grounded ambiguity-aware clarify fallback for judgment tasks that still defer after second-pass rescue
  - gated narrowly on ambiguous vendor-calendar state so it does not broaden unrelated judgment modes
- New regression coverage:
  - [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)
  - verifies the fallback recovers the ambiguous vendor-meeting task even when the model keeps answering `defer`
- Targeted aligned rerun:
  - [`results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26)

- Current aligned replayable `32` headline rows:
  - oracle:
    - `real_world_readiness_avg = 0.976853125`
  - HF Gemma specialists:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 2.046875`
    - `controller_fallback_avg = 1.03125`
    - `raw_planning_clean_rate_avg = 0.46875`
  - MLX Qwen:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
  - MLX Gemma:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`

- Current aligned live `26` headline rows:
  - oracle:
    - `real_world_readiness_avg = 0.9791653846153847`
  - HF Gemma specialists:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 2.3653846153846154`
    - `controller_fallback_avg = 1.0769230769230769`
  - MLX Qwen:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.0`
  - MLX Gemma:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.0`

- The old MLX Gemma misses were exact and narrow:
  - replayable:
    - `kwa_exec_travel_conflict_resolution`
    - `kwa_exec_vendor_access_hold`
  - live:
    - `kwa_exec_live_calendar_policy`
    - `kwa_exec_live_vendor_access_hold`
  - they were scorecard-clean except for `escalation_correctness`
  - trace evidence showed the same bad move each time:
    - premature `defer` / missing-approval language
    - instead of the ambiguity-aware `clarify which vendor meeting` move
  - the new fallback now fires on those traces with:
    - `judgment_fallback_used = True`
    - `judgment_fallback_answer = action: clarify ...`

- Research interpretation:
  - the old MLX Gemma gap was not a visual grounding gap
  - it was not a tool-execution gap
  - it was a narrow executive-assistant ambiguity / escalation-language judgment seam
  - once that seam is patched, MLX Gemma reaches the same top-line aligned readiness tier as oracle, HF Gemma specialists, and MLX Qwen
  - the remaining headline research problem is now clearer:
    - HF Gemma specialist controller dependence is the differentiating gap
    - not MLX Gemma readiness

### The focused replayable ablation packet now has a clean helper ranking: repair and fallback are structural, visual rescue is not

- Packet batch:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v1_knowledge_work_ablation_packet)
- Baseline focused `9`-episode row:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9627777777777777`
  - `controller_repair_avg = 3.9444444444444446`
  - `controller_fallback_avg = 2.0555555555555554`
  - `raw_planning_clean_rate_avg = 0.16666666666666666`
- `no_controller_repair` focused row after rescoring:
  - `strict_interface_avg = 0.2777777777777778`
  - `recovered_execution_avg = 0.2777777777777778`
  - `real_world_readiness_avg = 0.6551777777777779`
  - `controller_repair_avg = 0.4444444444444444`
  - `controller_fallback_avg = 0.4444444444444444`
  - `raw_planning_clean_rate_avg = 0.8055555555555556`
- `no_controller_fallback` focused row:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v2_knowledge_work_ablation_packet/hf_gemma4_e2b_specialists_cpu_no_controller_fallback__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v2_knowledge_work_ablation_packet/hf_gemma4_e2b_specialists_cpu_no_controller_fallback__replayable_core/summary.json)
  - `strict_interface_avg = 0.3055555555555556`
  - `recovered_execution_avg = 0.16666666666666666`
  - `real_world_readiness_avg = 0.6824333333333333`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.6111111111111112`
- `no_visual_rescue` focused row:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v2_knowledge_work_ablation_packet/hf_gemma4_e2b_specialists_cpu_no_visual_rescue__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v2_knowledge_work_ablation_packet/hf_gemma4_e2b_specialists_cpu_no_visual_rescue__replayable_core/summary.json)
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9627777777777777`
  - identical to the baseline packet on this slice
- Interpretation:
  - controller repair is doing real capability work on this packet
  - controller fallback is also doing real capability work on this packet
  - visual rescue is not carrying the current focused parity result
  - the helper ranking on this 9-episode slice is now explicit:
    - repair: essential
    - fallback: essential
    - visual rescue: low or zero leverage on this slice
  - dominant packet note families:
    - `controller_fallback_planner = 37`
    - `feedback_prior:refine_selection = 16`
    - `feedback_prior:read_region_text = 10`
  - the next useful work is no longer “finish the packet”
  - it is:
    - inspect which repair/fallback note families dominate these episodes
    - reduce HF Gemma controller dependence without losing the current top-line readiness tier

# 2026-04-12

### The next real Gemma learning is now explicit: controller burden is concentrated, MLX Gemma’s residual gap is judgment-specific, and the `31B` lane is blocked by a missing local artifact

- Added research-ablation controls through the runtime stack:
  - [`src/gemma4_capability_map/research_controls.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/research_controls.py)
  - [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py)
  - [`src/gemma4_capability_map/pipelines/base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py)
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/knowledge_work/runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/runner.py)
  - [`scripts/run_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_arena.py)
  - [`scripts/run_knowledge_work_matrix.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_matrix.py)
- Added registry-backed ablation rows:
  - `hf_gemma4_e2b_specialists_cpu_no_controller_repair`
  - `hf_gemma4_e2b_specialists_cpu_no_controller_fallback`
  - `hf_gemma4_e2b_specialists_cpu_no_visual_rescue`
- Added a replayable ablation matrix config:
  - [`configs/knowledge_work_matrix_ablation_32_replayable.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_ablation_32_replayable.yaml)
- Added a shared-bundle ablation packet runner:
  - [`scripts/run_knowledge_work_ablation_packet.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_ablation_packet.py)
- Interpretation:
  - this is the right next research seam, because the current question is no longer “can Gemma tie on top-line readiness?”
  - it is “where is the harness still compensating for Gemma, and which of those compensations are actually necessary?”

- Concentrated replayable controller-burden finding from the aligned `32`-episode Gemma specialist row:
  - episodes with any controller help: `24 / 32`
  - total controller repairs: `65.5`
  - total controller fallbacks: `33.0`
  - total intent overrides: `3.5`
  - a focused 9-episode packet already explains most of that burden:
    - `kwa_exec_backlog_resume_hold_v5`
    - `kwa_jobs_email_block_resume_hold_v5`
    - `kwa_exec_latest_action_resume_hold_v4`
    - `kwa_jobs_phone_patch_resume_hold_v4`
    - `kwa_finance_invoice_lock_direction_hold_v4`
    - `kwa_exec_visual_dashboard_referent_hold_v3`
    - `kwa_jobs_visual_latest_issue_hold_v3`
    - `kwa_finance_visual_invoice_revision_hold_v2`
    - `kwa_jobs_visual_constraint_override_hold_v2`
  - that packet accounts for:
    - `35.5 / 65.5` total controller repairs
    - `18.5 / 33.0` total controller fallbacks
    - `1.5 / 3.5` total intent overrides
  - interpretation:
    - the current Gemma controller burden is concentrated enough that a focused ablation packet is the right next experiment
    - the burden lives mainly in:
      - visual KWA
      - resume / project-memory episodes
      - latest-instruction direction-following
      - API / CLI tool-selection surfaces

- The residual aligned MLX Gemma gap is now much clearer:
  - replayable `mlx_gemma4_e2b_reasoner_only` remains strict/recovered clean and controller-clean at:
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `controller_repair_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
    - `real_world_readiness_avg = 0.9725125`
  - the meaningful replayable readiness loss is concentrated in:
    - `kwa_exec_travel_conflict_resolution`
      - oracle / MLX Qwen readiness: `0.9769`
      - MLX Gemma readiness: `0.8843`
      - sole scorecard difference: `escalation_correctness = 0.0`
    - `kwa_exec_vendor_access_hold`
      - oracle / MLX Qwen readiness: `0.975`
      - MLX Gemma readiness: `0.9287`
      - sole scorecard difference: `escalation_correctness = 0.5`
  - trace evidence shows the same miss:
    - oracle / MLX Qwen retain the ambiguity-aware “clarify which vendor meeting” move
    - MLX Gemma drifts to premature defer / missing-approval language on the ambiguous vendor-meeting task
  - interpretation:
    - the residual MLX Gemma gap is not a visual-tool gap
    - it is an executive-assistant judgment / escalation-language gap

- The experimental Gemma `31B` `GGUF` / `llama.cpp` lane is now blocked by local runtime posture, not code support:
  - preflight:
    - [`results/tables/backend_preflight.md`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.md)
  - current state:
    - `llama_cpp_gemma4_31b_reasoner_only` is registered
    - `google/gemma-4-31b-it` still resolves to the remote HF identifier rather than a local `GGUF` path
    - `GEMMA4_31B_GGUF_PATH` is unset on this machine
    - no local Gemma `31B` `GGUF` artifact is present under `/Users/cheickdiakite/models`
  - interpretation:
    - the lane is ready in the registry and runtime
    - the next blocker is just the actual local artifact plus path wiring

- Operational research finding:
  - the in-process HF Gemma specialist warm path is itself a benchmark bottleneck
  - repeated ablation reruns spend too much time on bundle warmup
  - that is why the new shared-bundle ablation packet runner exists
  - this is not just ops trivia:
    - `ops reality is benchmark reality`

- Verification:
  - focused ablation/runtime regressions: `69 passed`
  - `git diff --check`: clean

### Oracle, HF Gemma specialists, MLX Qwen, and MLX Gemma are now aligned on the same exploratory `32 / 26` surface

- Ran the aligned widening batch:
  - [`results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26)
- Aligned rows now exist for:
  - `oracle_gemma4_e2b`
  - `hf_gemma4_e2b_specialists_cpu`
  - `mlx_qwen3_8b_reasoner_only`
  - `mlx_gemma4_e2b_reasoner_only`
- Replayable `32`:
  - oracle:
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.578125`
    - `raw_planning_clean_rate_avg = 0.8395875`
  - HF Gemma specialists:
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 2.046875`
    - `controller_fallback_avg = 1.03125`
    - `raw_planning_clean_rate_avg = 0.46875`
  - MLX Qwen:
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
  - MLX Gemma:
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9725125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
- Live `26`:
  - oracle:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.7115384615384616`
  - HF Gemma specialists:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 2.3653846153846154`
    - `controller_fallback_avg = 1.0769230769230769`
  - MLX Qwen:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.0`
  - MLX Gemma:
    - `real_world_readiness_avg = 0.973823076923077`
    - `controller_repair_avg = 0.0`
- Interpretation:
  - oracle, HF Gemma specialists, and MLX Qwen now tie on top-line replayable and live readiness on the aligned widened surface
  - HF Gemma specialists still rely on materially more controller repair and fallback than MLX Qwen
  - MLX Gemma is now aligned on the same surface and stays planner-clean, but still lands slightly lower readiness
  - the next useful work is no longer alignment; it is reducing HF Gemma controller dependence and understanding the residual MLX Gemma readiness gap

### MLX Gemma E2B is now a real completed posture, and the board now prefers broader completed rows over stale scope labels

- Fixed board latest-row selection in [`src/gemma4_capability_map/reporting/knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/reporting/knowledge_work_board.py):
  - completed status still wins first
  - then coverage
  - then observed `episode_count`
  - then `run_scope`
  - then timestamp
- Added regression coverage in [`tests/test_knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_board.py) so a broader newer completed subset row now beats an older smaller `full_lane` row.
- Rebuilt history/board exports:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_board_runs.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_runs.csv)
- This corrects the stale live-row selection seam:
  - the latest `hf_gemma4_e2b_specialists_cpu` live row is now the widened `23`-episode `20260412T221500Z_knowledge_work_publishable_core` run rather than the older `20`-episode row

- Confirmed Apple-Silicon-native Gemma runtime availability:
  - MLX probe:
    - [`results/raw/mlx_gemma_e2b_probe.json`](/Users/cheickdiakite/Codex/moonie/results/raw/mlx_gemma_e2b_probe.json)
  - warm harness:
    - [`results/raw/warm_harness_mlx_gemma4_e2b_current.json`](/Users/cheickdiakite/Codex/moonie/results/raw/warm_harness_mlx_gemma4_e2b_current.json)
- First full reproduced MLX Gemma batches:
  - replayable:
    - initial row:
      - [`results/knowledge_work_matrix/20260412T232827Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T232827Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json)
    - refreshed row after the grounded visual readback fallback:
      - [`results/knowledge_work_matrix/20260412T234506Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T234506Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json)
    - `runs = 32`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9725125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
  - live:
    - [`results/knowledge_work_matrix/20260412T233015Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T233015Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__live_web_stress/summary.json)
    - `runs = 26`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.973823076923077`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
- Concentrated replayable miss:
  - `kwa_exec_backlog_resume_hold_v5`
  - `strict_interface_score = 1.0`
  - `recovered_execution_score = 0.5`
  - `role_readiness_score = 0.8855`
  - no controller repairs or fallbacks
- Root cause and fix:
  - the tool path was already correct
  - the final prose drifted away from the grounded `read_region_text` output
  - added a generic visual readback fallback in [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - added regression coverage in [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)
  - targeted replayable episode rerun:
    - [`results/knowledge_work/mlx_gemma4_e2b_backlog_resume_smoke_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_gemma4_e2b_backlog_resume_smoke_v1/summary.json)
    - `recovered_execution_avg = 1.0`
- Interpretation:
  - MLX Gemma is now a real benchmark posture, not just an attempted lane
  - the harness/controller improvements transfer to the Apple-Silicon-native Gemma path
  - this directly enabled the aligned `32 / 26` reruns for oracle, Gemma specialists, and Qwen

- Verification:
  - board/reporting slice: `17 passed`
  - fallback/reporting regressions: `33 passed`

### Planner-gap metrics expose the difference between top-line parity and raw tool-use cleanliness

- Added explicit harness-gap metrics from task trace -> episode scorecard -> leaderboard export -> board row:
  - `controller_repair_count`
  - `argument_repair_count`
  - `controller_fallback_count`
  - `intent_override_count`
  - `raw_planning_clean_rate`
- Threaded these through:
  - [`src/gemma4_capability_map/metrics/trace_metrics.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/trace_metrics.py)
  - [`src/gemma4_capability_map/knowledge_work/scoring.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/scoring.py)
  - [`src/gemma4_capability_map/knowledge_work/exporters.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/exporters.py)
  - [`src/gemma4_capability_map/knowledge_work/replay.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/replay.py)
  - [`src/gemma4_capability_map/reporting/knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/reporting/knowledge_work_board.py)
- Fixed the rescore path in [`scripts/rescore_knowledge_work_runs.py`](/Users/cheickdiakite/Codex/moonie/scripts/rescore_knowledge_work_runs.py) so evaluation-only changes now recompute underlying stage task-trace metrics before rebuilding episode scorecards.
- Rebuilt board/history exports:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md)
- Verification:
  - focused metric/reporting/runtime slice: `64 passed`
  - full suite after the backfill: `230 passed`
- New finding:
  - the widened `29 / 23` headline rows still match on top-line readiness
  - they no longer look equivalent once controller help is measured
  - replayable `hf_gemma4_e2b_specialists_cpu` currently shows:
    - `controller_repair_avg = 1.8448`
    - `argument_repair_avg = 0.2069`
    - `controller_fallback_avg = 0.8966`
    - `intent_override_avg = 0.0862`
    - `raw_planning_clean_rate_avg = 0.5172`
  - replayable `mlx_qwen3_8b_reasoner_only` currently shows:
    - `controller_repair_avg = 0.0`
    - `argument_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `intent_override_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
- Interpretation:
  - the strong Gemma harness is currently closing the top-line gap, but not yet the raw tool-use cleanliness gap
  - this is a better and more publishable research result than a flat “Gemma equals Qwen” statement because it tells us exactly where to improve Gemma next

### Harder `v5` replayable smoke currently saturates, so the next discriminating move is metric-aware not blind widening

- Added a new harder wave on disk:
  - generated corpora now read `91 / 396 / 32 / 26`
  - new atomic tool/direction tasks:
    - `tool_020_exec_api_read_only_latest_action`
    - `tool_021_jobs_cli_patch_only_latest_email_fix`
    - `tool_022_finance_cli_diff_review_only_invoice_lock`
  - new atomic visual tasks:
    - `visual_027_dashboard_review_backlog_enablement_refinement`
    - `visual_028_live_dashboard_review_backlog_enablement_refinement`
    - `visual_029_form_latest_blocked_email_refinement`
    - `visual_030_live_form_latest_blocked_email_refinement`
  - new KWA episodes:
    - `kwa_exec_backlog_resume_hold_v5`
    - `kwa_jobs_email_block_resume_hold_v5`
    - `kwa_finance_diff_review_hold_v5`
    - `kwa_exec_live_backlog_resume_hold_v5`
    - `kwa_jobs_live_email_block_resume_hold_v5`
    - `kwa_finance_live_diff_review_hold_v5`
- Bounded replayable smoke results:
  - oracle:
    - [`results/knowledge_work/oracle_smoke_harder_wave_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/oracle_smoke_harder_wave_replayable_v1/summary.json)
  - Gemma specialists:
    - [`results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_harder_wave_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_harder_wave_replayable_v1/summary.json)
  - Qwen MLX:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_harder_wave_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_harder_wave_replayable_v1/summary.json)
- On this bounded 3-episode replayable slice, all three rows currently land the same summary and the same tool traces.
- Interpretation:
  - these new episodes are valid harder tasks
  - but under the current strong modular controller they are still measuring harness policy more than model separation
  - the next move should be either:
    - a more model-judgment-sensitive harder slice
    - or a Gemma-specific controller-cleanup pass followed by the widened `32 / 26` reruns

### Visual latest-filter fallback closes the current widened Qwen gap

- Added a narrow runtime fallback in [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py):
  - when a visual latest-filter task has the correct tool path
  - and the second-pass rescue still leaks a stale earlier fragment
  - and the final `read_region_text` output is clean
  - Moonie now uses that clean readback instead of preserving the stale answer phrasing
- Added the regression in [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py):
  - `test_visual_latest_readback_fallback_recovers_when_second_pass_still_leaks_stale_fragment`
- Verification:
  - focused smoke/planner slice: `49 passed`
  - full suite after the runtime fallback: `227 passed`
- Bounded validation:
  - live jobs replay:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_live_alignment_v4/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_live_alignment_v4/summary.json)
    - `recovered_execution_avg = 1.0`
  - replayable jobs replay:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_replayable_alignment_v4/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_replayable_alignment_v4/summary.json)
    - `recovered_execution_avg = 1.0`
- Full-lane refresh:
  - replayable:
    - [`results/knowledge_work_matrix/20260412T213721Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T213721Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json)
    - `runs = 29`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9774`
  - live:
    - [`results/knowledge_work_matrix/20260412T213438Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T213438Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json)
    - `runs = 23`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9798`
- Interpretation:
  - on the current widened `29 / 23` surface, `mlx_qwen3_8b_reasoner_only` now matches the widened oracle and Gemma specialist rows
  - this is not a “Gemma beats Qwen” result on the current surface
  - it is a stronger “harness/runtime/controller design materially changes local agent capability” result
  - the next discriminating move is to make the benchmark harder again and to widen comparator coverage beyond a single Qwen row

### Widened oracle and Qwen rows are now aligned on `29 / 23`, and the residual Qwen gap is concentrated

- Reran the widened oracle rows:
  - replayable:
    - [`results/knowledge_work_matrix/20260412T202500Z_knowledge_work_publishable_core/oracle_gemma4_e2b__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T202500Z_knowledge_work_publishable_core/oracle_gemma4_e2b__replayable_core/summary.json)
    - `runs = 29`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9774`
  - live:
    - [`results/knowledge_work_matrix/20260412T221500Z_knowledge_work_publishable_core/oracle_gemma4_e2b__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T221500Z_knowledge_work_publishable_core/oracle_gemma4_e2b__live_web_stress/summary.json)
    - `runs = 23`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9798`
- Reran the widened MLX Qwen rows:
  - replayable:
    - [`results/knowledge_work_matrix/20260412T202500Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T202500Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json)
    - `runs = 29`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 0.9655172413793104`
    - `real_world_readiness_avg = 0.9716551724137932`
  - live:
    - [`results/knowledge_work_matrix/20260412T221500Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T221500Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json)
    - `runs = 23`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 0.9565217391304348`
    - `real_world_readiness_avg = 0.9725521739130435`
- This makes the widened same-surface comparison explicit:
  - oracle is clean on both lanes
  - `hf_gemma4_e2b_specialists_cpu` is clean on both lanes
  - `mlx_qwen3_8b_reasoner_only` is now aligned on the same widened surface, but still trails the Gemma specialist stack on recovered execution and readiness
- The remaining widened-live Qwen misses are now concentrated, not broad:
  - `kwa_jobs_live_visual_latest_issue_hold_v3`
  - `kwa_jobs_live_phone_patch_resume_hold_v4`

### Planner/controller repair closed the API and CLI latest-direction seam

- Fixed the shared planner in [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - `api_fetch_record` now repairs toward the correct latest-issue jobs record type
  - `cli_search_logs` now infers the right log path and query for latest invoice-lock tasks
  - override logic now preserves these repaired args instead of leaving empty/generic placeholders
- Added targeted regressions in [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py):
  - `test_planner_repairs_form_issue_api_fetch_record_arguments`
  - `test_planner_repairs_cli_search_logs_arguments`
  - `test_planner_prefers_cli_search_logs_for_latest_invoice_lock_failure`
- Verification:
  - targeted planner slice: `34 passed`
  - full suite after the reruns and patch: `226 passed`
- Interpretation:
  - the new controller fix improved both oracle/Gemma/Qwen consistency on the widened live lane
  - the finance live resume/lock failure is now closed for Qwen
  - the remaining Qwen gap is no longer an API/CLI arg seam; it is now narrower direction-following and latest-issue preservation inside the jobs episodes

### Harnessability replayable full-lane rerun is now strict/recovered clean on `29`

- Fixed the new harnessability/controller failures in [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - added explicit CLI/API routing priority for `cli_apply_patch`, `api_fetch_record`, and `api_update_record`
  - added deterministic argument inference and repair for record ids, record types, invoice-lock updates, and phone-validation patches
  - added tool-name aliases so non-canonical record/patch names repair cleanly
- Added targeted planner regressions in [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py) and reran controller coverage plus the full suite:
  - `35 passed` targeted
  - `223 passed` full suite
- Cleared the three new replayable harnessability failures in a bounded smoke rerun:
  - [`results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_harnessability_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_harnessability_replayable_v1/summary.json)
  - all three now have `strict_interface_score = 1.0` and `recovered_execution_score = 1.0`
- Reran the full replayable widened lane for the headline Gemma specialist stack:
  - [`results/knowledge_work_matrix/20260412T190500Z_knowledge_work_full_lane_harnessability_core/hf_gemma4_e2b_specialists_cpu__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T190500Z_knowledge_work_full_lane_harnessability_core/hf_gemma4_e2b_specialists_cpu__replayable_core/summary.json)
  - `runs = 29`
  - `artifact_quality_avg = 0.9689793103448276`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9774`
- The new v4 replayable harnessability episodes are now controller-clean inside the full batch:
  - `kwa_exec_latest_action_resume_hold_v4`
  - `kwa_jobs_phone_patch_resume_hold_v4`
  - `kwa_finance_invoice_lock_direction_hold_v4`
- Rebuilt history exports:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_public_summary.json`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_public_summary.json)
- Interpretation:
  - Gemma specialist is now strict/recovered clean on the widened replayable `29`-episode surface
  - the older oracle/Qwen/live rows still sit on earlier reproduced surfaces and need matching reruns before a new parity claim is made
  - the right next step is to rerun oracle and `mlx_qwen3_8b_reasoner_only` on the widened `29 / 23` surface, then decide whether Gemma 4 `31B` `GGUF` / `llama.cpp` is the next posture row or whether live-lane parity should come first

### Harnessability and direction-following framing expanded to nine questions

- Reframed the repo around nine linked research questions instead of seven, with the new emphasis on harnessability and direction-following across `function_call`, CLI, and API surfaces.
- Kept the external-benchmark story clean: community signals and published tables are now treated as hypotheses until Moonie reproduces them locally.
- Refreshed the current generated-corpus accounting to `84 / 341 / 29 / 23`, while the publishable comparison board still stays on the `26 / 20` surface.
- Documented experimental Gemma 4 `31B` `GGUF` / `llama.cpp` runtime-posture support as implemented, but not yet reproduced locally because no local model/runtime is installed on this machine.
- No new benchmark row should be inferred from the widened slices or the experimental `31B` posture until those runs actually exist.

## 2026-04-11

### Visual rescue and planner fixes raised both Gemma and Qwen on the same full-lane surface

- Fixed the shared visual second-pass rescue gate in:
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/pipelines/base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py)
- The rescue path now triggers for visual tasks that still mention superseded earlier candidates after a correct refine chain, even when the answer still contains the nominal expected terms.
- Fixed the shared planner in [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - `latest approval-safe action` now correctly shadows the generic `action` filter
  - this removes the spurious extra `refine_selection(action)` after a successful `refine_selection(latest action)`
- Added regressions in:
  - [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)
  - [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py)
- Bounded validation now clears the previously failing jobs and finance slices for the patched systems:
  - Qwen jobs:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_visual_form_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_visual_form_hold_v2/summary.json)
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_visual_constraint_override_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_visual_constraint_override_v2/summary.json)
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_live_visual_latest_issue_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_live_visual_latest_issue_v2/summary.json)
  - shared finance slices:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_finance_visual_invoice_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_finance_visual_invoice_hold_v2/summary.json)
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_finance_live_visual_invoice_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_finance_live_visual_invoice_hold_v2/summary.json)
    - [`results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_finance_visual_invoice_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_finance_visual_invoice_hold_v2/summary.json)
    - [`results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_finance_live_visual_invoice_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_finance_live_visual_invoice_hold_v2/summary.json)
- Refreshed full-lane rows:
  - Qwen:
    - [`results/knowledge_work_matrix/20260412T022659Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T022659Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json)
    - `recovered_execution_avg = 0.9615384615384616`
    - `real_world_readiness_avg = 0.9716653846153847`
    - [`results/knowledge_work_matrix/20260412T022659Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T022659Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json)
    - `recovered_execution_avg = 0.975`
    - `real_world_readiness_avg = 0.976455`
  - Gemma specialist:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v4/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v4/summary.json)
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9780730769230769`
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v4/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v4/summary.json)
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9806250000000001`
- Interpretation:
  - the harness fixes improved both models on the same surface
  - Gemma specialist still remains ahead on the same board
  - that strengthens the claim that the real work here is full-stack local-agent improvement, not just model selection luck

### First reproduced Qwen full-lane row plus deterministic text-decode fixes

- Turned the Qwen comparator from “registry-ready” into a real same-surface board row.
- Fixed two benchmark-discipline bugs in [`src/gemma4_capability_map/models/gemma4_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/gemma4_runner.py):
  - direct HF text generation now forces `do_sample=False` instead of inheriting model-level sampling defaults
  - text chat-template calls now pass `enable_thinking=thinking`, which matters for Qwen3 because the model family can otherwise silently default to thinking-mode formatting in nominally non-thinking benchmark runs
- Added an Apple-Silicon-native Qwen path:
  - registered `Qwen/Qwen3-8B-MLX-4bit` in [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - added `QWEN3_8B_MLX_PATH` support in [`src/gemma4_capability_map/models/runtime_utils.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/runtime_utils.py)
  - added `mlx_qwen3_8b_reasoner_only` to [`configs/knowledge_work_matrix_experimental.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_experimental.yaml)
- Installed local checkpoints on this machine:
  - raw HF Qwen at `/Users/cheickdiakite/models/Qwen3-8B`
  - MLX Qwen at `/Users/cheickdiakite/models/Qwen3-8B-MLX-4bit`
- The direct-HF Qwen path is technically runnable after the decode fixes, but it remains too slow on this Apple M4 Pro to be the right primary comparison row.
- The MLX Qwen path is the correct local comparator posture here:
  - bounded replayable smoke:
    - [`results/knowledge_work/qwen3_8b_mlx_smoke_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/qwen3_8b_mlx_smoke_replayable_v1/summary.json)
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9794`
- Ran the full experimental `26 / 20` matrix for `mlx_qwen3_8b_reasoner_only`:
  - batch:
    - [`results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental)
  - replayable:
    - [`results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json)
    - `artifact_quality_avg = 0.9744807692307693`
    - `strict_interface_avg = 0.9711538461538461`
    - `recovered_execution_avg = 0.9230769230769231`
    - `real_world_readiness_avg = 0.96045`
  - live:
    - [`results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json)
    - `artifact_quality_avg = 0.9696049999999999`
    - `strict_interface_avg = 0.9625`
    - `recovered_execution_avg = 0.925`
    - `real_world_readiness_avg = 0.961875`
- Comparison read:
  - versus `hf_gemma4_e2b_reasoner_only`, MLX Qwen is stronger on strict-interface, recovered-execution, and readiness in both lanes
  - versus `hf_gemma4_e2b_specialists_cpu`, MLX Qwen matches artifact/browser/strict on the current board surface but still loses on recovered-execution and readiness
  - the extra misses are concentrated, not diffuse:
    - replayable:
      - `kwa_jobs_visual_constraint_override_hold_v2`
      - `kwa_jobs_visual_form_hold`
      - `kwa_finance_visual_invoice_revision_hold_v2`
      - `kwa_finance_visual_invoice_hold`
    - live:
      - `kwa_jobs_live_visual_latest_issue_hold_v3`
      - `kwa_finance_live_visual_invoice_revision_hold_v2`
      - `kwa_finance_live_visual_invoice_hold`
- Interpretation:
  - we now have the first honest same-surface Gemma-versus-non-Gemma comparison in the repo
  - it strengthens the claim that our Gemma specialist stack is not just better than bare Gemma reasoning, but still ahead of a real local Qwen baseline on the same harder surface
  - the next research move is not “run Qwen at all”; it is to understand the concentrated visual recovery gap and then decide the next comparator

### Qwen-ready comparator support plus stricter visual count scoring

- Added the first real non-Gemma comparator plumbing for local runs:
  - registered `Qwen/Qwen3-8B` in [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - added `hf_qwen3_8b_reasoner_only` to [`configs/knowledge_work_matrix_experimental.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_experimental.yaml)
  - added explicit `QWEN3_8B_PATH` support in [`src/gemma4_capability_map/models/runtime_utils.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/runtime_utils.py)
- Upgraded the HF reasoner path in [`src/gemma4_capability_map/models/gemma4_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/gemma4_runner.py):
  - Gemma multimodal models still use the processor/image-text path
  - text-only models like Qwen now use a tokenizer/chat-template path instead of assuming a multimodal processor
  - this closes the main architectural blocker for a real local Qwen run
- Tightened visual realism scoring in [`src/gemma4_capability_map/evals/visual_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/visual_eval.py):
  - count-heavy visual tasks no longer get full credit from a lucky final-answer number mention when the tool-side selection count is wrong
- Added targeted regressions in:
  - [`tests/test_gemma4_runner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_gemma4_runner.py)
  - [`tests/test_runtime_utils.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_utils.py)
  - [`tests/test_knowledge_work_matrix_script.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_matrix_script.py)
  - [`tests/test_run_knowledge_work_arena_script.py`](/Users/cheickdiakite/Codex/moonie/tests/test_run_knowledge_work_arena_script.py)
  - [`tests/test_visual_tool_orchestration.py`](/Users/cheickdiakite/Codex/moonie/tests/test_visual_tool_orchestration.py)
- Verification:
  - focused comparator/scoring slice: `46 passed`
  - full suite: `205 passed`
- Interpretation:
  - the repo is now Qwen-ready, but still not Qwen-complete
  - the next honest benchmark claim requires a real local Qwen checkpoint and a completed `26 / 20` board row
  - the visual benchmark got harder in the right way without changing the task surface

### External benchmark context layer with explicit claim boundaries

- Added a new published external benchmark registry in [`configs/external_benchmarks.yaml`](/Users/cheickdiakite/Codex/moonie/configs/external_benchmarks.yaml).
- Added external benchmark exports in [`src/gemma4_capability_map/reporting/knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/reporting/knowledge_work_board.py):
  - [`results/history/knowledge_work_external_benchmarks.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_external_benchmarks.csv)
  - [`results/history/knowledge_work_external_benchmark_summary.json`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_external_benchmark_summary.json)
- Added a dedicated `External Context` tab to the board in [`src/gemma4_capability_map/app/streamlit_app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/streamlit_app.py).
- Seeded the first official published rows with:
  - GPT-5.4 benchmarks from OpenAI
  - Gemini 3.1 Pro benchmarks from Google DeepMind
- The board now makes the provenance boundary explicit:
  - Moonie reproduced runs stay on the native leaderboard
  - published external scores are shown as context only
  - this avoids mixing our KWA results with vendor-reported public benchmark results as if they were same-harness rows
- Interpretation:
  - this strengthens the publishable story without weakening rigor
  - we can now say “we improved Gemma 4 on our own benchmark, and here is the broader public benchmark context”
  - we still cannot claim Gemma versus Qwen on the same surface until Qwen is actually run locally on the full `26 / 20` matrix
- Verification:
  - `tests/test_knowledge_work_board.py`: `13 passed`
  - full suite: `199 passed`

### Harder visual realism expansion plus non-Gemma comparator boundary

- Expanded the atomic visual-tool gold corpus in [`scripts/make_gold.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_gold.py):
  - total atomic tasks: `78`
  - total generated variants: `324`
  - total `visual_tool_orchestration` tasks: `26`
- Added new harder replayable/live realism tasks that specifically pressure multi-step visual state retention instead of one-shot OCR:
  - backlog -> enablement-ops refinement
  - latest-issue -> email refinement
- The first implementation exposed a real planner weakness rather than a bad test:
  - the controller skipped `refine_selection` and jumped straight from `extract_layout` to `read_region_text`
  - this produced stale or under-refined answers on the new dashboard/form tasks
- Fixed the planner in [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - added explicit visual-filter support for `backlog`, `enablement ops`, and `email`
  - preserved specificity so `support backlog` is not collapsed back to generic `backlog`
  - kept the existing stale-selection and final-readback behavior intact
- Comparator-readiness work also landed for future non-Gemma local systems:
  - [`src/gemma4_capability_map/models/runtime_utils.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/runtime_utils.py) now supports derived env vars like `LOCAL_MODEL_QWEN_QWEN3_32B_PATH`
  - [`scripts/preflight_backends.py`](/Users/cheickdiakite/Codex/moonie/scripts/preflight_backends.py) and [`scripts/probe_specialist_access.py`](/Users/cheickdiakite/Codex/moonie/scripts/probe_specialist_access.py) are now registry-driven rather than Gemma-hardcoded
- Reality check:
  - Qwen should be the first non-Gemma comparator
  - there is still no local Qwen checkpoint, env path, or full-lane Qwen run on this machine
  - therefore no honest Qwen row should be added yet
- Verification:
  - focused planner/schema/visual/runtime regressions: `55 passed`
  - full suite after the realism + comparator-readiness pass: `198 passed`
- Interpretation:
  - the benchmark is harder in a meaningful way
  - the new tasks surfaced controller weaknesses that are now fixed
  - the next publishable comparison step is still a real local Qwen run, not a placeholder registry row

### Publishable-default full-lane Gemma parity result

- Reran the publishable-default full-lane matrix for the direct in-process Gemma specialist stack:
  - `uv run python scripts/run_knowledge_work_matrix.py --system-id hf_gemma4_e2b_specialists_cpu`
- Batch:
  - [`results/knowledge_work_matrix/20260411T152330Z_knowledge_work_publishable_core`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260411T152330Z_knowledge_work_publishable_core)
- Result:
  - replayable:
    - `runs = 26`
    - `artifact_quality_avg = 0.9744807692307693`
    - `strict_interface_avg = 0.9711538461538461`
    - `recovered_execution_avg = 0.9615384615384616`
    - `real_world_readiness_avg = 0.9668576923076924`
  - live:
    - `runs = 20`
    - `artifact_quality_avg = 0.9696049999999999`
    - `strict_interface_avg = 0.9625`
    - `recovered_execution_avg = 0.95`
    - `real_world_readiness_avg = 0.966045`
- Board interpretation:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv) now shows the publishable-default direct in-process Gemma specialist row matching the oracle full-lane row on the same `26 / 20` surface
  - the reasoner-only Gemma control remains materially weaker on that same board surface
- Research interpretation:
  - this is the current strongest evidence that the repo’s controller/runtime/specialist-stack learnings made Gemma 4 better as a local full-stack agent on our own benchmark
  - this is a publishable Gemma-improvement claim
  - it is not yet a publishable Gemma-versus-Qwen claim because there is still no local Qwen profile or full-lane Qwen run in the repo
- Verification:
  - full suite after the rerun and history rebuild:
    - `194 passed`

## 2026-04-10

### Shared runtime + product-surface pass

- Formalized a shared local-agent substrate instead of leaving the repo as benchmark-only plumbing:
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/runtime/schemas.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/schemas.py)
  - [`src/gemma4_capability_map/runtime/workflows.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/workflows.py)
- Added packaged workflow configuration in [`configs/packaged_workflows.yaml`](/Users/cheickdiakite/Codex/moonie/configs/packaged_workflows.yaml) so benchmark-backed KWA episodes can be launched as reusable local workflows instead of only as benchmark runs.
- Added first-class local entrypoints:
  - CLI:
    - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
    - package script: `moonie-agent`
  - local API:
    - [`src/gemma4_capability_map/api/app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/api/app.py)
    - package script: `moonie-agent-api`
- Refactored the benchmark/runtime seam so [`BasePipeline.run()`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py) now delegates into shared runtime execution logic rather than keeping a separate benchmark-only path.
- Added transitional Streamlit product surfaces over the same runtime contract:
  - [`src/gemma4_capability_map/app/views/operator_console.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/views/operator_console.py)
  - [`src/gemma4_capability_map/app/views/mobile_companion.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/views/mobile_companion.py)
  - [`src/gemma4_capability_map/app/theme.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/theme.py)
  - [`src/gemma4_capability_map/app/view_models.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/view_models.py)
- Added runtime/API regression coverage:
  - [`tests/test_runtime_core.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_core.py)
  - [`tests/test_runtime_api.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_api.py)
- Verification:
  - focused benchmark/runtime suite: `52 passed`
  - full suite: `154 passed`
- Interpretation:
  - the repo now has a real shared substrate for both benchmark and product work
  - approval/hold state is now a first-class runtime concept, not just an episode-side score artifact
  - the next risk is no longer “do we have any usable product surface?” but “can we keep the product surfaces and benchmark semantics aligned as the system expands?”

### Direct-HF specialist full-lane refresh plus softer invoice memo fix

- Reran the full direct in-process HF specialist-backed exploratory references after the visual referent-repair planner hardening:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v3/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9976833333333334`
    - `browser_workflow_avg = 0.991025`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9589916666666666`
  - live:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v3/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9704444444444444`
- Interpretation:
  - the board-facing direct-HF specialist comparison rows now reflect the repaired visual execution path
  - the earlier full-lane `v1` strict/recovered losses on the invoice/form episodes were real controller problems, and they are now gone at the broad-lane level
- Then moved onto the softer invoice artifact/readiness gap instead of re-debugging execution:
  - traced the `artifact_quality_avg = 0.7692` miss to the generic memo path in [`runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/runner.py)
  - exact failing checks were:
    - required heading order
    - required `invoice lock` signal
    - native heading-order alignment
  - patched the generic memo generation path so `Brief` and `Stage Goal` lead the artifact instead of being appended after risks/recommendation/output
  - patched the memo review path so revised notes produce a real ordered revision with:
    - `invoice lock`
    - approval-hold preservation
    - review response
    - revision diff
- Added a targeted regression in [`tests/test_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_arena.py) to keep the finance visual note ordered and `invoice lock`-aware.
- Validated the softer memo fix in bounded direct-HF specialist reruns:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v3/summary.json)
    - `artifact_quality_avg = 1.0`
    - `real_world_readiness_avg = 0.9722`
  - live:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v3/summary.json)
    - `artifact_quality_avg = 1.0`
    - `real_world_readiness_avg = 0.9769`
- Rebuilt the history/board exports:
  - [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md)
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
- The board now points at the refreshed full-lane `v3` rows, so the invoice memo-quality gain is visible in the local comparison surface without needing another direct-HF specialist rerun.

### Bounded replayable/live reruns closed the visual invoice/form execution failures

- Started from the concentrated direct-HF specialist misses in:
  - `kwa_finance_visual_invoice_hold`
  - `kwa_finance_live_visual_invoice_hold`
  - `kwa_jobs_live_visual_form_hold`
- Compared current-contract episode specs against the older full-lane service-backed traces and found a benchmarking drift issue:
  - current invoice stage-2 episodes now point to `visual_015_slide_policy_revision_pressure` and `visual_018_live_slide_policy_revision_pressure`
  - older full-lane service-backed traces were still using the pre-visual stage-2 path, so those broad rows were not exact apples-to-apples controls for the current episode contract
- Hardened the planner/controller path in [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - stop defaulting `refine_selection` to fake `sel-001`
  - stop defaulting `read_region_text` to placeholder region ids
  - infer `slide callout` instead of collapsing slide-policy revision tasks back to `risk callout`
  - add semantic preconditions so `refine_selection` and `read_region_text` must bind to the latest valid visual selection context
  - force visual repair to use the latest logical `image_id` and region instead of stale asset-path or placeholder-shaped arguments
- Added targeted regressions in [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py):
  - reject `refine_selection` with no prior visual selection
  - repair `read_region_text` from the latest local refinement context
  - preserve `slide callout` targets for policy-revision tasks
- Verification after the planner hardening:
  - targeted planner/visual/KWA tests: `55 passed`
- Reran bounded controls sequentially to keep machine load safe after the earlier hard shutdown:
  - service-backed replayable invoice:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_replayable_v1/summary.json)
    - `artifact_quality_avg = 0.7692`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - service-backed live invoice:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_live_v1/summary.json)
    - `artifact_quality_avg = 0.7692`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - service-backed live jobs form:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_jobs_visual_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_jobs_visual_live_v1/summary.json)
    - `artifact_quality_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - direct-HF specialists replayable invoice:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v2/summary.json)
    - `artifact_quality_avg = 0.7692`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - direct-HF specialists live invoice:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v2/summary.json)
    - `artifact_quality_avg = 0.7692`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - direct-HF specialists live jobs form:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_jobs_visual_live_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_jobs_visual_live_v2/summary.json)
    - `artifact_quality_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
- Interpretation:
  - the remaining visual invoice/form bug family is closed as an execution/controller issue
  - the invoice pair still carries a softer `artifact_quality_avg = 0.7692`, but that weakness is shared by the service-backed control and therefore should now be treated as an artifact/readiness target rather than an in-process-only orchestration failure
  - the broad direct-HF specialist full-lane references are now known to be stale for those rows and should be refreshed with a future full `24 / 18` rerun when we want the board rows to reflect the fix

### Direct-HF specialist-backed full-lane comparison after the visual stale-selection planner fix

- Fixed the visual planner/controller follow-up path in [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - keep ordered pending visual filters instead of collapsing back to the earliest generic filter
  - continue `refine_selection` after a successful `refine_selection` when the user request still contains a more specific visual constraint
  - switch to `read_region_text` only after pending visual refinements are exhausted
- Added targeted regressions in [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py) for the real stale-selection recovery failure path:
  - malformed or empty parsed controller output after one successful visual refinement
  - post-final-refinement readback on the latest region
- Verified the narrow fix first:
  - targeted planner + visual tests: `25 passed`
  - refreshed bounded service-backed visual smokes recovered on the dashboard episode:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_replayable_refresh_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_replayable_refresh_v2/summary.json)
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_live_refresh_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_live_refresh_v2/summary.json)
  - refreshed bounded direct-HF specialist smokes now match that recovered behavior:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_replayable_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_replayable_v2/summary.json)
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_live_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_live_v2/summary.json)
- Ran the full direct in-process HF specialist-backed exploratory references on the full KWA surface:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v1/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9834`
    - `browser_workflow_avg = 0.9910`
    - `strict_interface_avg = 0.9844`
    - `recovered_execution_avg = 0.9792`
    - `real_world_readiness_avg = 0.9452`
  - live:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v1/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9779`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 0.9514`
    - `recovered_execution_avg = 0.9444`
    - `real_world_readiness_avg = 0.9460`
- Comparative interpretation:
  - adding real specialists materially improves the direct-HF path relative to `hf_gemma4_e2b_reasoner_only`
  - the recovery is real but incomplete; the direct-HF specialist-backed stack still trails the `hf_service` specialist-backed baseline on the full `24 / 18` surface
  - the dashboard stale-selection issue is no longer the blocker
  - the remaining concentrated misses are now:
    - replayable:
      - `kwa_finance_visual_invoice_hold`
    - live:
      - `kwa_finance_live_visual_invoice_hold`
      - `kwa_jobs_live_visual_form_hold`
- Rebuilt the history/board exports after the new comparative rows landed:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md)

### Direct-HF full-lane comparison versus the service-backed reasoner baseline

- Added new registry-backed local systems in [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml):
  - `hf_gemma4_e2b_reasoner_only`
  - `hf_gemma4_e2b_specialists_cpu`
- Extended KWA system inference in [`scripts/run_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_arena.py) so direct in-process HF reasoner-only and specialist-backed runs resolve to stable `system_id` values instead of anonymous exploratory directories.
- Added comparison regressions in:
  - [`tests/test_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_arena.py)
  - [`tests/test_knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_board.py)
- Ran the new direct in-process HF reasoner-only full-lane exploratory references:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_inprocess_reasoner_full_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_reasoner_full_replayable_v1/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9834`
    - `browser_workflow_avg = 0.9910`
    - `strict_interface_avg = 0.9531`
    - `recovered_execution_avg = 0.9375`
    - `real_world_readiness_avg = 0.9330`
  - live:
    - [`results/knowledge_work/model_backed_hf_inprocess_reasoner_full_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_reasoner_full_live_v1/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9779`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 0.9306`
    - `recovered_execution_avg = 0.9167`
    - `real_world_readiness_avg = 0.9379`
- Rebuilt history/board exports after the new system landed:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md)
- Comparative interpretation:
  - the new direct-HF system is materially weaker than the `hf_service_gemma4_reasoner_only` full-lane baseline on both replayable and live lanes
  - the weakness is not diffuse across the corpus
  - the misses concentrate in the visual KWA episodes:
    - replayable:
      - `kwa_exec_visual_dashboard_brief`
      - `kwa_jobs_visual_form_hold`
      - `kwa_finance_visual_invoice_hold`
    - live:
      - `kwa_exec_live_visual_dashboard_brief`
      - `kwa_jobs_live_visual_form_hold`
      - `kwa_finance_live_visual_invoice_hold`
  - that is a useful benchmark result because it isolates a deployment/runtime comparison:
    - same base model family
    - same benchmark surface
    - different execution path
    - different robustness on multimodal, referent-heavy KWA episodes
- Operational note:
  - the machine had a hard interruption during earlier probing, so the comparison wave was switched to a safer sequence:
    - stop duplicate reasoner services
    - validate a single replayable and a single live episode first
    - only then run the full `24 / 18` comparison
- Runtime environment note:
  - `mlx` is still blocked locally because `scripts/preflight_backends.py` reports `ModuleNotFoundError: mlx`
  - `google/gemma-4-E4B-it` remains probe-only locally on this Mac and should not be treated as the next full-lane comparison candidate

### Visual tool orchestration, visual KWA slices, and corrected mixed-pressure widening

- Added a new atomic benchmark family, `visual_tool_orchestration`, across:
  - [`schemas.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/schemas.py)
  - [`visual_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/visual_eval.py)
  - [`visual_executor.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/visual_executor.py)
  - [`registry.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/registry.py)
  - [`base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py)
- The new visual tool family now exposes:
  - `segment_entities(image_id, entity_query)`
  - `refine_selection(selection_id, filter_query)`
  - `extract_layout(image_id, target_query)`
  - `read_region_text(image_id, region_id)`
- Added a seeded visual executor for canonical scoring and a local visual executor path for live stress, behind the same tool contract so the benchmark measures orchestration instead of bespoke adapter logic.
- Added a new visual gold corpus in [`visual_tools.jsonl`](/Users/cheickdiakite/Codex/moonie/data/gold/visual_tools.jsonl) plus generated assets from [`make_visual_assets.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_visual_assets.py):
  - replayable visual tasks: `8`
  - live visual tasks: `4`
  - total atomic corpus now: `64` gold tasks, `282` explicit variants
- Canonical visual atomic lane references:
  - replayable:
    - [`results/visual_tool_orchestration/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/visual_tool_orchestration/replayable_core/summary.json)
    - `runs = 8`
    - `success_rate = 1.0`
    - `strict_interface_rate = 1.0`
    - `recovered_execution_rate = 1.0`
    - `real_world_readiness_avg = 1.0`
  - live:
    - [`results/visual_tool_orchestration/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/visual_tool_orchestration/live_web_stress/summary.json)
    - `runs = 4`
    - `success_rate = 1.0`
    - `strict_interface_rate = 1.0`
    - `recovered_execution_rate = 1.0`
- Added six job-shaped visual KWA episodes in [`make_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_knowledge_work_arena.py):
  - replayable:
    - `kwa_exec_visual_dashboard_brief`
    - `kwa_jobs_visual_form_hold`
    - `kwa_finance_visual_invoice_hold`
  - live:
    - `kwa_exec_live_visual_dashboard_brief`
    - `kwa_jobs_live_visual_form_hold`
    - `kwa_finance_live_visual_invoice_hold`
- Bounded oracle visual KWA slices are now available:
  - replayable:
    - [`results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json)
    - `artifact_quality_avg = 0.8932`
    - `browser_workflow_avg = 0.9782`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9141`
  - live:
    - [`results/knowledge_work/kwa_visual_live_oracle_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/kwa_visual_live_oracle_v1/summary.json)
    - `artifact_quality_avg = 0.8932`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9165`
- The first replayable specialist-backed visual KWA slice is now clean at:
  - [`results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json)
  - `artifact_quality_avg = 0.8932`
  - `browser_workflow_avg = 0.9782`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9141`
- The decisive fixes were benchmark-contract and plumbing fixes, not a deep model change:
  - the planner needed logical `image_id` hints instead of asset-path-only prompts
  - visual placeholder values like `$selection` and `$region` needed explicit repair instead of passing through as valid arguments
  - visual tasks needed the same answer-rescue path used by retrieval/full-stack tasks because terse outputs like `2 remain` were operationally correct but failed benchmark surface expectations
- The canonical oracle KWA lanes were then rerun on the expanded corpus:
  - replayable core:
    - [`results/knowledge_work/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_core/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9866`
    - `browser_workflow_avg = 0.9910`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9510`
  - live web stress:
    - [`results/knowledge_work/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_web_stress/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9822`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9630`
- Widened the fully specialist-backed mixed-pressure matrix again with the visual KWA additions and reran the corrected references:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9822`
    - `browser_workflow_avg = 0.9880`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9436`
  - live:
    - [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json)
    - `runs = 15`
    - `artifact_quality_avg = 0.9786`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9555`
- The earlier `visual_*_v1` mixed-pressure runs should be treated as diagnosis artifacts for image-id wiring and answer-surface rescue, not as the current model-backed references.
- Extended the board/reporting layer with richer cuts in:
  - [`knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/reporting/knowledge_work_board.py)
  - [`streamlit_app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/streamlit_app.py)
  - new exports:
    - [`results/history/knowledge_work_role_breakdown.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_role_breakdown.csv)
    - [`results/history/knowledge_work_category_breakdown.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_category_breakdown.csv)
    - [`results/history/knowledge_work_track_breakdown.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_track_breakdown.csv)
- The board now also exposes runtime-facing metadata from manifests when available:
  - `warmup_load_ms`
  - `last_request_elapsed_ms`
  - `requests_completed`
  - `total_cost_per_mtok`
- Example local row now visible in [`knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv):
  - `Gemma 4 E2B + FunctionGemma + EmbeddingGemma (HF local)`
  - `warmup_load_ms = 37909`
  - `last_request_elapsed_ms = 4402`
  - `requests_completed = 204`
- Interpretation:
  - there is now a clean atomic path to measure “one model reasons, one model sees/segments/extracts” behavior locally
  - the job-shaped visual episodes are strong enough to keep in the specialist-backed widening matrix, but they still surface softer artifact-quality gaps rather than strict interface failures
  - the benchmark is moving in the right direction: more realistic multimodal orchestration, still replayable, still separately scorable on strict vs recovered vs readiness
- Ran the broader full-lane specialist-backed exploratory references on the entire generated KWA corpus:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9866`
    - `browser_workflow_avg = 0.9910`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9510`
  - live:
    - [`results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9822`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9630`
- Interpretation:
  - the current local specialist-backed stack is now stable on the entire generated KWA corpus, not only on narrower mixed-pressure subsets
  - the remaining benchmark signal is increasingly about soft realism and comparative system evaluation, not keeping the current local stack alive through the existing corpus
- Normalized historical KWA comparison metadata in the board layer:
  - unknown or legacy slice-specific `system_id` values are now resolved back onto the registry-backed system identities
  - latest-board selection now prefers `full_lane` exploratory runs over narrower subsets, which fixes the public comparison surface for reasoner-only vs specialist-backed full-corpus baselines
- Tightened visual orchestration scoring and repair:
  - visual argument matching now requires the latest valid `selection_id` / `region_id` referent, not any non-empty placeholder replacement
  - planner repair for `refine_selection` and `read_region_text` now fills placeholder ids without overwriting valid user-intended filters like `support backlog`
  - the canonical visual-tool lanes were rerun after that hardening and are now clean again at:
    - [`results/visual_tool_orchestration/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/visual_tool_orchestration/replayable_core/summary.json)
    - `runs = 11`
    - `success_rate = 1.0`
    - [`results/visual_tool_orchestration/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/visual_tool_orchestration/live_web_stress/summary.json)
    - `runs = 7`
    - `success_rate = 1.0`

### Benchmark board and mixed-pressure specialist-backed widening

- Added a registry-backed KWA board/reporting layer:
  - registry:
    - [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - board/scatter exports:
    - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
    - [`results/history/knowledge_work_board_runs.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_runs.csv)
    - [`results/history/knowledge_work_scatter.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_scatter.csv)
    - [`results/history/knowledge_work_board.json`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board.json)
  - Streamlit surface:
    - [`streamlit_app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/streamlit_app.py) now exposes `knowledge_work_board`
- Added explicit `system_id` support to KWA runs and normalized board rows by:
  - `system_id`
  - `lane`
  - `run_intent`
- Widened the fully specialist-backed mixed-pressure replayable slice and corrected it after a real replayable-only refusal-versus-escalate miss:
  - initial replayable widening:
    - [`model_backed_hf_specialists_cross_role_hardmix_replayable_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_replayable_v1/summary.json)
    - this should now be treated as a diagnosis artifact, not the current reference
  - corrected replayable reference:
    - [`model_backed_hf_specialists_cross_role_hardmix_replayable_v2`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_replayable_v2/summary.json)
    - metrics:
      - `runs = 12`
      - `artifact_quality_avg = 1.0`
      - `browser_workflow_avg = 0.9914`
      - `strict_interface_avg = 1.0`
      - `recovered_execution_avg = 1.0`
      - `real_world_readiness_avg = 0.9222`
      - `escalation_correctness_avg = 1.0`
  - live mixed-pressure reference:
    - [`model_backed_hf_specialists_cross_role_hardmix_live_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_live_v1/summary.json)
    - metrics:
      - `runs = 12`
      - `artifact_quality_avg = 1.0`
      - `browser_workflow_avg = 1.0`
      - `strict_interface_avg = 1.0`
      - `recovered_execution_avg = 1.0`
      - `real_world_readiness_avg = 0.9691`
      - `escalation_correctness_avg = 1.0`
- The replayable billing miss was not solved by scoring relaxation. The decisive execution fix was stronger refusal-over-escalate guidance in [`base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py), then replayable `v2` matched the live mixed-pressure slice on strict and recovered execution.
- Added trace-backed KWA rescoring in:
  - [`scripts/rescore_knowledge_work_runs.py`](/Users/cheickdiakite/Codex/moonie/scripts/rescore_knowledge_work_runs.py)
- Used the rescoring path to harden memory-retention evaluation without rerunning models:
  - the scorer in [`scoring.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/scoring.py) now rewards preserved salient facts, not only verbatim stored strings
  - `kwa_finance_partner_deck_revision` now shows the more truthful split:
    - `revision_responsiveness = 0.0435`
    - `memory_retention_score = 1.0`
    - `role_readiness_score = 0.9114`
  - interpretation:
    - the episode still has a genuine revision-quality weakness
    - the old `memory_retention_score = 0.5` was too brittle and is no longer the right reading

### Harder human-nuance specialist-backed closure

- Ran the replayable harder human-nuance slice at [`results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json):
  - episodes:
    - `kwa_exec_stale_brief_hold`
    - `kwa_jobs_constraint_preservation_hold`
    - `kwa_finance_stale_assumption_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 0.9828`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9364`
    - `escalation_correctness_avg = 1.0`
- Ran the live harder human-nuance slice at [`results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json):
  - episodes:
    - `kwa_exec_live_stale_brief_hold`
    - `kwa_jobs_live_constraint_hold`
    - `kwa_finance_live_stale_assumption_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9383`
    - `escalation_correctness_avg = 1.0`
- Per-episode leaderboards are clean:
  - replayable: [`episode_leaderboard.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/episode_leaderboard.csv)
  - live: [`episode_leaderboard.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/episode_leaderboard.csv)
- Interpretation:
  - the specialist-backed stack now handles these harder human-style failure modes in bounded form, not just under oracle execution
  - the next benchmark question is composition: stale context + constraint pressure + approval gating + revision, not isolated handling of each one

### Harder human-nuance KWA oracle expansion

- Expanded `KnowledgeWorkArena` in [`make_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_knowledge_work_arena.py) from `18/12` to `21/15` episodes.
- Added new replayable episodes:
  - `kwa_exec_stale_brief_hold`
  - `kwa_jobs_constraint_preservation_hold`
  - `kwa_finance_stale_assumption_hold`
- Added new live episodes:
  - `kwa_exec_live_stale_brief_hold`
  - `kwa_jobs_live_constraint_hold`
  - `kwa_finance_live_stale_assumption_hold`
- These episodes explicitly test:
  - stale-context reconciliation
  - preserving the human’s original constraint under external pressure
  - removing stale financial assumptions before approval-gated release
- Exposed `forbidden_fragments` through the `_artifact(...)` helper so artifact contracts can fail stale or unsafe outputs directly instead of only rewarding the right fragments.
- Hardened [`scoring.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/scoring.py) so browser-workflow scoring now recognizes ordered branch structure:
  - `validation_failed -> recovered`
  - `recovered -> approval_required|blocked`
- Fixed a canonical-runner hygiene bug in [`run_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_arena.py):
  - `--limit` now defaults to `None`
  - canonical runs now execute the full lane unless a limit is explicitly requested
- Added regression coverage in:
  - [`test_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_arena.py)
  - [`test_schemas.py`](/Users/cheickdiakite/Codex/moonie/tests/test_schemas.py)
- Regenerated the KWA corpus:
  - `uv run python scripts/make_knowledge_work_arena.py`
- Refreshed canonical oracle lanes:
  - replayable core at [`results/knowledge_work/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_core/summary.json)
    - `runs = 21`
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 0.9929`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9333`
  - live web stress at [`results/knowledge_work/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_web_stress/summary.json)
    - `runs = 15`
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9630`
- Fixed one artifact-contract false negative during rollout:
  - the stale-assumption model contract originally required `hold`, which belongs in the memo/note artifact rather than the spreadsheet model
  - after removing that mismatch, both canonical lanes returned to `artifact_quality_avg = 1.0`
- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_knowledge_work_arena.py -q`
  - `21 passed`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
  - `120 passed`

### Broader live cross-role specialist-backed baseline closure

- Ran the matching broader live-web stress specialist-backed cross-role baseline at [`results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json).
- Episodes:
  - `kwa_exec_live_brief`
  - `kwa_exec_live_calendar_policy`
  - `kwa_exec_live_vendor_access_hold`
  - `kwa_jobs_live_requirements_extract`
  - `kwa_jobs_live_career_plan`
  - `kwa_jobs_live_submission_hold`
  - `kwa_finance_live_earnings_update`
  - `kwa_finance_live_comps_revision`
  - `kwa_finance_live_billing_patch_hold`
- Aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9794`
  - `escalation_correctness_avg = 1.0`
- The per-episode leaderboard at [`episode_leaderboard.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/episode_leaderboard.csv) is clean across all `9` episodes.
- Interpretation:
  - the controller/planner hardening from the replayable broad fix generalizes to the matching live-web stress slice
  - the current specialist-backed benchmark frontier moves away from this balanced cross-role subset and toward wider matrix volume and harder mixed-evidence / revision-heavy episodes

### Broader replayable cross-role specialist-backed baseline closure

- Patched [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py) so parallel audit tasks now:
  - enforce the full pending `inspect_image + read_repo_file` batch before accepting a partial model plan
  - block `propose_patch` until both successful evidence sources exist
  - infer patch arguments from combined successful feedback instead of trusting the latest single tool call
- Added focused regressions in [`test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py) for:
  - full-batch enforcement on the initial parallel audit turn
  - repo-read priority after only image feedback
  - canonical patch repair after both audit inputs exist
- Verification after the planner fix:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_tool_planner.py tests/test_smoke_eval.py tests/test_answer_match.py -q`
  - `40 passed`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
  - `117 passed`
- The broader replayable specialist-backed cross-role reference is now clean at [`results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json):
  - episodes:
    - `kwa_exec_board_prep_pack`
    - `kwa_exec_inbox_triage`
    - `kwa_exec_vendor_access_hold`
    - `kwa_jobs_tailored_packet`
    - `kwa_jobs_revise_after_feedback`
    - `kwa_jobs_submission_hold`
    - `kwa_finance_three_statement_model`
    - `kwa_finance_partner_deck_revision`
    - `kwa_finance_billing_patch_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 0.9939`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9175`
    - `escalation_correctness_avg = 1.0`
- The earlier `model_backed_hf_specialists_cross_role_broad_v1` miss should be treated as a diagnosis artifact for the parallel-audit controller leak, not as the current reference state.

### Judgment hardening and specialist-backed policy closure

- Added judgment-aware answer scoring in [`answer_match.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/answer_match.py):
  - explicit `action:` extraction for judgment-mode tasks
  - `expected_action + basis` scoring instead of pure fragment matching
  - backward-compatible fallback for older oracle outputs that still emit legacy fragment answers without an `action:` line
- Broadened operational semantic aliases in [`answer_match.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/answer_match.py) so policy-safety phrasing like `high-risk` and `safety control` is treated as evidence for `unsafe` on refusal tasks.
- Wired the judgment-aware scorer through all task evaluators:
  - [`agent_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/agent_eval.py)
  - [`tool_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/tool_eval.py)
  - [`retrieval_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/retrieval_eval.py)
  - [`thinking_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/thinking_eval.py)
- Updated real-world readiness derivation in [`real_world_metrics.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/real_world_metrics.py) so `escalation_readiness` uses explicit judgment correctness when present instead of inheriting generic answer-match behavior.
- Updated second-pass rescue adoption in [`base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py) so judgment-mode rescues are accepted when they satisfy the judgment contract, not only when they satisfy legacy `expected_answer_contains` fragments.
- Added and tightened regressions in:
  - [`test_answer_match.py`](/Users/cheickdiakite/Codex/moonie/tests/test_answer_match.py)
  - [`test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)

### Specialist-backed policy replayable status

- The replayable specialist-backed policy subset is now clean at:
  - [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json)
- Current replayable specialist-backed policy metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9818`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9363`
  - `escalation_correctness_avg = 1.0`
- Episode breakdown is now clean across all three replayable policy-hold episodes:
  - [`kwa_exec_vendor_access_hold`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/episode_leaderboard.csv)
  - [`kwa_jobs_screening_hold`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/episode_leaderboard.csv)
  - [`kwa_finance_billing_patch_hold`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/episode_leaderboard.csv)
- This now aligns the replayable specialist-backed policy subset with the already-clean live subset at [`results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json).

### Broader specialist-backed policy exploratory sweeps

- Added explicit `run_intent` handling in [`run_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_arena.py), with the default now inferred as:
  - `canonical` when writing to the latest lane pointer
  - `exploratory` when using `--no-update-latest`
- Updated [`build_knowledge_work_history.py`](/Users/cheickdiakite/Codex/moonie/scripts/build_knowledge_work_history.py) so generated history separates:
  - latest canonical by lane
  - latest exploratory by lane
  - best historical by lane
- Added history regressions in [`test_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_arena.py) for canonical vs exploratory inference and markdown report structure.
- Ran a broader replayable specialist-backed policy sweep at [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json):
  - episodes:
    - `kwa_exec_board_send_hold`
    - `kwa_exec_vendor_access_hold`
    - `kwa_jobs_submission_hold`
    - `kwa_jobs_screening_hold`
    - `kwa_finance_committee_hold`
    - `kwa_finance_billing_patch_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 0.9827`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9364`
    - `escalation_correctness_avg = 1.0`
- Ran the matching broader live specialist-backed policy sweep at [`results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json):
  - episodes:
    - `kwa_exec_live_send_hold`
    - `kwa_exec_live_vendor_access_hold`
    - `kwa_jobs_live_submission_hold`
    - `kwa_jobs_live_screening_hold`
    - `kwa_finance_live_committee_hold`
    - `kwa_finance_live_billing_patch_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9383`
    - `escalation_correctness_avg = 1.0`
- The one remaining live-policy miss before the final rerun was `kwa_exec_live_vendor_access_hold`, specifically the `agent_013_ambiguous_vendor_defer` stage answering `defer` for organizer approval instead of `clarify` for an ambiguous meeting target.
- The decisive fix was not scoring alone. In [`base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py), `clarify` now has explicit precedence over `defer` when the exact target is still not identifiable, even if approvals might also be needed later.
- With that precedence rule plus the earlier judgment-aware scorer, both broader specialist-backed policy sweeps are now clean.

### Verification

- Focused judgment/scoring regressions:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_answer_match.py tests/test_smoke_eval.py -q`
  - `26 passed`
- Wider judgment/KWA regression set:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_answer_match.py tests/test_smoke_eval.py tests/test_tool_planner.py tests/test_knowledge_work_arena.py tests/test_schemas.py -q`
  - `54 passed`
- Full suite:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
  - `111 passed`

## 2026-04-09

### Scope advanced

- Strengthened multilingual multimodal coverage by replacing benchmark-critical French prompt variants with exact translations in [`language.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/stressors/language.py).
- Hardened multilingual answer scoring in [`answer_match.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/answer_match.py) so accented French action phrases and weekday/time phrases map cleanly to benchmark expectations.
- Added two new screenshot-centric thinking tasks in [`make_gold.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_gold.py): `think_011_incident_screenshot_toggle` and `think_012_billing_invoice_lock`.
- Replaced heuristic-only specialist placeholders with real HF-capable `FunctionGemmaRunner` and `EmbeddingGemmaRetriever` implementations in [`functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/functiongemma_runner.py) and [`embeddinggemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/embeddinggemma_runner.py).
- Expanded the alpha corpus to `52` gold tasks and regenerated `232` explicit variants, keeping the benchmark balanced at `13` tasks per track.
- Added two new routing tasks, two new retrieval tasks, and two new retrieval-bearing full-stack tasks in [`make_gold.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_gold.py) so broader specialist-backed comparisons are possible without changing the benchmark contract.
- Added a first-class real-world autonomy layer:
  - task-level `benchmark_tags`
  - task-level `real_world_profile`
  - trace-level propagation of those fields
  - real-world metrics such as `state_integrity_score`, `collateral_damage_free`, `intervention_free_success`, and `real_world_readiness_score`
- Tagged `16` current tasks as real-world job-like probes across `release_ops`, `billing_ops`, `calendar_ops`, `incident_ops`, `finance_ops`, and `access_ops`.
- Added the first dedicated real-world matrix at [`alpha_real_world_matrix.yaml`](/Users/cheickdiakite/Codex/moonie/configs/alpha_real_world_matrix.yaml) plus design notes in [`real-world-benchmarking.md`](/Users/cheickdiakite/Codex/moonie/docs/real-world-benchmarking.md).
- The real-world matrix is now service-backed and canonical on this machine:
  - the previous in-process real-world run stalled during repeated HF warmup and should not be treated as authoritative
  - [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world) completed cleanly with subprocess-isolated experiments and `hf_service` as the Gemma 4 reasoner path
- `KnowledgeWorkArena` has now been hardened beyond simple markdown-contract episodes:
  - replayable-core grew first to `15` episodes and now to `18`
  - live-web stress grew first to `9` episodes and now to `12`
  - new partial-progress hold episodes now require the correct move to be `defer`, `escalate`, or `refuse to send` after useful work has already been completed
  - browser traces now capture validation rules, state updates, approval gates, blocked reasons, sandbox endpoints for dry-run submissions, and explicit state-machine transitions
  - finance and job artifacts now materialize as real `.xlsx`, `.pptx`, and `.docx` work products before grading
  - artifact graders now check formulas, deck section structure, revision diffs, application-packet consistency, workbook formula cells, document heading order, and slide-specific bullet expectations instead of only generic fragment presence
  - long `KnowledgeWorkArena` runs now checkpoint after each episode through `progress.json`, partial traces, and partial summaries instead of only writing at the end
  - the current canonical `KnowledgeWorkArena` summaries are [`results/knowledge_work/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_core/summary.json) and [`results/knowledge_work/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_web_stress/summary.json)
  - the canonical oracle lane summaries have now been refreshed on the expanded corpus:
    - replayable core: `18` runs, `artifact_quality = 1.0`, `browser_workflow = 0.9942`, `strict_interface = 1.0`, `recovered_execution = 1.0`, `real_world_readiness = 0.9327`
    - live-web stress: `12` runs, `artifact_quality = 1.0`, `browser_workflow = 1.0`, `strict_interface = 1.0`, `recovered_execution = 1.0`, `real_world_readiness = 0.9691`
  - two new bounded oracle policy-hardening snapshots now exist:
    - [`results/knowledge_work/replayable_policy_hardening_oracle/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_policy_hardening_oracle/summary.json)
    - [`results/knowledge_work/live_policy_hardening_oracle/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_policy_hardening_oracle/summary.json)
    - both kept `strict_interface = 1.0` and `recovered_execution = 1.0` while adding harder `validation_failed -> recovered -> approval_required|blocked` policy branches
  - the first real specialist-backed policy-hardening snapshots now also exist:
    - replayable:
      - [`results/knowledge_work/model_backed_hf_specialists_policy_replayable/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable/summary.json)
      - `artifact_quality = 1.0`
      - `browser_workflow = 0.9818`
      - `strict_interface = 1.0`
      - `recovered_execution = 0.8333`
      - `real_world_readiness = 0.8468`
      - `escalation_correctness = 0.5`
    - live:
      - [`results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json)
      - `artifact_quality = 1.0`
      - `browser_workflow = 1.0`
      - `strict_interface = 1.0`
      - `recovered_execution = 1.0`
      - `real_world_readiness = 0.9383`
      - `escalation_correctness = 1.0`
  - the replayable failure is concentrated rather than diffuse:
    - [`kwa_finance_billing_patch_hold`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable/episode_leaderboard.csv) is the dominant miss
    - the unsafe disable request was answered with `action: defer` instead of the benchmark-required `refuse`
    - that drove `recovered_execution = 0.5`, `escalation_correctness = 0.0`, and `collateral_damage_free = 0.5` for that episode even though `strict_interface` stayed `1.0`
  - the executive replayable miss is smaller but also judgment-shaped:
    - `kwa_exec_vendor_access_hold` remained interface-clean and state-clean, but its clarify/defer surface only reached `escalation_correctness = 0.5`
  - the first finished non-oracle episode baseline now exists at [`results/knowledge_work/model_backed_hf_exec_hold/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_exec_hold/summary.json)
  - a broader multi-episode HF reasoner pilot exists at [`results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json), currently stopped after two completed episodes so the repo has a clean finished model-backed baseline plus a separate partial pilot artifact
  - the next realism hardening pass is now in place:
    - real `.xlsx`, `.pptx`, and `.docx` artifacts are graded from the native files, not only from extracted markdown-like text
    - replayable and live hold episodes now include explicit `validation_failed -> recovered -> approval_required` branches instead of only linear happy-path holds
    - `KnowledgeWorkArena` runner/runtime now warms the real router and retriever, records specialist device configuration, and checkpoints warmup state before episode execution
  - the first bounded fully specialist-backed `KnowledgeWorkArena` run now exists at [`results/knowledge_work/model_backed_hf_specialists_finance/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_finance/summary.json)
    - configuration:
      - reasoner: `hf_service` on `google/gemma-4-E2B-it`
      - router: real HF `google/functiongemma-270m-it` on `cpu`
      - retriever: real HF `google/embeddinggemma-300m` on `cpu`
      - episode: `kwa_finance_three_statement_model`
    - result:
      - `artifact_quality_avg = 1.0`
      - `browser_workflow_avg = 1.0`
      - `strict_interface_avg = 0.7`
      - `recovered_execution_avg = 1.0`
      - `real_world_readiness_avg = 0.8574`
    - the relevant trace-level finding is that the full-stack stage completed correctly but still required controller repair on `agent_001_budget_compare`, leaving `tool_exact = 0.0`, `arg_exact = 0.0`, and `interface_reliability_score = 0.4`

### Backend findings

- Backend preflight now has a dedicated artifact at [`backend_preflight.json`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json) and [`backend_preflight.md`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.md).
- Backend posture is session-sensitive. Earlier preflight snapshots showed MLX crashing with a Metal initialization exception; the latest preflight at [`backend_preflight.json`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json) shows MLX healthy again and currently recommends `mlx` for the local default path.
- HF auth is present and detected via `HF_TOKEN`; local backend posture should now be decided from preflight rather than fixed prose.
- A reusable local `hf_service` reasoner path now exists so repeated benchmark runs can reuse a warmed HF model instead of paying full warmup on each experiment.
- Service observability is now first-class through `state.json`, `events.jsonl`, `service.log`, and `requests.jsonl` under [`results/runtime/hf_reasoner`](/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner).
- The first canonical real-world run justified `hf_service` as a research-execution primitive on this Mac even though preflight may still recommend `mlx` for the general local default. The issue was not inference speed alone; it was repeated HF warmup stability across a matrix.
- The reusable worker is not the current default on this Mac. Cold fresh-process HF startup can be dominated by import cost before model loading; the live worker probe showed `torch` import around `75.6s` and `torch + transformers` around `214.4s`, while the warmed standalone import probe later measured `1.9s` and `4.1s`. That gap is now a tracked observability variable instead of an assumption.
- The first finished model-backed `KnowledgeWorkArena` executive episode re-confirmed that cold service startup remains a first-order cost on this machine:
  - fresh `hf_service` boot to ready on `google/gemma-4-E2B-it` took about `345s`
  - once ready, the episode itself completed cleanly with `artifact_quality = 1.0`, `strict_interface = 1.0`, `recovered_execution = 1.0`, and `role_readiness = 0.9056`
- The first fully specialist-backed finance `KnowledgeWorkArena` pilot shows that specialist stabilization is materially better than before:
  - `hf_service` warmup to ready for the bounded finance pilot completed in about `37.9s`, recorded in [`results/knowledge_work/model_backed_hf_specialists_finance/manifest.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_finance/manifest.json)
  - real HF `FunctionGemma` and `EmbeddingGemma` both loaded successfully on `cpu` inside the same `KnowledgeWorkArena` run
  - the remaining weakness is now interface quality under full-stack composition, not specialist loading stability
- The stopped two-episode HF reasoner pilot also produced usable partial evidence:
  - completed `kwa_jobs_tailored_packet`
  - completed `kwa_finance_three_statement_model`
  - partial `role_readiness_avg = 0.9074`
- Backend preflight now marks dead service workers as `stale` instead of treating their last `state.json` as a live `loading` service.
- A dedicated import-timing artifact now exists at [`hf_import_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.json) and [`hf_import_probe.md`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.md).

### Benchmark posture

- Added a dedicated specialist probe matrix at [`alpha_specialist_matrix.yaml`](/Users/cheickdiakite/Codex/moonie/configs/alpha_specialist_matrix.yaml) for:
  - multilingual multimodal thinking verification
  - real EmbeddingGemma retrieval probes
  - real FunctionGemma routing probes
- The focused specialist rerun at [`20260409T120000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T120000Z_alpha_specialist_probe) completed on the real HF path with `4/4` success. That closes the last known French screenshot drift miss on the authoritative HF multimodal slice.
- The subsequent specialist replacement pass clarified the remaining blockers:
  - [`20260409T160700Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T160700Z_alpha_specialist_probe) showed `EmbeddingGemma` is blocked by Hugging Face manual gating, not by the local harness.
  - [`20260409T161500Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T161500Z_alpha_specialist_probe) showed the original FunctionGemma repo id was wrong in config, and after correcting it to `google/functiongemma-270m-it`, the real blocker is also Hugging Face manual gating.
- Specialist access is now recorded proactively in [`specialist_access_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.json) and [`specialist_access_probe.md`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.md) so these failures surface before a matrix run starts.
- Real specialist replacement is now materially validated rather than partially aspirational:
  - [`20260409T163000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T163000Z_alpha_specialist_probe) established that real `EmbeddingGemma` retrieval is strong (`1.0` success on the bounded retrieval slice), while real `FunctionGemma` routing was weaker under renamed-field schema drift (`0.75` success).
  - The first repair pass in [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py) fixed controller fallback so renamed schema keys are preserved when the router output collapses to pads.
  - The second repair pass in [`executor.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/executor.py) fixed the runtime adapter so renamed schema keys are translated back to canonical handler arguments without losing the original variant arguments in the trace.
  - Tool-track scoring in [`tool_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/tool_eval.py) now requires validator success, closing a benchmark-integrity gap where malformed executions could still appear as successful if tool choice and argument strings matched the gold event.
- The corrected specialist routing rerun at [`20260409T172000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T172000Z_alpha_specialist_probe) is now the authoritative FunctionGemma routing snapshot: `8/8` success, `recovery_correct = 1.0`, `malformed_call_rate = 0.0`, and no failing variants.
- The integrated clean matrix at [`20260409T180500Z_alpha_integrated_specialists`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T180500Z_alpha_integrated_specialists) is now the authoritative expanded clean snapshot:
  - local heuristic clean baselines remain strong on retrieval and full-stack (`mlx` retrieval `1.0`, `mlx` modular full-stack `1.0`)
  - the expanded clean thinking track is weaker than the earlier narrow slice (`mlx` thinking `0.8333`, `hf` thinking-off `0.9167`)
  - `hf` thinking-on materially regressed on the expanded image-heavy clean slice (`0.75`) with `thinking_overflow`, `generation_truncated`, and image-grounding misses
  - real `EmbeddingGemma` clean retrieval on the full `12`-task retrieval track remained `1.0`
  - real `FunctionGemma` clean routing on the full `12`-task routing track fell to `0.8333`, with both failures concentrated on direct patch-record intents: `tool_008_patch_record_clean` and `tool_012_billing_patch_record_clean`
  - real specialist-backed modular full-stack remained `1.0` on the narrow retrieval-bearing clean slice
- The specialist-backed drift matrix at [`20260409T190500Z_alpha_specialist_drift`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T190500Z_alpha_specialist_drift) is now the authoritative variant snapshot for real specialist lanes:
  - real `EmbeddingGemma` retrieval variants scored `0.9` task success while keeping `Recall@k = 1.0` and `evidence_hit_rate = 1.0`; both misses were answer-side, not retrieval-side
  - the failing retrieval variants were `retr_011_approval_policy_language_fr` and `retr_012_rollout_toggle_multimodal_context_long_history`
  - real `FunctionGemma` routing variants scored `0.75`; every failure came from `tool_012_billing_patch_record` across clean, code-switched, schema-renamed, and context-noised variants, all with the same `wrong_tool` + `arg_mismatch` signature
  - real specialist-backed modular full-stack variants scored `0.9375`; the only user-visible failing variant was `agent_011_runbook_guided_patch_language_fr` with `answer_mismatch`, while strict interface metrics remained lower because two tool-selection mismatches were recovered downstream
- The first canonical real-world autonomy snapshot at [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world) established a sharper boundary between bounded task execution and true job-like autonomy:
  - `hf_e2b_real_world_thinking_variants`: `0.0` success, `0.0` readiness
  - `hf_e2b_real_world_retrieval_variants`: `0.875` success, `1.0` strict interface, `0.88125` readiness
  - `hf_e2b_real_world_routing_variants`: `0.5` success, `0.5` strict interface, `0.6375` readiness
  - `hf_e2b_real_world_full_stack_variants`: `0.75` strict success, `1.0` recovered execution, `0.8696` readiness
  - the biggest real-world failure families are now explicit:
    - no-tool escalation judgment on `think_013_prod_approval_escalation`
    - French answer-surface misses on retrieval and full-stack variants
    - billing-patch and unsafe-billing-disable routing/refusal failures under real FunctionGemma routing

### Architecture findings

- Real retrieval is stronger than real routing under drift on the current benchmark. `EmbeddingGemma` kept perfect retrieval evidence metrics under variants, while `FunctionGemma` exposed a stable intent-class weakness around direct patch-record requests.
- The current bottleneck for retrieval-backed tasks is answer synthesis, not document finding. This is the strongest current argument for splitting retrieval quality claims from final-answer quality claims in published reporting.
- HF thinking-on is not currently the default reasoning path for this machine or benchmark slice. On the expanded clean thinking track, it is slower and less reliable than thinking-off because of overflow and truncation behavior.
- Patch-oriented routing needs a stronger intent prior than the current specialist stack provides. The repeated `tool_012_billing_patch_record` failures suggest a real router-side ambiguity between "inspect/lookup" and "propose/update patch record" tool classes.
- Real specialist-backed modular full-stack is operationally viable. Even under drift, it stayed above `0.93` success on the current narrow slice, which is enough to justify scaling that lane rather than treating it as experimental-only.
- In `KnowledgeWorkArena`, bounded specialist-backed execution is now good enough to expose the next real problem:
  - native artifacts can score perfectly
  - recovered execution can stay perfect
  - browser workflow can stay near-perfect even with explicit recovery branches
  - strict interface can now be pulled back to `1.0` when the controller is taught to respect next-step tool feedback instead of re-normalizing malformed outputs into repeated prior actions
  - that makes the next benchmark frontier less about obvious controller repetition and more about harder multi-step policy and judgment failures
- The history layer now needs stricter canonical semantics:
  - exploratory policy-only runs can share the same lane label as canonical runs
  - “latest by lane” is therefore not always the authoritative published state
  - the canonical summary pointers in continuity/docs are now the source of truth until the history report is split into canonical vs exploratory views
- Real-world autonomy is materially weaker than bounded task execution.
  - The model can preserve state and complete many operational tasks once tools are in motion.
  - It is still weak at deciding when not to act, when to escalate, and when a high-cost billing intent should be refused or redirected.
- Answer-surface multilinguality is still an end-to-end bottleneck.
  - Retrieval evidence and final state can both be correct while the real-world benchmark still fails because the answer layer misses the required French action phrasing.
- In `KnowledgeWorkArena`, the next specialist-backed weakness is now clearly action-class selection under replayable policy pressure:
  - the same fully specialist-backed stack can stay perfect on the live policy subset while missing replayable refusal/clarify expectations
  - this is evidence that the current problem is not broad runtime instability
  - it is likely a prompt/action-contract issue around `refuse` vs `defer` vs `clarify` after partial progress has already occurred
- H1 visual-sequence canaries clarified the current Gemma harnessing bottleneck:
  - concrete FunctionGemma prompt hints made the H1 HF service baseline controller-clean on the full five-episode slice: `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, `raw_planning_clean_rate_avg = 1.0`
  - the remaining failures moved to disabled-helper rows, especially visual refinement chains that repeat valid but stale filters and fail to reach `read_region_text`
  - [`20260506T_h1_visual_sequence_hint_canary_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_sequence_hint_canary_v1_knowledge_work_ablation_packet) showed base specialists stay clean on the three visual H1 episodes while both `no_controller_repair` and `no_deterministic_visual_follow_on` drop to `real_world_readiness_avg = 0.8837333333333333`
  - [`20260506T_h1_visual_filter_repair_canary_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_filter_repair_canary_v1_knowledge_work_ablation_packet) restored the `no_deterministic_visual_follow_on` mini-row to `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, and `failure_candidate_count = 0`
  - the causal bug was accepting valid `refine_selection` calls whose `filter_query` repeated an already-completed visual filter; the controller now treats the pending visual filter as a semantic argument precondition
  - this does not solve `no_controller_repair`: that row still measures real model-side dependence because disabled repair accepts the valid-but-wrong refinements as-is
- The full H1 visual-filter-repair ablation converted the canary result into the current authoritative H1 controller snapshot:
  - [`20260506T_h1_hf_service_visual_filter_repair_ablation_v1`](../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet) completed `7` systems across `35` replayable H1 episode rows
  - baseline specialists stayed clean: `real_world_readiness_avg = 0.9749800000000001`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, `raw_planning_clean_rate_avg = 1.0`
  - `no_deterministic_visual_follow_on` returned to baseline top-line readiness after the pending-filter repair, but remained controller-heavy: `controller_repair_avg = 0.6`, `argument_repair_avg = 0.3`, `controller_fallback_avg = 0.1`, `raw_planning_clean_rate_avg = 0.845`
  - all rows except `no_controller_repair` now match baseline on readiness, strict interface, and recovered execution
  - `no_controller_repair` remains the only top-line causal helper: `real_world_readiness_avg = 0.8874599999999999`, `strict_interface_avg = 0.775`, `recovered_execution_avg = 0.7`, while raw syntax is mostly clean at `0.975`
  - trace mining found only `3` failure candidates, all in `no_controller_repair`, with residual modes `repair_disabled`, `visual_readback_missing`, `visual_repeated_refinement`, and `visual_stepwise_control`
  - the next useful target is therefore model-side or contract-side visual sequence semantics, not visual rescue, generic fallback, or placeholder repair
- The compact H1 visual-semantics packet makes that residual controller-dependence cheap to replay:
  - [`20260506T_h1_visual_semantics_no_repair_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_no_repair_v1_knowledge_work_ablation_packet) ran `3` systems over the `3` residual visual episodes
  - baseline specialists stayed clean on the packet: `real_world_readiness_avg = 0.9715666666666666`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `raw_planning_clean_rate_avg = 1.0`
  - `no_controller_repair` reproduced the gap more sharply: `real_world_readiness_avg = 0.8257`, `strict_interface_avg = 0.625`, `recovered_execution_avg = 0.5`, while `raw_planning_clean_rate_avg = 1.0`
  - `no_deterministic_visual_follow_on` still recovered fully but needed help: `controller_repair_avg = 0.8333333333333334`, `argument_repair_avg = 0.5`, `raw_planning_clean_rate_avg = 0.7833333333333333`
  - trace mining found `3` failure candidates, all in `no_controller_repair`, with `visual_readback_missing`, `visual_repeated_refinement`, and `visual_stepwise_control` concentrated on the executive backlog and jobs form visual chains
  - this packet should be the default next loop for candidate visual sequencing fixes before another full H1 replayable rerun
- A stronger FunctionGemma visual system prompt was a negative top-line result on that compact packet:
  - [`20260506T_h1_visual_semantics_prompt_contract_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_prompt_contract_v1_knowledge_work_ablation_packet) kept baseline specialists clean and preserved full recovery for `no_deterministic_visual_follow_on`
  - `no_controller_repair` stayed unchanged at `real_world_readiness_avg = 0.8257`, `strict_interface_avg = 0.625`, `recovered_execution_avg = 0.5`, and `raw_planning_clean_rate_avg = 1.0`
  - trace mining again found `3` failure candidates, all in `no_controller_repair`
  - the candidate did reduce one comparison-row argument-repair note, but did not teach the raw disabled-repair row to stop replaying stale visual calls
  - next hypothesis: the exact next-call directive needs to be injected as a final turn-level router instruction after tool-result messages, not only as system-prompt wording
- The final turn-level FunctionGemma visual directive validated that hypothesis on the compact packet:
  - [`20260506T_h1_visual_semantics_turn_directive_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_turn_directive_v1_knowledge_work_ablation_packet) restored all three packet rows to `real_world_readiness_avg = 0.9715666666666666`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, and `raw_planning_clean_rate_avg = 1.0`
  - `no_controller_repair` no longer needed semantic repair on this packet; the raw FunctionGemma calls followed the full visual chains unaided
  - `no_deterministic_visual_follow_on` also became controller-clean on this packet: `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`
  - trace mining found `0` failure candidates; the only note family was the expected `controller_repair_disabled` ablation marker
  - this separates two prompt-shaping facts: generic system prose did not fix stale visual replay, but a final recency-weighted exact-call directive did
  - next required check is the full H1 replayable ablation after this directive
- The full H1 replayable ablation confirmed the turn-directive gain across all current H1 rows:
  - [`20260506T_h1_hf_service_turn_directive_ablation_v1`](../results/knowledge_work_h1_slice/20260506T_h1_hf_service_turn_directive_ablation_v1_knowledge_work_ablation_packet) completed `7` systems across `35` H1 episode rows
  - every row matched baseline: `real_world_readiness_avg = 0.9749800000000001`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, `raw_planning_clean_rate_avg = 1.0`
  - `no_controller_repair` moved from the prior full-H1 `0.8874599999999999` readiness to baseline, eliminating the last top-line causal helper on this slice
  - `no_deterministic_visual_follow_on` also became controller-clean, so the final router directive now subsumes the earlier deterministic visual follow-on benefit on H1
  - trace mining found `0` failure candidates; only ablation-marker notes remained
  - research interpretation: the H1 causal controller signal was real, but it pointed to a prompt-contract recency problem in FunctionGemma routing rather than an unavoidable need for controller repair
  - next benchmark need: define H1b with harder visual/API/approval interactions because current H1 is saturated again
- H1b is now scaffolded as the next saturation breaker:
  - [`configs/knowledge_work_h1b_slice.yaml`](../configs/knowledge_work_h1b_slice.yaml) reuses existing packaged workflow episode pairs that were outside H1
  - [`docs/continuity/h1b-slice.md`](continuity/h1b-slice.md) records the purpose, episode set, stressors, and commands
  - the new slice concentrates older but harder visual/revision/resume cases: dashboard referent carryover, latest-action resume, jobs constraint override, jobs phone patch resume, and finance invoice revision
  - H1b should be run first as a compact `visual_policy_no_controller_repair` packet before a full H1b ablation
- The first compact H1b visual-policy packet stayed controller-clean:
  - [`20260506T_h1b_visual_policy_packet_v1`](../results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet) ran baseline, `no_controller_repair`, and `no_deterministic_visual_follow_on` over `3` H1b episodes
  - all three rows matched at `real_world_readiness_avg = 0.9472999999999999`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace mining found `0` failure candidates
  - interpretation: these H1b episodes are harsher on artifact/readiness level than H1, but they still do not restore visual controller dependence after the final FunctionGemma turn directive
  - next check is a full H1b ablation across all five episodes and seven rows
- The full H1b ablation confirmed that the follow-up slice is also saturated with respect to the current controller-helper ablations:
  - [`20260506T_h1b_hf_service_ablation_v1`](../results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet) ran `7` systems across the `5` H1b replayable episodes
  - all seven rows matched at `real_world_readiness_avg = 0.9581199999999999`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace mining found `30` note events and `0` failure candidates
  - remaining note counts are only disabled-helper markers: `controller_repair_disabled = 22` and `intent_priority_disabled = 8`
  - interpretation: H1b lowers absolute readiness relative to H1 because its artifacts are harder, but it does not restore controller dependence after the final FunctionGemma turn directive. The next useful research move is live CLI validation plus a new H1c slice with genuinely new visual/API/approval interactions.
- H1c live-policy pressure is now benchmark-clean for both HF service specialists and local MLX Gemma:
  - [`20260506T_h1c_live_policy_packet_v1`](../results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet) ran the compact live-policy helper packet across baseline HF service specialists, `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`
  - all four HF service rows matched at `real_world_readiness_avg = 0.9779666666666667`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace mining found `14` note events and `0` failure candidates; the notes were only disabled-helper markers
  - [`20260506T_h1c_mlx_live_primary_v1`](../results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_v1_knowledge_work_h1c_live_policy_controller_dependence_v1) then ran all five H1c live episodes on `mlx_gemma4_e2b_reasoner_only`
  - the MLX primary row reached `real_world_readiness_avg = 0.97936`, `artifact_quality_avg = 0.95`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - direct trace inspection found no non-empty MLX `planning_repair_notes`
  - interpretation: H1c did not re-break top-line or controller-clean execution in the benchmark runner. The live CLI smoke packets remain important because they showed local MLX repair/fallback on overlapping workflows; the next research question is repeatability of that CLI/runtime-path signal, not another broad H1c rerun.
- H1d/H1e and the MLX directive probe clarified the current local Gemma harnessing boundary:
  - the final tool-turn directive was ported into the MLX/HF/GGUF Gemma runner prompt path
  - H1d directive-v2 eliminated the compact controller-stress failures: all four rows reached readiness `0.97936`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, and repair/fallback/argument repair `0.0`
  - H1e then expanded to all ten packaged live workflow families and still found `0` failure candidates; all four MLX rows matched at readiness `0.96891`
  - the directive probe stayed at exact JSON `7 / 8` because MLX paraphrases one visual selector from `"validation error"` to `"phone issue"`
  - executor-level visual aliasing makes that paraphrase executable: the latest probe records exact `7 / 8`, executable visual target `1 / 1`
  - interpretation: exact-copy cleanliness and executable live readiness now need to be tracked separately for visual selector text
- H1f reopened controller-dependence by removing the tool-turn directive:
  - [`20260506T_h1f_mlx_no_directive_v1`](../results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1) ran five compact live workflow families across contracted MLX, no-directive MLX, and three no-directive/no-helper variants
  - contracted MLX was clean: readiness `0.97936`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`
  - no-directive MLX preserved top-line readiness only by using controller help: repair/fallback/argument repair `0.70 / 0.20 / 0.50`, raw clean `0.30`
  - no-directive + no controller repair dropped to readiness `0.73818`, strict/recovered `0.475 / 0.300`
  - no-directive + no controller fallback dropped to readiness `0.92104`
  - no-directive + no argument repair dropped to readiness `0.82036`
  - interpretation: the directive is a causal harness intervention, but controller repair/fallback/argument repair remain causal once that prompt contract is absent
- H1g is the current negative result for remaining helper families under the directive:
  - [`20260506T_h1g_mlx_remaining_helpers_v1`](../results/knowledge_work_h1_slice/20260506T_h1g_mlx_remaining_helpers_v1_knowledge_work_h1g_mlx_remaining_helper_ablation_v1) ran baseline, `no_visual_rescue`, `no_intent_priority`, and `no_deterministic_visual_follow_on`
  - all four rows matched at readiness `0.97936`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - trace mining found `0` failure candidates
  - interpretation: under the current directive, visual rescue, intent priority, and deterministic visual follow-on are not carrying the compact live MLX slice; the useful expansion was therefore the full-H1e no-directive packet, not more tuning of those helpers
- H1h confirms the no-directive causal ordering across the full ten-workflow live MLX set:
  - [`20260507T_h1h_mlx_full_no_directive_v1`](../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1) ran contracted MLX, no-directive MLX, and the three no-directive/no-helper variants over all ten H1e live workflow families
  - contracted MLX and no-directive MLX both reached readiness `0.96891`, but the no-directive row needed controller repair/fallback/argument repair `0.70 / 0.25 / 0.45` and raw clean stayed `0.30`
  - no-directive + no controller repair dropped to readiness `0.73801`, strict/recovered `0.481 / 0.300`
  - no-directive + no controller fallback dropped to `0.89598`
  - no-directive + no argument repair dropped to `0.83016`
  - the H1h/H1f comparison shows no new causal ordering; the larger workflow set mostly adds more instances of the same failure families: fallback planner, visual stepwise control, repair disabled, fallback disabled, argument repair, visual repeated refinement, and visual readback missing
  - workflow-family attribution now makes the next target concrete: executive latest-action resume, jobs phone patch resume, jobs visual form hold, and executive stale brief packet are the worst no-repair rows
  - next empirical move: run an attributable Gemini CLI baseline packet over the same H1h workflow family set, then derive a smaller no-directive stress packet from the worst H1h workflows
- The H1h Gemini CLI dry-run baseline is now recorded as an external-reference packet:
  - [`20260507T_h1h_gemini_cli_dry_run_baseline_v1`](../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1) contains ten prompt/command artifacts, one per H1h workflow family
  - the run intentionally used a missing binary, so it recorded safe dry-run prompts without executing Gemini CLI or making external side effects
  - this is useful as a baseline interface contract: future real Gemini CLI execution can be compared against the exact same workflow-family prompts
- The no-directive MLX tool probe gives a raw-output explanation for H1h controller burden:
  - [`20260507T_mlx_no_directive_probe_v1`](../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1) ran the eight exact-call probe cases with `disable_tool_turn_directive`
  - exact match fell from contracted MLX `7 / 8` to no-directive `0 / 8`
  - the visual executable target fell from `1 / 1` to `0 / 1`
  - [`probe_case_deltas.csv`](../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1/probe_case_deltas.csv) shows CLI/API cases still often choose the right tool but drift on canonical arguments, while visual referent and parallel cases collapse to no tool call
  - interpretation: the directive is not cosmetic; it is the main model-side contract that keeps local MLX Gemma inside Moonie's tool interface. When it is absent, H1h readiness parity comes from controller repair/fallback rather than raw model compliance
- H1i turns the H1h worst-family attribution into a faster MLX prompt-contract loop:
  - [`configs/knowledge_work_h1i_slice.yaml`](../configs/knowledge_work_h1i_slice.yaml) keeps executive latest-action resume, jobs phone patch resume, jobs visual form hold, and executive stale brief packet
  - [`20260507T_h1i_mlx_worst_no_directive_v1`](../results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1) ran the same five MLX tool-contract rows over those four live workflow families
  - contracted MLX stayed clean at readiness `0.97710`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - no-directive with helpers stayed top-line clean at readiness `0.97710`, but controller repair/fallback/argument repair rose to `1.00 / 0.50 / 0.50` and raw clean fell to `0.00`
  - no-directive + no controller repair fell to readiness `0.64697`, strict/recovered `0.297 / 0.000`
  - no-directive + no controller fallback fell to readiness `0.83125`; no-directive + no argument repair fell to `0.81220`
  - interpretation: H1i is now the best fast packet for candidate prompt contracts. It is smaller than H1h, but it amplifies the same causal ordering and should be used before spending another full H1h run
- The MLX tool-contract report is now a reproducible research artifact rather than only continuity prose:
  - human report: [`docs/reports/mlx-tool-contract-harnessing.md`](reports/mlx-tool-contract-harnessing.md)
  - generated report: [`results/reports/mlx_tool_contract_harnessing/report.md`](../results/reports/mlx_tool_contract_harnessing/report.md)
  - generator: [`scripts/build_mlx_tool_contract_report.py`](../scripts/build_mlx_tool_contract_report.py)
  - test: [`tests/test_mlx_tool_contract_report.py`](../tests/test_mlx_tool_contract_report.py)
  - it synthesizes H1f, H1h, H1i, the contracted/no-directive probe comparison, and the H1h Gemini CLI dry-run baseline
  - the report figures now make the active mechanism visible:
    - H1i top-line readiness can hide interface and recovery collapse when repair is disabled
    - H1i amplifies the H1h no-directive controller burden
    - the no-directive probe drops exact-call compliance from `7 / 8` to `0 / 8`
    - H1i failure modes remain concentrated in fallback planner, visual stepwise control, argument repair, fallback disabled, and repair disabled
  - this should be regenerated after any new H1i, H1h, probe, or Gemini baseline packet
- The next MLX harnessing wave is now concretely queued as prompt-contract candidates rather than an open-ended prompt-tuning idea:
  - candidate systems are defined in [`configs/model_registry.yaml`](../configs/model_registry.yaml):
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required`
  - each candidate keeps `disable_tool_turn_directive = true` and adds a generic `tool_prompt_contract_id`, which means the candidate is testing interface-shaping language rather than reintroducing the exact final directive by another name
  - generated report artifacts now include [`prompt_contract_candidates.csv`](../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv) and [`prompt_contract_candidate_targets.svg`](../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_candidate_targets.svg)
  - dry-run probe packet [`20260507T_prompt_contract_candidates_dry_run_v2`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_dry_run_v2) freezes the three candidate probe commands without running MLX and records both contracted and no-directive probe baselines
  - H1i now has a named graduation packet, `mlx_prompt_contract_candidates`, that runs contracted MLX, no-directive MLX, and the three candidate rows across the same four worst-family live workflows
  - empirical gate: execute the probe packet first, compare exact/executable rates against contracted and no-directive baselines, regenerate the report, and only then spend H1i live runtime on candidates that improve raw protocol behavior
- The first executed prompt-contract probe gate is a partial-gain result, not a solved interface result:
  - [`20260507T_prompt_contract_candidates_execute_v1`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1) ran all three candidate rows against the eight-case probe and wrote contracted plus no-directive comparisons for each candidate
  - [`candidate_gate_summary.md`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1/candidate_gate_summary.md) is the compact candidate read
  - `schema_anchor_v1` recovered one exact case over no-directive (`exact_match_rate = 0.125`, `delta_exact_vs_no_directive = 0.125`) but remained far below contracted MLX (`delta_exact_vs_contracted = -0.75`)
  - `literal_argument_guard_v1` and `tool_required_parallel_v1` recovered the executable visual target (`executable_match_rate = 1.0`) but did not improve exact JSON copy (`exact_match_rate = 0.0`)
  - `tool_required_parallel_v1` remains dominated by `no_tool_call` (`6` cases), which means its current wording is not yet solving the failure family it was meant to target
  - interpretation: these candidates can be tried on H1i as mechanism probes, but the second prompt-contract wave should combine schema anchoring with visual executable recovery and explicitly reduce no-call failures
- The first H1i prompt-contract candidate packet saturated and therefore did not validate the probe gains as live improvements:
  - [`20260507T_h1i_prompt_contract_candidates_v1`](../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet) ran contracted MLX, no-directive MLX, and all three prompt-contract candidates over the four H1i worst-family live workflows
  - all five rows matched at `real_world_readiness_avg = 0.97710`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace analysis found `0` note events and `0` failure candidates
  - the analyzer now derives `tool_turn_directive_enabled` from row research controls rather than the shared warmed bundle snapshot, so disabled-directive rows are attributed correctly
  - interpretation: the probe remains the stronger discriminator; H1i candidate v1 is saturated and the next second-stage slice needs repeated no-directive trials or probe-derived live cases where visual/parallel no-call failures are stable
- H1/H1i ablation packet runners now support repeated episode execution:
  - `scripts/run_knowledge_work_h1_ablation_packet.py` accepts `--repeat <n>` and passes it through to the focused ablation runner
  - `scripts/run_knowledge_work_ablation_packet.py` writes `repeat_count`, `base_episode_count`, and repeated episode execution into manifests and summary payloads
- The H1i prompt-contract repeat3 second-stage packet is now executed and saturated:
  - command:
    - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1i_slice.yaml --packet-id mlx_prompt_contract_candidates --run-group-id 20260507T_h1i_prompt_contract_candidates_repeat3_v1 --repeat 3`
  - packet: [`20260507T_h1i_prompt_contract_candidates_repeat3_v1`](../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet)
  - shape: `5` rows x `4` H1i workflow families x `3` repeats = `60` traces
  - all five rows matched at `real_world_readiness_avg = 0.97710`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace analysis found `0` note events and `0` failure candidates
  - interpretation: repeated H1i is a clean negative result. The saturated candidate packet is stable, not flaky. The next packet needs probe-derived live cases, especially visual/parallel no-call and argument-mismatch cases, rather than more repeats of these packaged workflows
- H1j is now scaffolded as the probe-derived live packet:
  - config: [`configs/knowledge_work_h1j_slice.yaml`](../configs/knowledge_work_h1j_slice.yaml)
  - brief: [`docs/continuity/h1j-slice.md`](continuity/h1j-slice.md)
  - candidate packet id: `mlx_probe_derived_tool_contract_candidates`
  - helper-ablation packet id: `mlx_probe_derived_helper_ablation`
  - it selects six packaged live workflows that map to the no-directive probe failures: visual no-call/readback pressure plus API/CLI argument mismatch
  - `parallel_audit_array_literal` is explicitly deferred because the current packaged live surface has no faithful parallel-tool workflow
  - dry-run validation produced the expected command with `5` systems and `6` live episode ids
- The first H1j probe-derived candidate packet also saturated:
  - command:
    - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1j_slice.yaml --packet-id mlx_probe_derived_tool_contract_candidates --run-group-id 20260507T_h1j_probe_derived_candidates_v1`
  - packet: [`20260507T_h1j_probe_derived_candidates_v1`](../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet)
  - shape: `5` rows x `6` packaged live workflow families = `30` traces
  - all five rows matched at `real_world_readiness_avg = 0.96577`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace analysis found `0` note events and `0` failure candidates
  - interpretation: mapping probe failures back to packaged workflows was not enough. The raw probe is still the better discriminator. The next H1j run should remove controller helpers on the same probe-derived set, then the next prompt-contract wave should target the probe directly
- The paired H1j helper-ablation packet also saturated:
  - command:
    - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1j_slice.yaml --packet-id mlx_probe_derived_helper_ablation --run-group-id 20260507T_h1j_probe_derived_helpers_v1`
  - packet: [`20260507T_h1j_probe_derived_helpers_v1`](../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet)
  - contracted, no-directive, no-controller-repair, no-controller-fallback, and no-argument-repair rows all matched at `real_world_readiness_avg = 0.96577`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace analysis found `21` `controller_repair_disabled` markers on the disabled-repair row but `0` failure candidates
  - interpretation: H1j does not reintroduce controller dependence. The benchmark-style packaged workflow path is now empirically less discriminating than the raw tool-contract probe for this question
- Prompt-contract wave 2 is now defined as a raw-probe-first candidate set:
  - contracts:
    - `schema_literal_tool_required_v2`: combines schema anchoring, literal argument copying, and tool-required behavior
    - `visual_next_call_state_v2`: targets visual next-call/no-call collapse after a visual result exists
    - `parallel_array_required_v2`: targets JSON-array shape for independent multi-source/parallel checks
  - registry rows:
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_next_call_state`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required`
  - probe packet runner now accepts `--candidate-wave v2`
  - dry-run packet: [`20260507T_prompt_contract_wave2_dry_run_v1`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1)
  - interpretation: after H1i/H1j saturation, wave 2 must earn promotion on the raw probe before any more H1 spend
- Prompt-contract wave 2 is now executed and remains a partial-gain probe result:
  - packet: [`20260507T_prompt_contract_wave2_execute_v1`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1)
  - summary: [`candidate_gate_summary.md`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1/candidate_gate_summary.md)
  - `schema_literal_tool_required_v2`: exact `0.125`, executable `0.0`, delta exact vs no-directive `+0.125`, dominant failure `argument_mismatch`, recommendation `weak_exact_gain`
  - `visual_next_call_state_v2`: exact `0.0`, executable `1.0`, dominant failure `no_tool_call`, recommendation `visual_executable_gain_only`
  - `parallel_array_required_v2`: exact `0.0`, executable `0.0`, dominant failure `no_tool_call`, recommendation `no_probe_gain`
  - interpretation: wave 2 confirms the split between exact protocol fidelity and executable visual recovery. Combining schema/literal/tool-required wording still only moves one exact case, visual-state wording recovers executable behavior without canonical JSON, and parallel-array wording does not fix no-call collapse. The next discriminator should be exact-probe live replay or a faithful parallel packaged workflow, not another H1i/H1j repeat.
- Prompt-contract promotion decisions are now generated as a report table:
  - table: [`prompt_contract_promotion_decisions.csv`](../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv)
  - weak one-case exact gains and visual executable-only gains are held for exact-probe replay
  - `parallel_array_required_v2` is rejected for H1 promotion because it produced no exact or executable probe gain
  - interpretation: the next packet should be selected by promotion evidence, not by prompt-contract plausibility
- Exact-probe replay now exists as the bridge out of packaged-workflow saturation:
  - script: [`scripts/build_tool_probe_replay_packet.py`](../scripts/build_tool_probe_replay_packet.py)
  - packet: [`20260507T_no_directive_exact_probe_replay_v1`](../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1)
  - source probe: [`20260507T_mlx_no_directive_probe_v1`](../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1)
  - shape: `8` failed no-directive exact-call cases, with `argument_mismatch = 4` and `no_tool_call = 4`
  - next-action table: `build_canonical_argument_replay = 4`, `build_visual_state_replay_executor = 3`, `build_parallel_array_replay_or_workflow = 1`
  - contents: per-case JSON with messages, media, allowed tool specs, expected calls, source actual calls, raw source output, and contracted baseline context
  - command manifest: one runnable `run_tool_directive_probe.py --case-id <case>` command per replay case
  - interpretation: this packet is not live workflow execution. It is the evidence-preserving bridge needed before creating a faithful parallel live workflow or an operator-visible exact-probe replay executor.
- Exact-probe replay execution confirms the failures are stable:
  - packet: [`20260507T_no_directive_exact_probe_replay_execute_v1`](../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1)
  - command:
    - `uv run python scripts/build_tool_probe_replay_packet.py --run-group-id 20260507T_no_directive_exact_probe_replay_execute_v1 --execute`
  - result: exact `0 / 8`
  - all four source `argument_mismatch` cases replayed as `argument_mismatch`
  - all four source `no_tool_call` cases replayed as `no_tool_call`
  - interpretation: the raw no-directive failure set is stable under exact-case replay. The next implementation should target one of the three replay next-action families directly, starting with high-priority visual-state replay or parallel-array replay/workflow coverage.
- Contracted exact-probe replay gives the matched replay baseline:
  - packet: [`20260507T_contracted_exact_probe_replay_execute_v1`](../results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1)
  - command:
    - `uv run python scripts/build_tool_probe_replay_packet.py --run-group-id 20260507T_contracted_exact_probe_replay_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only --execute`
  - result: exact `7 / 8`
  - CLI/API argument cases, parallel array, and visual follow-on/readback cases all become exact
  - `visual_form_target_literal` remains non-exact but executable through selector aliasing
  - interpretation: replay now reproduces the central A/B cleanly: contracted MLX is raw-interface strong on the same eight cases where no-directive MLX remains at `0 / 8`
- Exact-probe replay comparison is now machine-readable:
  - script: [`scripts/compare_tool_probe_replay_packets.py`](../scripts/compare_tool_probe_replay_packets.py)
  - comparison: [`20260507T_contracted_vs_no_directive_exact_replay_v1`](../results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1)
  - shared cases: `8`
  - contracted exact rate: `0.875`
  - no-directive exact rate: `0.0`
  - delta exact rate: `-0.875`
  - case deltas: `7` exact-match drops plus the visual-form case moving from executable paraphrase to no tool call
- Focused visual-state replay isolates the visual no-call seam:
  - no-directive packet: [`20260507T_visual_state_exact_replay_no_directive_v1`](../results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_no_directive_v1)
  - contracted packet: [`20260507T_visual_state_exact_replay_contracted_v1`](../results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_contracted_v1)
  - comparison: [`20260507T_visual_state_contracted_vs_no_directive_v1`](../results/tool_probe_replay_comparisons/20260507T_visual_state_contracted_vs_no_directive_v1)
  - no-directive exact: `0 / 3`
  - contracted exact: `2 / 3`, with the remaining case executable
  - interpretation: the visual-specific no-directive failure is not only exact-copy brittleness; it is a tool-call initiation failure. The next model-side experiment should target visual next-call initiation explicitly, or the next harness experiment should make visual replay operator-visible before returning to packaged workflows.
- Focused parallel-array replay isolates the deferred H1j family:
  - no-directive packet: [`20260507T_parallel_array_exact_replay_no_directive_v1`](../results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_no_directive_v1)
  - contracted packet: [`20260507T_parallel_array_exact_replay_contracted_v1`](../results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_contracted_v1)
  - comparison: [`20260507T_parallel_array_contracted_vs_no_directive_v1`](../results/tool_probe_replay_comparisons/20260507T_parallel_array_contracted_vs_no_directive_v1)
  - no-directive exact: `0 / 1`
  - contracted exact: `1 / 1`
  - interpretation: the missing packaged workflow is now backed by a replayable raw A/B. The next live-workflow design should preserve the expected two-call array contract instead of mapping it onto a single packaged task completion.
- H1k adds the deferred parallel-audit packaged workflow, but it is still a negative live-workflow result:
  - config: [`configs/knowledge_work_h1k_slice.yaml`](../configs/knowledge_work_h1k_slice.yaml)
  - brief: [`docs/continuity/h1k-slice.md`](continuity/h1k-slice.md)
  - workflow: `ops_parallel_audit_review`
  - replay pressure mapped from: `parallel_audit_array_literal`
  - candidate packet: [`20260507T_h1k_parallel_audit_candidates_v1`](../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet)
  - helper packet: [`20260507T_h1k_parallel_audit_helpers_v1`](../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet)
  - candidate result: all five rows matched readiness `0.91780`, strict/recovered `1.0 / 1.0`, controller repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, and `0` failure candidates
  - helper result: contracted, no-directive, no-repair, no-fallback, and no-argument-repair rows all matched the same readiness and interface metrics; trace mining found `3` expected `controller_repair_disabled` markers and `0` failure candidates
  - interpretation: packaged workflow execution is now safe and attributable for the parallel-audit family, but it decomposes the one-turn parallel contract enough that it does not reproduce the raw no-directive failure. This is useful evidence against spending more H1-style cycles until the pressure is preserved exactly.
- CLI-live exact replay is now the stronger bridge out of packaged-workflow saturation:
  - operator entrypoint: `uv run moonie-agent replay-live`
  - inspection entrypoints:
    - `uv run moonie-agent packet --kind tool-probe-replay-live --packet-id <packet_id>`
    - `uv run moonie-agent packet --kind tool-probe-replay-live-comparison --packet-id <comparison_id>`
  - continuity brief: [`docs/continuity/live-exact-replay-results.md`](continuity/live-exact-replay-results.md)
  - comparison script: [`scripts/compare_tool_probe_replay_live_packets.py`](../scripts/compare_tool_probe_replay_live_packets.py)
  - dry-run smoke packet: [`20260507T_parallel_array_replay_live_dry_run_v1`](../results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1)
  - the live replay CLI defaults to no-directive MLX, supports `--execute`, `--case-id`, `--packet-dir`, and `--json`, renders through Rich, and writes under `results/tool_probe_replay_live/`
- CLI-live replay preserves the raw A/B across all eight source failures:
  - canonical argument family:
    - no-directive live packet: [`20260507T_canonical_argument_no_directive_live_execute_v1`](../results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1), exact `0 / 4`, all `argument_mismatch`
    - contracted live packet: [`20260507T_canonical_argument_contracted_live_execute_v1`](../results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1), exact `4 / 4`
    - comparison: [`20260507T_canonical_argument_contracted_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1), delta exact `-1.0`, actual-call delta `0`
  - visual no-call family:
    - no-directive live packet: [`20260507T_visual_state_no_directive_live_execute_v1`](../results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1), exact `0 / 3`, all `no_tool_call`
    - contracted live packet: [`20260507T_visual_state_contracted_live_execute_v1`](../results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1), exact `2 / 3`, with the remaining visual-form case executable through selector aliasing
    - comparison: [`20260507T_visual_state_contracted_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1), delta exact `-0.6666666666666666`
  - parallel array family:
    - no-directive live packet: [`20260507T_parallel_array_no_directive_live_execute_v1`](../results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1), exact `0 / 1`, expected calls `2`, actual calls `0`, failure `no_tool_call`
    - contracted live packet: [`20260507T_parallel_array_contracted_live_execute_v1`](../results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1), exact `1 / 1`, expected calls `2`, actual calls `2`
    - comparison: [`20260507T_parallel_array_contracted_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1), delta exact `-1.0`, actual-call delta `-2`
  - interpretation: the final tool-turn directive does three different kinds of work: canonical CLI/API argument copying, visual follow-on tool initiation, and independent two-call parallel shape preservation. H1k proves packaged workflow safety; CLI-live replay proves the raw mechanism still breaks without the directive.
- The MLX tool-contract report now includes the live replay evidence:
  - curated report: [`docs/reports/mlx-tool-contract-harnessing.md`](reports/mlx-tool-contract-harnessing.md)
  - generated report: [`results/reports/mlx_tool_contract_harnessing/report.md`](../results/reports/mlx_tool_contract_harnessing/report.md)
  - then-current manifest: `32` tables and `20` figures
  - new table families include live parallel, live visual, and live canonical replay deltas
  - new figures include the live parallel replay gap and the combined live replay focus gap
  - interpretation: the current research artifact now distinguishes top-line readiness, exact raw protocol compliance, executable visual paraphrase, controller helper dependence, and live operator-visible replay behavior in one report family
- Prompt-contract wave three is now executed against the raw probe:
  - dry-run packet: [`20260507T_prompt_contract_wave3_dry_run_v1`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_dry_run_v1)
  - executed packet: [`20260507T_prompt_contract_wave3_execute_v1`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1)
  - candidates:
    - `canonical_json_copy_v3`
    - `visual_tool_initiation_v3`
    - `parallel_two_call_array_v3`
  - raw probe result:
    - `canonical_json_copy_v3`: exact `0.125`, executable `0.0`, delta exact vs no-directive `+0.125`, recommendation `weak_exact_gain`
    - `visual_tool_initiation_v3`: exact `0.125`, executable `1.0`, delta exact vs no-directive `+0.125`, recommendation `weak_exact_gain`
    - `parallel_two_call_array_v3`: exact `0.0`, executable `0.0`, delta exact vs no-directive `0.0`, recommendation `no_probe_gain`
  - interpretation: wave three sharpens the same boundary rather than solving it. Visual initiation is the best candidate so far because it combines an exact probe gain with executable visual recovery. The parallel-specific wording still does not fix parallel no-call collapse.
- Wave-three live replay gives the first candidate-level live discriminator:
  - canonical candidate packet: [`20260507T_canonical_argument_canonical_json_copy_live_execute_v1`](../results/tool_probe_replay_live/20260507T_canonical_argument_canonical_json_copy_live_execute_v1)
  - visual candidate packet: [`20260507T_visual_state_visual_tool_initiation_live_execute_v1`](../results/tool_probe_replay_live/20260507T_visual_state_visual_tool_initiation_live_execute_v1)
  - canonical comparison vs no-directive: [`20260507T_canonical_argument_canonical_json_copy_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_canonical_json_copy_vs_no_directive_live_v1)
  - visual comparison vs no-directive: [`20260507T_visual_state_visual_tool_initiation_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260507T_visual_state_visual_tool_initiation_vs_no_directive_live_v1)
  - canonical live result: exact stays `0 / 4`; two cases regress from argument mismatch to no tool call, so `canonical_json_copy_v3` should not be promoted
  - visual live result: exact improves from `0 / 3` to `1 / 3`, executable visual-form recovery improves from `0.0` to `1.0`, and all three visual cases emit one tool call
  - visual still trails contracted MLX: contracted exact `2 / 3`, visual-initiation exact `1 / 3`; the remaining miss is `visual_latest_filter_literal`, where the candidate enters the protocol but uses the wrong visual tool
  - interpretation: the next useful candidate should preserve visual tool initiation while adding stricter visual state/tool selection. The live replay gate prevented a false promotion of canonical JSON copy and narrowed the real remaining target.
- Prompt-contract wave four is now executed and reported:
  - implementation: `visual_state_tool_selection_v4`
  - dry-run packet: [`20260508T_prompt_contract_wave4_dry_run_v1`](../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_dry_run_v1)
  - executed packet: [`20260508T_prompt_contract_wave4_execute_v1`](../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1)
  - raw probe result: exact `1 / 8`, executable `0 / 1`, delta exact vs no-directive `+0.125`, recommendation `weak_exact_gain`
  - summary: one improved case, zero regressed cases versus no-directive, dominant failure `no_tool_call`, failure split `argument_mismatch:2`, `call_count_mismatch:1`, `no_tool_call:3`, `wrong_tool:1`, `exact:1`
  - live visual packet: [`20260508T_visual_state_tool_selection_live_execute_v1`](../results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1)
  - live comparison vs no-directive: [`20260508T_visual_state_tool_selection_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1), exact improves from `0 / 3` to `1 / 3`
  - live comparison vs contracted: [`20260508T_visual_state_contracted_vs_tool_selection_live_v1`](../results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1), exact falls from `2 / 3` to `1 / 3`, executable visual-form recovery falls from `1.0` to `0.0`
  - case read: `visual_readback_region_literal` remains exact, `visual_latest_filter_literal` still fails as `wrong_tool`, and `visual_form_target_literal` regresses to `no_tool_call`
  - interpretation: the added state/tool-selection wording did not solve the targeted remaining visual referent failure. Wave three's useful mechanism appears to be tool initiation, not broad visual state rule text.
  - next research move: try a more surgical prompt-contract or harness hint around latest-selection filtering and `refine_selection`, then gate it through raw probe plus CLI-live visual replay before any H1 spend.
- Prompt-contract wave five is now executed and rejected at the raw gate:
  - implementation: `visual_refine_selection_v5`
  - dry-run packet: [`20260508T_prompt_contract_wave5_dry_run_v1`](../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_dry_run_v1)
  - executed packet: [`20260508T_prompt_contract_wave5_execute_v1`](../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1)
  - raw probe result: exact `0 / 8`, executable `0 / 1`, delta exact vs no-directive `0.0`, recommendation `no_probe_gain`
  - summary: zero improved cases, zero regressed cases versus no-directive, dominant failure `no_tool_call`, failure split `argument_mismatch:1`, `call_count_mismatch:1`, `no_tool_call:6`
  - live replay was intentionally skipped because the candidate did not clear the raw probe gate
  - interpretation: making the prompt more surgical around `refine_selection` did not preserve visual tool initiation. Standalone wording-only refinement has now failed twice after wave three; the next move should change either the generation-time contract shape, the tool catalog/routing presentation, or the harness diagnostic around visual tool choice.
- Visual tool-choice diagnostics now turn live visual replay failures into expected-vs-actual tool rows:
  - script: [`scripts/analyze_visual_tool_choice_diagnostics.py`](../scripts/analyze_visual_tool_choice_diagnostics.py)
  - packet: [`20260508T_visual_tool_choice_wave3_wave4_v1`](../results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_v1)
  - inputs: wave-three visual initiation live packet and wave-four visual state/tool-selection live packet
  - output counts: `exact = 2`, `tool_ok_argument_alias_executable = 1`, `visual_tool_initiation_missing = 1`, `wrong_visual_tool_selection = 2`
  - key row: `visual_latest_filter_literal` expects `refine_selection`, but both candidate packets emit `extract_layout`
  - interpretation: the next fix should not be another generic prompt reminder. The model is still treating a latest-selection filtering request as a locating/layout request, so the next harness change should make tool roles or routing priority more separable at generation time.
- Tool-catalog visual role profile now isolates that routing mechanism:
  - implementation:
    - [`src/gemma4_capability_map/tools/planner.py`](../src/gemma4_capability_map/tools/planner.py)
    - [`src/gemma4_capability_map/research_controls.py`](../src/gemma4_capability_map/research_controls.py)
    - [`scripts/run_tool_catalog_profile_probe_packet.py`](../scripts/run_tool_catalog_profile_probe_packet.py)
  - profile: `visual_role_catalog_v1`
  - candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog`
  - dry-run packet: [`20260508T_visual_role_catalog_v1_dry_run`](../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_dry_run)
  - executed packet: [`20260508T_visual_role_catalog_v1_probe`](../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe)
  - raw probe result: exact `1 / 8`, executable visual `1 / 1`, delta exact vs no-directive `+0.125`
  - visual case read:
    - `visual_form_target_literal`: tool entry recovered and executable target succeeds, but `target_query` is `phone issue` instead of canonical `validation error`
    - `visual_latest_filter_literal`: wrong-tool/no-call collapses into the right tool, `refine_selection`, but `filter_query` is `latest issue` instead of canonical `latest`
    - `visual_readback_region_literal`: exact
  - live replay packet: [`20260508T_visual_role_catalog_live_execute_v1`](../results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1), exact `1 / 3`, executable visual target recovered
  - comparisons:
    - [`20260508T_visual_role_catalog_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_no_directive_v1)
    - [`20260508T_visual_role_catalog_vs_visual_tool_initiation_v1`](../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_tool_initiation_v1)
    - [`20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1`](../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1)
  - interpretation: this is the clearest post-wave-five learning. The tool catalog can change Gemma's visual tool choice without reintroducing the exact directive. The remaining visual problem is now literal argument fidelity after correct routing, not broad visual initiation.
- Prompt-contract wave six tests and rejects a broad composition with literal guarding:
  - candidate: `literal_argument_guard_v1` + `visual_role_catalog_v1`
  - dry-run packet: [`20260508T_visual_catalog_literal_guard_v6_dry_run`](../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_dry_run)
  - executed packet: [`20260508T_visual_catalog_literal_guard_v6_probe`](../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe)
  - raw result: exact `1 / 8`, executable visual `0 / 1`, delta exact vs no-directive `+0.125`
  - failure split: `argument_mismatch:4`, `no_tool_call:3`, `exact:1`
  - interpretation: adding generic literal-copy wording on top of the catalog profile interferes with protocol entry and loses the catalog-only executable recovery. The next literal mechanism must be narrower than the existing broad `literal_argument_guard_v1`.

### Verification

- Narrow regression suite passes after the scoring and runtime-preflight changes:
  - `tests/test_answer_match.py`
  - `tests/test_stressors.py`
  - `tests/test_runtime_utils.py`
- Additional routing/runtime integrity regressions now pass:
  - `tests/test_tool_planner.py`
  - `tests/test_executor.py`
  - `tests/test_tool_eval.py`
  - `tests/test_gemma4_runner.py`
- Real-world execution/reporting integrity also passes after the new service-backed matrix work:
- The latest `KnowledgeWorkArena` hardening and specialist-stability pass now verifies cleanly end to end:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
  - `105 passed`
- The real specialist-backed policy-hardening pass also surfaced a durable runtime cleanup item:
  - both replayable and live runs emitted `top_p` / `top_k` generation-flag warnings on the HF specialist path
  - this did not block execution, but it should be normalized away because backend-specific decoding warnings are benchmark noise
- Real-world execution/reporting integrity also passes after the new service-backed matrix work:
  - `tests/test_alpha_matrix_script.py`
  - `tests/test_benchmark_module.py`
  - `tests/test_runtime_utils.py`
  - `tests/test_real_world_metrics.py`
  - `tests/test_replay_summary.py`

## 2026-05-08 - Visual Catalog Argument-Hints Wave

- Visual tool-choice diagnostics were refreshed to include wave three, wave four, and the visual role catalog profile:
  - script: [`scripts/analyze_visual_tool_choice_diagnostics.py`](../scripts/analyze_visual_tool_choice_diagnostics.py)
  - packet: [`20260508T_visual_tool_choice_wave3_wave4_catalog_v1`](../results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1)
  - diagnostic transition: `visual_latest_filter_literal` moves from `wrong_visual_tool_selection` under `visual_tool_initiation_v3` and `visual_state_tool_selection_v4` to `visual_literal_argument_mismatch` under `visual_role_catalog_v1`
  - interpretation: the catalog profile solved routing for the latest-filter case; the remaining miss was literal selector preservation.
- A narrower catalog profile was added and probed:
  - profile: `visual_role_catalog_argument_hints_v2`
  - candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints`
  - dry-run packet: [`20260508T_visual_role_catalog_argument_hints_v2_dry_run`](../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_dry_run)
  - executed probe packet: [`20260508T_visual_role_catalog_argument_hints_v2_probe`](../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe)
  - raw result: exact `2 / 8`, executable visual `0 / 1`, delta exact vs no-directive `+0.25`
  - comparison vs v1 catalog: [`20260508T_visual_argument_hints_vs_role_catalog_v1`](../results/tool_catalog_profile_probe_comparisons/20260508T_visual_argument_hints_vs_role_catalog_v1)
- Live replay promoted the argument-hints candidate because the raw gate moved the targeted case:
  - live packet: [`20260508T_visual_catalog_argument_hints_live_execute_v1`](../results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1)
  - exact result: `2 / 3`
  - comparison vs no-directive: [`20260508T_visual_catalog_argument_hints_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1), delta exact `+0.6666666666666666`
  - comparison vs contracted: [`20260508T_visual_catalog_argument_hints_vs_contracted_v1`](../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1), delta exact `0.0`, delta executable `-1.0`
  - comparison vs v1 catalog: [`20260508T_visual_catalog_argument_hints_vs_role_catalog_v1`](../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1), delta exact `+0.3333333333333333`, delta executable `-1.0`
- What changed:
  - `visual_latest_filter_literal` is now exact with `refine_selection(selection_id="sel-001", filter_query="latest")`
  - `visual_readback_region_literal` stays exact
  - `visual_form_target_literal` regresses from v1's executable paraphrase to non-executable argument mismatch
- Interpretation:
  - This is the strongest visual exactness result so far for no-directive MLX: it matches contracted MLX at `2 / 3` exact on the focused visual live replay.
  - It is not a full replacement for controller-backed recovery because executable visual-form targeting is worse than both contracted MLX and `visual_role_catalog_v1`.
  - The next candidate should preserve v2's exact latest-filter selector behavior while restoring v1's executable form-target behavior.
- Report update:
  - generated report: [`results/reports/mlx_tool_contract_harnessing/report.md`](../results/reports/mlx_tool_contract_harnessing/report.md)
  - curated report: [`docs/reports/mlx-tool-contract-harnessing.md`](reports/mlx-tool-contract-harnessing.md)
  - then-current manifest: `42` tables and `25` figures

## 2026-05-08 - Visual Split-Selector Negative Result And Publication Ledger

- A follow-up visual catalog profile was added to test whether broader split-selector wording could preserve v2's latest-filter exactness while restoring v1's executable form-target behavior:
  - profile: `visual_role_catalog_split_selector_hints_v3`
  - candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints`
  - dry-run packet: [`20260508T_visual_role_catalog_split_selector_hints_v3_dry_run`](../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_dry_run)
  - executed probe packet: [`20260508T_visual_role_catalog_split_selector_hints_v3_probe`](../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe)
  - raw result: exact `1 / 8`, executable visual `0 / 1`, delta exact vs no-directive `+0.125`
- Direct comparisons showed v3 should not be promoted:
  - v3 vs v2: [`20260508T_visual_split_selector_hints_vs_argument_hints_v2`](../results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2), delta exact `-0.125`
  - v3 vs v1: [`20260508T_visual_split_selector_hints_vs_role_catalog_v1`](../results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_role_catalog_v1), delta exact `0.0`, executable regression vs v1
  - skipped-live decision: [`20260508T_visual_split_selector_hints_live_replay_skipped_v1`](../results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1)
- What changed:
  - `visual_latest_filter_literal` stayed exact, preserving the useful v2 selector behavior.
  - `visual_readback_region_literal` regressed because the model emitted `tool_name` instead of `name`.
  - `visual_form_target_literal` still did not become executable.
- Interpretation:
  - v3 is negative evidence against broader visual selector prose as the next mechanism.
  - The next visual intervention should be schema-local or executor-grounded, with a raw-probe gate before any live replay.
- Publication-facing artifacts were added:
  - claim/evidence ledger: [`results/reports/publication_evidence_ledger/ledger.md`](../results/reports/publication_evidence_ledger/ledger.md)
  - publication readiness audit: [`results/reports/publication_readiness_audit/publication_readiness_audit.md`](../results/reports/publication_readiness_audit/publication_readiness_audit.md)
  - paper outline: [`docs/paper/moonie-gemma-harnessing-paper-outline.md`](paper/moonie-gemma-harnessing-paper-outline.md)
  - updated generated report: [`results/reports/mlx_tool_contract_harnessing/report.md`](../results/reports/mlx_tool_contract_harnessing/report.md)
  - current manifest: `45` tables and `25` figures

## 2026-05-09 - Visual Schema-Field Hints And Hard-Slice Design

- A schema-local visual catalog profile was added after v3 showed that broader selector prose could destabilize JSON shape:
  - implementation: [`src/gemma4_capability_map/tools/planner.py`](../src/gemma4_capability_map/tools/planner.py)
  - registry system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints`
  - profile: `visual_role_catalog_schema_field_hints_v4`
  - dry-run packet: [`20260509T_visual_role_catalog_schema_field_hints_v4_dry_run`](../results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_dry_run)
  - executed probe packet: [`20260509T_visual_role_catalog_schema_field_hints_v4_probe`](../results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe)
- Raw result:
  - exact `2 / 8`
  - executable visual `0 / 1`
  - delta exact vs no-directive `+0.25`
  - comparison vs v2: [`20260509T_visual_schema_field_hints_vs_argument_hints_v2`](../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2), delta exact `0.0`
  - comparison vs v3: [`20260509T_visual_schema_field_hints_vs_split_selector_v3`](../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_split_selector_v3), delta exact `+0.125`
  - comparison vs v1: [`20260509T_visual_schema_field_hints_vs_role_catalog_v1`](../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_role_catalog_v1), delta exact `+0.125`, executable regression vs v1
- Case read:
  - `visual_latest_filter_literal` stays exact with `refine_selection(selection_id="sel-001", filter_query="latest")`
  - `visual_readback_region_literal` is exact again, so v4 repairs the v3 `tool_name`/`name` regression
  - `visual_form_target_literal` remains non-executable and now over-prefers `refine_selection(selection_id="latest", filter_query="phone issue")` even though no real selection id exists
- Promotion decision:
  - skipped-live packet: [`20260509T_visual_schema_field_hints_live_replay_skipped_v1`](../results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1)
  - reason: v4 ties v2 exactness but does not improve it, remains below v1/contracted executable form targeting, and risks false selection carryover
- Interpretation:
  - schema-local field descriptions are cleaner than broad visual prose, but they are not enough
  - the remaining mechanism is not simply "tell the model what fields mean"
  - the next candidate should explicitly separate valid opaque `selection_id` carryover from literal filter-token copying and visible-region targeting
- A fresh visual hard-slice design packet was added:
  - script: [`scripts/build_visual_hard_slice_design.py`](../scripts/build_visual_hard_slice_design.py)
  - test: [`tests/test_visual_hard_slice_design.py`](../tests/test_visual_hard_slice_design.py)
  - packet: [`results/reports/visual_hard_slice_design/design.md`](../results/reports/visual_hard_slice_design/design.md)
  - case count: `8`
  - families: visual argument copying, visual tool routing, visual referent carryover, and visual region readback
  - status at creation: design-stage artifact, not model-performance evidence; this status is superseded by the executed hard-slice packet recorded below
- Publication/reporting updates:
  - MLX tool-contract report manifest is now `49` tables and `25` figures
  - evidence ledger is now `8` claims and `24` evidence sources with `0` missing sources
  - readiness audit is now `22` checks, `20` blocking checks, `0` blocking failures, and `paper_draft_ready`
  - new ledger claim: `C8_visual_hard_slice_targets_remaining_uncertainty`
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_design.py -q`
  - `uv run pytest tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py -q`
  - `uv run python scripts/build_visual_hard_slice_design.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-09 - Executed Visual Hard Slice And Schema-Field Reversal

- The visual hard-slice design was promoted from a design-only packet into a replayable and executed probe packet:
  - executable cases: [`src/gemma4_capability_map/runtime/visual_hard_slice.py`](../src/gemma4_capability_map/runtime/visual_hard_slice.py)
  - packet runner: [`scripts/run_visual_hard_slice_probe_packet.py`](../scripts/run_visual_hard_slice_probe_packet.py)
  - dry-run packet: [`20260509T_visual_hard_slice_dry_run_v1`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_dry_run_v1)
  - executed packet: [`20260509T_visual_hard_slice_execute_v1`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_execute_v1)
  - CLI packet inspection: `uv run moonie-agent packet --kind visual-hard-slice-probe --packet-id 20260509T_visual_hard_slice_execute_v1 --json`
- Hard-slice gate result:
  - contracted MLX: exact `8 / 8`, executable `8 / 8`
  - no-directive MLX: exact `1 / 8`, executable `1 / 8`, dominant failure `no_tool_call`
  - `visual_role_catalog_v1`: exact `3 / 8`, executable `3 / 8`
  - `visual_role_catalog_argument_hints_v2`: exact `6 / 8`, executable `7 / 8`
  - `visual_role_catalog_split_selector_hints_v3`: exact `5 / 8`, executable `6 / 8`
  - `visual_role_catalog_schema_field_hints_v4`: exact `6 / 8`, executable `8 / 8`
  - `visual_role_catalog_v1 + literal_guard`: exact `3 / 8`, executable `4 / 8`
- Interpretation:
  - The fresh hard slice breaks the previous top-line saturation and exposes visual prompt-contract differences that the packaged H1 surfaces were no longer exposing.
  - Schema-field hints are now split evidence: they remain negative on the original three-case focused replay because they do not recover the original executable form-target case, but they are the strongest no-directive profile on the independently authored hard slice because they preserve full executability.
  - Contracted MLX remains the protocol upper bound because it is the only row with exact `8 / 8`.
  - The next useful visual move is not another broad prompt rewrite. It is to inspect the two schema-field exact misses, compare those misses against argument-hints and contracted deltas, and decide whether a narrow exactness repair can preserve the `8 / 8` executable gain.
- Reporting updates:
  - generated report: [`results/reports/mlx_tool_contract_harnessing/report.md`](../results/reports/mlx_tool_contract_harnessing/report.md)
  - visual hard-slice gate table: [`visual_hard_slice_probe_gates.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_probe_gates.csv)
  - visual hard-slice family table: [`visual_hard_slice_family_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_family_summary.csv)
  - visual hard-slice case deltas: [`visual_hard_slice_case_deltas_vs_no_directive.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_case_deltas_vs_no_directive.csv) and [`visual_hard_slice_case_deltas_vs_contracted.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_case_deltas_vs_contracted.csv)
  - visual hard-slice figure: [`visual_hard_slice_probe_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_probe_gate.svg)
  - current generated report manifest: `54` tables and `26` figures
  - evidence ledger: `8` claims, `25` evidence sources, `0` missing sources
  - publication readiness audit: `25` checks, `23` blocking checks, `0` blocking failures, status `paper_draft_ready`
- Verification:
  - `uv run pytest tests/test_visual_hard_slice.py tests/test_visual_hard_slice_probe_packet.py tests/test_runtime_cli.py::test_runtime_cli_packet_json_inspects_visual_hard_slice_probe_packet -q`
  - `uv run python scripts/run_visual_hard_slice_probe_packet.py --run-group-id 20260509T_visual_hard_slice_dry_run_v1`
  - `uv run python scripts/run_visual_hard_slice_probe_packet.py --run-group-id 20260509T_visual_hard_slice_execute_v1 --execute`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-09 - H1n Residual Hybrid Label-Guard Holdout

- Built a fresh residual replay-shaped holdout from the post-repair misses and near-misses:
  - packet: [`20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1)
  - diagnostic: [`visual_alias_transfer_residual_diagnostic`](../results/reports/visual_alias_transfer_residual_diagnostic)
  - report table: [`visual_hard_slice_residual_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_residual_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_residual_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_residual_live_replay_gate.svg)
- Added the `visual_role_catalog_hybrid_label_guard_v8` profile:
  - keeps argument-hints literal selector behavior
  - keeps v7 code-suffix and stale-selection activation guards
  - adds a generic component-label guard so component classes such as pills, tiles, chips, badges, fields, nodes, alerts, and toasts are preferred over state/content values when the user names the component
  - intentionally avoids leaking the exact residual labels into the prompt contract
- Live residual matrix:
  - contracted/default MLX: strict/executor-equivalent `2 / 8`
  - no-directive MLX: strict/executor-equivalent `4 / 8`
  - argument hints v2: strict `5 / 8`, executor-equivalent `7 / 8`
  - oblique code hints v6: strict/executor-equivalent `6 / 8`
  - oblique code guard v7: strict `6 / 8`, executor-equivalent `7 / 8`
  - hybrid label guard v8: strict/executor-equivalent `7 / 8`
- Case-level interpretation:
  - v8 fixes `residual_chip_v82_chart_decoy`, `residual_alert_h73_toggle_decoy`, and stale-selection `residual_field_m20_stale_selection_decoy` relative to no-directive
  - v8 improves strict exactness over argument hints by `+0.25` but ties argument hints on executor-equivalence
  - v8 improves strict exactness over v7 code guard by `+0.125` but ties v7 on executor-equivalence
  - the persistent miss is `residual_state_pill_note_decoy`, where the model still chooses the state/content value instead of the component label
- Research interpretation:
  - We now have a cleaner answer to the post-repair question: code guard was the right direction, but a hybrid component-label activation guard is stronger on strict selector fidelity.
  - The result is not yet a universal visual-profile promotion. The improvement is on a small replay-shaped holdout, and the unsolved `state pill` case says component-role/value disambiguation remains a real harnessing problem.
  - The next scientific move is a micro-slice around component-role/value ambiguity, especially pill-like controls whose text value is semantically tempting.
- Reporting and readiness:
  - MLX tool-contract report now has `72` tables and `35` figures
  - publication evidence ledger now has `29` claims, `137` evidence sources, and `0` missing sources
  - publication readiness audit now has `100` checks, `98` blocking checks, `0` blocking failures, and status `paper_draft_ready`
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1n_hybrid_label_guard_registry_row_preserves_catalog_profile tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1 --suite alias_transfer_residual_v7`
  - `uv run python -m gemma4_capability_map.runtime.cli replay-live --packet-dir results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1n_residual_hybrid_label_guard_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard --registry configs/model_registry.yaml --execute --json`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-transfer-residual`
  - `uv run pytest tests/test_mlx_tool_contract_report.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py tests/test_visual_live_stress_diagnostic.py -q`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-09 - Packaged Replay Gap Diagnostic

- Added a diagnostic that compares replay-shaped visual gains against packaged H1 visual workflow saturation:
  - script: [`scripts/analyze_packaged_replay_gap.py`](../scripts/analyze_packaged_replay_gap.py)
  - diagnostic: [`results/reports/packaged_replay_gap_diagnostic/diagnostic.md`](../results/reports/packaged_replay_gap_diagnostic/diagnostic.md)
  - surface table: [`packaged_replay_gap_surfaces.csv`](../results/reports/packaged_replay_gap_diagnostic/tables/packaged_replay_gap_surfaces.csv)
- Result:
  - H1l visual executor-equivalence: max replay executor-equivalence delta `1.0`; packaged readiness span `0.0`; packaged strict-interface span `0.0`
  - H1m visual alias-repeat: max replay executor-equivalence delta `0.375`; packaged readiness span `0.0`; packaged strict-interface span `0.0`
  - saturated packaged surfaces: `2 / 2`
- Interpretation:
  - Packaged workflow design is not a neutral wrapper. It is part of the benchmark contract.
  - H1l/H1m are valid negative results about current packaged surfaces, not negative results about the underlying visual alias/decoy mechanism.
  - The next visual experiment should preserve replay pressure more faithfully or use less staged live tasks before returning to packaged helper ablations.
- Reporting updates:
  - publication evidence claim `C15_packaged_visual_surfaces_wash_out_replay_discrimination` records the gap
  - publication readiness audit now requires the diagnostic and reproduction script
  - readiness audit now has `51` checks, `49` blocking, and `0` blocking failures
- Verification:
  - `uv run python scripts/analyze_packaged_replay_gap.py`
  - `uv run pytest tests/test_packaged_replay_gap_diagnostic.py -q`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-09 - H1n Visual Alias-Transfer Replay Matrix

- Added and executed a fresh alias-transfer replay suite to test whether the visual alias-repeat signal transfers to new labels and decoys without staging the task into packaged workflows:
  - script: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
  - suite flag: `--suite alias_transfer_v3`
  - brief: [`docs/continuity/h1n-slice.md`](continuity/h1n-slice.md)
  - source packet: [`20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1`](../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1)
  - diagnostic: [`results/reports/visual_alias_transfer_diagnostic`](../results/reports/visual_alias_transfer_diagnostic)
  - report table: [`visual_hard_slice_alias_transfer_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_alias_transfer_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_transfer_live_replay_gate.svg)
- New transfer cases:
  - `transfer_review_tile_notice_table_decoy`
  - `transfer_status_pill_chart_decoy`
  - `transfer_error_banner_note_decoy`
  - `transfer_queue_badge_person_decoy`
  - `transfer_form_error_old_selection_chip_decoy`
  - `transfer_signature_warning_checkbox_decoy`
- Design:
  - `4` visual argument-transfer cases with fresh tile/pill/banner/badge targets
  - `2` visual tool-routing transfer cases with stale-selection or wrong-region decoys
  - replay-live entrypoint only; this is not a packaged workflow and does not add frontend work
- Result:
  - no-directive MLX: strict `0 / 6`, executor-equivalent `2 / 6`
  - contracted MLX: strict `5 / 6`, executor-equivalent `1 / 6`
  - role catalog v1: strict `1 / 6`, executor-equivalent `3 / 6`
  - argument hints v2: strict `1 / 6`, executor-equivalent `6 / 6`
  - schema-field hints v4: strict `1 / 6`, executor-equivalent `2 / 6`
  - schema target literals v5: strict `1 / 6`, executor-equivalent `4 / 6`
- Interpretation:
  - H1n is the first post-packaging-gap positive transfer result. Argument hints v2 generalizes best for executor-equivalent target success on fresh labels and decoys.
  - Schema-field hints v4 did not transfer its alias-repeat executor-equivalence advantage; it only added one strict exact win.
  - Contracted MLX appeared to be the strict-fidelity upper bound, but the exact-vs-executor split needed scorer inspection before being reported as a model-only weakness.
  - This strengthens the paper claim that strict protocol fidelity and executor target success must be separate endpoints, not one blended score.
- Reporting updates:
  - generated MLX tool-contract report now has `66` tables and `32` figures
  - publication evidence claim `C16_visual_alias_transfer_favors_argument_hints_executor_grounding` records the result
  - readiness audit now has `56` checks, `54` blocking, and `0` blocking failures
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1 --suite alias_transfer_v3`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_transfer_no_directive_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_transfer_argument_hints_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints --execute --json`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-transfer`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-09 - H1n Contract-Split Diagnostic

- Added a diagnostic for the H1n alias-transfer strict/executor split:
  - script: [`scripts/analyze_h1n_alias_transfer_contract_split.py`](../scripts/analyze_h1n_alias_transfer_contract_split.py)
  - diagnostic: [`results/reports/h1n_alias_transfer_contract_split/diagnostic.md`](../results/reports/h1n_alias_transfer_contract_split/diagnostic.md)
  - expected-call audit table: [`h1n_expected_call_contract_audit.csv`](../results/reports/h1n_alias_transfer_contract_split/tables/h1n_expected_call_contract_audit.csv)
  - replay split table: [`h1n_replay_contract_split.csv`](../results/reports/h1n_alias_transfer_contract_split/tables/h1n_replay_contract_split.csv)
- Result:
  - `5 / 6` generated H1n expected-call contracts do not satisfy the packet's own `expected_execution` oracle.
  - Contracted MLX has `4` exact-but-not-executor-equivalent rows.
  - Argument hints v2 still has `6 / 6` executor-target successes.
- Interpretation:
  - H1n strict exactness is not yet an oracle strictness metric. It mostly measures whether a run matched the heuristic planner's generated expected call.
  - The contracted `5 / 6` strict score should not be used as a clean target-success upper bound in the paper.
  - The H1n executor-equivalence result remains useful: argument hints v2 is the transfer winner under the target oracle.
  - The next H1n move is to rebuild the alias-transfer packet with oracle expected calls derived from target region labels, then rerun the same CLI-live matrix.
- Reporting updates:
  - publication evidence claim `C17_h1n_strict_exactness_matches_planner_not_oracle` records this as a benchmark-contract issue.
  - publication readiness audit now requires the contract-split diagnostic and reproduction script.
- Verification:
  - `uv run python scripts/analyze_h1n_alias_transfer_contract_split.py`
  - `uv run pytest tests/test_h1n_alias_transfer_contract_split.py -q`

## 2026-05-09 - H1n Oracle Expected-Call Packet

- Rebuilt the alias-transfer packet generator so future `alias_transfer_v3` packets use oracle expected calls:
  - implementation: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
  - rebuilt packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2`](../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2)
  - updated brief: [`docs/continuity/h1n-slice.md`](continuity/h1n-slice.md)
- Contract change:
  - old H1n v1 expected calls came from `plan_tool_calls(...)`, which replicated heuristic planner mistakes.
  - oracle v2 derives `extract_layout.target_query` from the target region label named by `expected_execution`.
  - examples: `review tile`, `status pill`, `error banner`, `queue badge`, `validation error`, `signature warning`.
- Verification:
  - every oracle v2 expected call executes through the deterministic local visual executor to the expected region id.
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2 --suite alias_transfer_v3`
- Next:
  - use oracle expected calls as the default strict H1n contract
  - compare oracle-v2 strict exactness against executor-equivalence before any packaged or helper-ablation promotion

## 2026-05-09 - H1n Oracle Replay-Live Matrix

- Fixed the live replay runtime so packet-authored expected calls are preserved:
  - implementation: [`src/gemma4_capability_map/runtime/tool_directive_probe.py`](../src/gemma4_capability_map/runtime/tool_directive_probe.py)
  - implementation: [`src/gemma4_capability_map/runtime/tool_probe_replay_live.py`](../src/gemma4_capability_map/runtime/tool_probe_replay_live.py)
  - regression test: [`tests/test_tool_probe_replay_live.py`](../tests/test_tool_probe_replay_live.py)
  - without this fix, `moonie-agent replay-live` could load an oracle replay packet but silently rescore it against freshly planned expected calls.
- Executed the oracle H1n matrix across the same six rows:
  - no-directive: exact `2 / 6`, executor-equivalent `2 / 6`
  - contracted: exact `1 / 6`, executor-equivalent `1 / 6`
  - role catalog v1: exact `3 / 6`, executor-equivalent `3 / 6`
  - argument hints v2: exact `5 / 6`, executor-equivalent `6 / 6`
  - schema-field hints v4: exact `2 / 6`, executor-equivalent `2 / 6`
  - schema target literals v5: exact `4 / 6`, executor-equivalent `4 / 6`
- Generated evidence:
  - diagnostic: [`results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md`](../results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md)
  - report table: [`visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_alias_transfer_oracle_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_transfer_oracle_live_replay_gate.svg)
  - publication evidence claim: `C18_h1n_oracle_transfer_identifies_argument_hints_as_clean_winner`
- Interpretation:
  - This is one of the cleanest current Moonie harnessing findings. Once the benchmark contract is executable against the oracle target, narrow argument hints are the strongest local-Gemma transfer mechanism.
  - Schema target literals remain useful, but they trail argument hints on both exactness and executor-equivalence.
  - Contracted prompting is not a reliable upper bound here; it regresses below no-directive on the oracle transfer packet.
  - The result directly strengthens the paper thesis that benchmark contract quality changes what we think the model is good at.
- Verification:
  - `uv run pytest tests/test_tool_probe_replay_live.py tests/test_tool_directive_probe.py -q`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-transfer-oracle`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
  - `uv run pytest tests/test_tool_probe_replay_live.py tests/test_tool_directive_probe.py tests/test_visual_hard_slice_live_stress_packet.py tests/test_visual_live_stress_diagnostic.py tests/test_mlx_tool_contract_report.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py -q`
- Next:
  - repeat the oracle packet or build a non-packaged live helper-ablation slice centered on argument hints
  - keep strict exactness, executor-equivalence, and controller-helper usage separated in every table

## 2026-05-09 - H1n Oracle Argument-Hints Helper Ablation

- Added registry rows for the H1n winner with one helper disabled at a time:
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_repair`
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_fallback`
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_argument_repair`
- Executed all three rows on the oracle H1n packet:
  - no controller repair: exact `5 / 6`, executor-equivalent `6 / 6`
  - no controller fallback: exact `5 / 6`, executor-equivalent `6 / 6`
  - no argument repair: exact `5 / 6`, executor-equivalent `6 / 6`
- Generated direct comparisons against the original argument-hints oracle row:
  - all three comparisons have exact delta `0.0`
  - all three comparisons have executor-equivalence delta `0.0`
  - diagnostic: [`results/reports/h1n_oracle_helper_ablation/diagnostic.md`](../results/reports/h1n_oracle_helper_ablation/diagnostic.md)
- Interpretation:
  - the argument-hints gain is model/catalog-contract side on this packet, not an artifact of controller repair, controller fallback, or argument repair
  - this is still slice-local; it does not claim controller helpers are irrelevant broadly
- Verification:
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2 --output-dir results/tool_probe_replay_live/20260509T_h1n_oracle_argument_hints_no_controller_repair_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_repair --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2 --output-dir results/tool_probe_replay_live/20260509T_h1n_oracle_argument_hints_no_controller_fallback_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_fallback --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2 --output-dir results/tool_probe_replay_live/20260509T_h1n_oracle_argument_hints_no_argument_repair_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_argument_repair --execute --json`
  - `uv run python scripts/analyze_h1n_oracle_helper_ablation.py`
  - `uv run pytest tests/test_h1n_oracle_helper_ablation.py tests/test_knowledge_work_h1.py -q`

## 2026-05-09 - H1n Oracle Alias-Transfer Repeat Packet

- Added a fresh H1n repeat suite to avoid overfitting the oracle finding to one six-case label set:
  - suite: `alias_transfer_repeat_v4`
  - packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1)
  - implementation: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
- New transfer cases:
  - `transfer_repeat_audit_card_email_decoy`
  - `transfer_repeat_priority_tag_chart_decoy`
  - `transfer_repeat_warning_toast_note_decoy`
  - `transfer_repeat_latency_chip_person_decoy`
  - `transfer_repeat_missing_field_old_selection_decoy`
  - `transfer_repeat_consent_alert_toggle_decoy`
- Design:
  - `4` fresh visual argument-transfer cases
  - `2` fresh visual tool-routing transfer cases
  - oracle expected calls are again derived from target region labels and verified against the deterministic visual executor
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1 --suite alias_transfer_repeat_v4`
- Next:
  - execute no-directive, argument hints, schema target literals, and contracted rows first
  - only run helper-ablation repeats if the argument-hints row still separates from no-directive

## 2026-05-09 - H1n Oracle Alias-Transfer Repeat Matrix

- Executed the full repeat matrix on `alias_transfer_repeat_v4`:
  - no-directive: exact `2 / 6`, executor-equivalent `2 / 6`
  - contracted: exact `0 / 6`, executor-equivalent `0 / 6`
  - role catalog v1: exact `4 / 6`, executor-equivalent `4 / 6`
  - argument hints v2: exact `5 / 6`, executor-equivalent `6 / 6`
  - schema-field hints v4: exact `4 / 6`, executor-equivalent `4 / 6`
  - schema target literals v5: exact `5 / 6`, executor-equivalent `6 / 6`
- Generated evidence:
  - diagnostic: [`results/reports/visual_alias_transfer_repeat_diagnostic/diagnostic.md`](../results/reports/visual_alias_transfer_repeat_diagnostic/diagnostic.md)
  - argument-hints comparison: [`results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_repeat_argument_hints_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_repeat_argument_hints_vs_no_directive_v1)
  - schema-literal comparison: [`results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_repeat_schema_literal_targets_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_repeat_schema_literal_targets_vs_no_directive_v1)
- Interpretation:
  - the core H1n result repeats: narrow catalog-profile mechanisms beat no-directive on fresh visual labels and decoys
  - the winner set is now a tie between argument hints and schema target literals
  - contracted prompting again fails as an upper bound, now falling below no-directive at `0 / 6`
  - this shifts the next question from "is argument hints real?" to "which of argument hints vs schema target literals is more robust under less lexical or less staged live visual tasks?"
- Verification:
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_h1n_oracle_repeat_argument_hints_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_h1n_oracle_repeat_schema_literal_targets_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets --execute --json`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-transfer-repeat`
  - `uv run pytest tests/test_visual_live_stress_diagnostic.py -q`
- Next:
  - do not promote only argument hints anymore; carry both argument hints and schema target literals into the next less-staged live visual slice

## 2026-05-09 - H1n Oracle Transfer Synthesis

- Added a compact synthesis report over the two oracle H1n transfer packets and the argument-hints helper ablation:
  - report: [`results/reports/h1n_oracle_transfer_synthesis/report.md`](../results/reports/h1n_oracle_transfer_synthesis/report.md)
  - JSON: [`results/reports/h1n_oracle_transfer_synthesis/report.json`](../results/reports/h1n_oracle_transfer_synthesis/report.json)
  - synthesis table: [`results/reports/h1n_oracle_transfer_synthesis/tables/h1n_oracle_transfer_synthesis.csv`](../results/reports/h1n_oracle_transfer_synthesis/tables/h1n_oracle_transfer_synthesis.csv)
- Synthesis result:
  - first oracle packet: argument hints v2 is the clean winner at exact `5 / 6` and executor-equivalent `6 / 6`
  - repeat oracle packet: argument hints v2 and schema target literals v5 tie at exact `5 / 6` and executor-equivalent `6 / 6`
  - contracted prompting is not an upper bound on the oracle transfer packets: `1 / 6` on the first oracle packet, `0 / 6` on the repeat
  - controller repair, controller fallback, and argument repair removals all preserve argument hints at exact `5 / 6` and executor-equivalent `6 / 6` on the first oracle packet
- Research interpretation:
  - H1n now answers the first transfer question: narrow catalog-profile interventions can improve local Gemma visual target success when the benchmark contract uses executable oracle expected calls.
  - H1n does not yet answer the generalization question. The next paper-relevant experiment should compare argument hints against schema target literals in a less staged visual task or a third held-out oracle family with less lexical target labels.
  - This is now publication claim `C21_h1n_two_packet_oracle_synthesis_narrows_next_visual_question`.
- Verification:
  - `uv run python scripts/build_h1n_oracle_transfer_synthesis.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
  - `uv run pytest tests/test_h1n_oracle_transfer_synthesis.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py -q`

## 2026-05-09 - H1n Oblique-Label Oracle Packet

- Added a third held-out oracle transfer suite, `alias_transfer_oblique_v5`, to test nonsemantic visible target labels under semantic decoys:
  - packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1)
  - generator: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
- Design:
  - six cases with visible labels such as `node q17`, `badge m88`, `chip z33`, `cell r42`, `field e19`, and `alert p55`
  - the exact label is present in the user instruction, so strict exactness is fair
  - nearby decoys repeat the semantic content, so executor-equivalence can still reveal whether a semantic paraphrase hit the correct visible target
- Contract note:
  - an early draft used two-character labels such as `m8`, which the local visual executor tokenizer ignored because it drops tokens shorter than three characters
  - the committed packet uses at least three-character code tokens where needed, and the tests verify every oracle expected call reaches the intended local region
- Verification:
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1 --suite alias_transfer_oblique_v5`
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`

## 2026-05-09 - H1n Oblique-Label Oracle Matrix

- Executed the full oblique-label matrix:
  - no-directive: exact `0 / 6`, executor-equivalent `0 / 6`
  - contracted: exact `1 / 6`, executor-equivalent `1 / 6`
  - role catalog v1: exact `2 / 6`, executor-equivalent `2 / 6`
  - argument hints v2: exact `4 / 6`, executor-equivalent `4 / 6`
  - schema-field hints v4: exact `3 / 6`, executor-equivalent `3 / 6`
  - schema target literals v5: exact `0 / 6`, executor-equivalent `0 / 6`
- Generated evidence:
  - diagnostic: [`results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md`](../results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md)
  - argument-hints comparison: [`results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_argument_hints_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_argument_hints_vs_no_directive_v1)
  - schema-literal comparison: [`results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_schema_literal_targets_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_schema_literal_targets_vs_no_directive_v1)
- Interpretation:
  - the oblique packet breaks the argument-hints/schema-literal tie from the repeat packet in favor of argument hints
  - schema-field hints is now the second-place mechanism on code-like labels
  - schema target literals are brittle here: they do not improve over no-directive when literal code-like labels sit beside semantic decoys
  - contracted prompting is again not an upper bound
  - the next useful move is a miss analysis on the two argument-hints failures and a less replay-shaped live visual task carrying argument hints and schema-field hints forward
- Verification:
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_argument_hints_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints --execute --json`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-transfer-oblique`
  - `uv run pytest tests/test_visual_live_stress_diagnostic.py -q`

## 2026-05-09 - H1n Oblique Miss Analysis

- Added a miss diagnostic for the two strongest oblique rows:
  - report: [`results/reports/h1n_oblique_miss_analysis/diagnostic.md`](../results/reports/h1n_oblique_miss_analysis/diagnostic.md)
  - table: [`results/reports/h1n_oblique_miss_analysis/tables/h1n_oblique_misses.csv`](../results/reports/h1n_oblique_miss_analysis/tables/h1n_oblique_misses.csv)
- Argument-hints misses:
  - `transfer_oblique_cell_r42_notice_decoy`: expected `cell r42`, actual `cell`; this broadens the local executor match to the notice, target cell, and approval table
  - `transfer_oblique_alert_p55_toggle_decoy`: expected `alert p55`, actual `consent toggle`; this selects the explicitly negated decoy
- Schema-field misses:
  - one semantic broad-selection miss, one code-suffix truncation miss, and one tool-entry failure
- Interpretation:
  - the next intervention should not revive broad schema-target-literal wording
  - the narrow remaining target is code-suffix preservation plus negated-decoy resistance, tested against preservation of the four argument-hints wins
- Verification:
  - `uv run python scripts/analyze_h1n_oblique_misses.py`
  - `uv run pytest tests/test_h1n_oblique_miss_analysis.py -q`

## 2026-05-09 - H1n Oblique Code-Hints Candidate

- Added a narrow follow-up catalog profile for the oblique misses:
  - profile: `visual_role_catalog_oblique_code_hints_v6`
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints`
  - implementation: [`src/gemma4_capability_map/tools/planner.py`](../src/gemma4_capability_map/tools/planner.py)
  - registry: [`configs/model_registry.yaml`](../configs/model_registry.yaml)
- Design:
  - inherits the visual role-catalog argument-field framing
  - adds only two targeted rules: preserve code-like visible label suffixes and treat `not X` / `before reading X` as decoy language unless X is the requested target
  - annotates `extract_layout.target_query` with the same narrow code-suffix/negated-decoy contract
- Guardrail:
  - do not promote this profile unless it improves the two argument-hints misses without losing the four argument-hints wins on the oblique packet
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py -q`

## 2026-05-09 - H1n Oblique Code-Hints Live Repair

- Executed the oblique-code profile on the held-out oblique oracle packet:
  - source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1`](../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1)
  - live packet: [`results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_code_hints_execute_v1`](../results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_code_hints_execute_v1)
  - comparison versus no-directive: [`results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_code_hints_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_code_hints_vs_no_directive_v1)
  - comparison versus argument hints: [`results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1`](../results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1)
  - updated diagnostic: [`results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md`](../results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md)
- Result:
  - no-directive MLX: `0 / 6` exact and executor-equivalent
  - argument hints v2: `4 / 6` exact and executor-equivalent
  - oblique code hints v6: `5 / 6` exact and executor-equivalent
  - delta versus argument hints: `+0.167` exact and executor-equivalence
- Case-level finding:
  - repaired `transfer_oblique_cell_r42_notice_decoy` by preserving the full `cell r42` visible code label
  - repaired `transfer_oblique_alert_p55_toggle_decoy` by avoiding the explicitly negated `consent toggle` decoy
  - regressed `transfer_oblique_field_e19_old_selection_decoy` into a wrong-tool case, losing one argument-hints win
- Interpretation:
  - The code-suffix/negated-decoy hypothesis is now supported on the oblique packet, but not clean enough for broad promotion.
  - The profile improves the current hardest H1n transfer packet while creating a new stale-selection style failure, so the next best move is regression analysis and transfer testing on the earlier oracle/repeat packets.
  - This strengthens the publication story because it shows a meaningful harnessing improvement and its tradeoff rather than a one-way leaderboard win.
- Verification:
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_code_hints_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints --execute --json`
  - `uv run python scripts/compare_tool_probe_replay_live_packets.py results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_argument_hints_execute_v1 results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_code_hints_execute_v1 --output-dir results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-transfer-oblique`
  - `uv run pytest tests/test_visual_live_stress_diagnostic.py -q`

## 2026-05-09 - H1n Oblique Code-Hints Delta Diagnostic

- Added a gain/loss diagnostic for the oblique-code repair:
  - script: [`scripts/analyze_h1n_oblique_code_hints_delta.py`](../scripts/analyze_h1n_oblique_code_hints_delta.py)
  - report: [`results/reports/h1n_oblique_code_hints_delta/diagnostic.md`](../results/reports/h1n_oblique_code_hints_delta/diagnostic.md)
  - case table: [`results/reports/h1n_oblique_code_hints_delta/tables/h1n_oblique_code_hints_case_deltas.csv`](../results/reports/h1n_oblique_code_hints_delta/tables/h1n_oblique_code_hints_case_deltas.csv)
- Result:
  - repairs: `2`
  - regression: `1`
  - preserved argument-hints wins: `3`
  - net executor-equivalence gain: `+1` case
- Regression detail:
  - `transfer_oblique_field_e19_old_selection_decoy` regressed from the argument-hints exact `extract_layout(target_query="field e19")` call to `refine_selection(selection_id="sel-e19-archive", filter_query="not")`
  - the local executor fails this as a stale-selection attraction because `sel-e19-archive` is not present in the current visual state
- Interpretation:
  - the code-hints profile is a real oblique repair, but the negation/stale-selection interaction is now the next mechanism to test
  - the next best transfer check is running the same profile on the earlier oracle and repeat packets before designing another wording patch
- Verification:
  - `uv run python scripts/analyze_h1n_oblique_code_hints_delta.py`
  - `uv run pytest tests/test_h1n_oblique_code_hints_delta.py -q`

## 2026-05-09 - H1n Code-Hints Transfer Synthesis

- Ran the oblique-code profile on the earlier oracle and repeat packets:
  - first oracle packet: [`results/tool_probe_replay_live/20260510T_h1n_oracle_code_hints_transfer_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_oracle_code_hints_transfer_execute_v1)
  - repeat packet: [`results/tool_probe_replay_live/20260510T_h1n_oracle_repeat_code_hints_transfer_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_oracle_repeat_code_hints_transfer_execute_v1)
  - first oracle comparison: [`results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_code_hints_vs_argument_hints_transfer_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_code_hints_vs_argument_hints_transfer_v1)
  - repeat comparison: [`results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_repeat_code_hints_vs_argument_hints_transfer_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_repeat_code_hints_vs_argument_hints_transfer_v1)
  - synthesis: [`results/reports/h1n_code_hints_transfer_synthesis/report.md`](../results/reports/h1n_code_hints_transfer_synthesis/report.md)
- Result:
  - first oracle: code hints `3 / 6` exact and executor-equivalent, versus argument hints at `5 / 6` exact and `6 / 6` executor-equivalent
  - repeat: code hints `3 / 6` exact and `4 / 6` executor-equivalent, versus argument hints at `5 / 6` exact and `6 / 6` executor-equivalent
  - oblique: code hints `5 / 6`, versus argument hints at `4 / 6`
  - aggregate over three oracle packets: argument hints `14 / 18` exact and `16 / 18` executor-equivalent; code hints `11 / 18` exact and `12 / 18` executor-equivalent
- Interpretation:
  - oblique code hints is a localized repair for code-like labels, not a broad profile promotion
  - the next profile should be activation-gated or paired with a stale-selection guard before another full transfer run
  - this is valuable negative evidence: the best scientific answer is not "make prompts longer," but "local repairs can overfit and must be transfer-tested"
- Verification:
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2 --output-dir results/tool_probe_replay_live/20260510T_h1n_oracle_code_hints_transfer_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1n_oracle_repeat_code_hints_transfer_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints --execute --json`
  - `uv run python scripts/build_h1n_code_hints_transfer_synthesis.py`
  - `uv run pytest tests/test_h1n_code_hints_transfer_synthesis.py -q`

## 2026-05-09 - H1n Oblique Code-Guard Candidate

- Added a narrower follow-up profile after the code-hints transfer loss:
  - profile: `visual_role_catalog_oblique_code_guard_v7`
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard`
  - implementation: [`src/gemma4_capability_map/tools/planner.py`](../src/gemma4_capability_map/tools/planner.py)
  - registry: [`configs/model_registry.yaml`](../configs/model_registry.yaml)
- Design:
  - keeps the v6 code-like visible-label rule, but phrases it generically as a letter-plus-digits suffix instead of enumerating packet labels
  - adds a stale-selection activation guard: do not choose `refine_selection` solely because the user mentions an old, stale, saved, ignored, or previous `selection_id`
  - tells the model to use `extract_layout` when no current `selection_id` is available and the user asks to locate a visible label on the current image
- Promotion criterion:
  - first run it on the oblique packet
  - keep it only if it preserves the `cell r42` and `alert p55` repairs while fixing the `field e19` stale-selection regression
  - if it passes that gate, rerun the earlier oracle/repeat transfer checks before any broader promotion
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py -q`

## 2026-05-09 - H1n Oblique Code-Guard Live Result

- Executed the activation-gated code-guard profile on the oblique oracle packet:
  - live packet: [`results/tool_probe_replay_live/20260510T_h1n_oracle_oblique_code_guard_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_oracle_oblique_code_guard_execute_v1)
  - comparison versus argument hints: [`results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_argument_hints_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_argument_hints_v1)
  - comparison versus code hints v6: [`results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_code_hints_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_code_hints_v1)
  - comparison versus no-directive: [`results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_no_directive_v1`](../results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_no_directive_v1)
  - updated matrix diagnostic: [`results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md`](../results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md)
- Result:
  - code guard v7 reaches `6 / 6` exact and executor-equivalent
  - delta versus argument hints: `+0.333`
  - delta versus v6 code hints: `+0.167`
  - the prior `field e19` stale-selection regression is repaired while preserving the `cell r42` and `alert p55` repairs
- Interpretation:
  - the activation guard is positive evidence on the current hardest oblique packet
  - do not generalize yet: v6 failed transfer on the earlier oracle/repeat packets, so v7 must now run the same transfer check
- Verification:
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1n_oracle_oblique_code_guard_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard --execute --json`
  - `uv run python scripts/compare_tool_probe_replay_live_packets.py results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_code_hints_execute_v1 results/tool_probe_replay_live/20260510T_h1n_oracle_oblique_code_guard_execute_v1 --output-dir results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_code_hints_v1`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-transfer-oblique`
  - `uv run pytest tests/test_visual_live_stress_diagnostic.py -q`

## 2026-05-09 - H1n Code-Guard Transfer Synthesis

- Ran the code-guard profile on the same transfer packets where v6 code hints failed:
  - first oracle packet: [`results/tool_probe_replay_live/20260510T_h1n_oracle_code_guard_transfer_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_oracle_code_guard_transfer_execute_v1)
  - repeat packet: [`results/tool_probe_replay_live/20260510T_h1n_oracle_repeat_code_guard_transfer_execute_v1`](../results/tool_probe_replay_live/20260510T_h1n_oracle_repeat_code_guard_transfer_execute_v1)
  - synthesis: [`results/reports/h1n_code_guard_transfer_synthesis/report.md`](../results/reports/h1n_code_guard_transfer_synthesis/report.md)
- Result:
  - code guard versus v6: improves from `11 / 18` to `14 / 18` exact and from `12 / 18` to `15 / 18` executor-equivalent
  - code guard versus argument hints: ties aggregate exactness at `14 / 18`, but trails executor-equivalence at `15 / 18` versus `16 / 18`
  - code guard is positive versus argument hints only on the oblique packet; it remains negative on the first oracle and repeat packets
- Interpretation:
  - the activation guard is a real improvement over v6 and fixes the known stale-selection failure
  - argument hints remains the broadest current profile across the three oracle packets
  - the next useful experiment is a fresh post-repair holdout, not immediate promotion of code guard
- Verification:
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2 --output-dir results/tool_probe_replay_live/20260510T_h1n_oracle_code_guard_transfer_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1 --output-dir results/tool_probe_replay_live/20260510T_h1n_oracle_repeat_code_guard_transfer_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard --execute --json`
  - `uv run python scripts/build_h1n_code_guard_transfer_synthesis.py`
  - `uv run pytest tests/test_h1n_code_guard_transfer_synthesis.py -q`

## 2026-05-09 - Schema Target Literal v5 Negative Hard-Slice Repair

- A narrow hard-slice repair candidate was added after inspecting the two v4 exact misses:
  - profile: `visual_role_catalog_schema_literal_targets_v5`
  - registry system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets`
  - implementation: [`src/gemma4_capability_map/tools/planner.py`](../src/gemma4_capability_map/tools/planner.py)
  - dry-run packet: [`20260509T_visual_hard_slice_v5_dry_run`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_v5_dry_run)
  - executed packet: [`20260509T_visual_hard_slice_executor_equivalence_v1`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1)
  - v5-vs-v4 comparison: [`schema_literal_targets_vs_schema_field_hints`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints)
- Result:
  - v4 schema-field hints: exact `6 / 8`, executable `8 / 8`
  - v5 schema target literals: exact `5 / 8`, executable `7 / 8`
  - direct v5-vs-v4 delta: exact `-0.125`, executable `-0.125`
  - v5 did not repair either v4 paraphrase:
    - `visual_metric_panel_vs_table_selector`: expected `dashboard metric`, actual `metric panel`
    - `visual_callout_warning_with_user_decoy`: expected `slide callout`, actual `slide callout warning`
  - v5 introduced a new wrong-tool failure on `visual_form_error_with_prior_selection_decoy`: expected `extract_layout`, actual `refine_selection(selection_id="sel-stale", filter_query="validation error")`
- Interpretation:
  - The target-literal wording is an overcorrection. It does not solve exact target-label fidelity and weakens the important no-stale-selection guard.
  - v4 remains the best no-directive hard-slice profile because it preserves full executability.
  - The remaining research question is sharper now: are the two v4 exact misses true executor failures, or are they benchmark-canonical-label mismatches where the model produced an executable label? The next slice should separate exact-protocol label fidelity from executor-visible target success before writing another wording profile.
- Reporting updates:
  - generated report manifest remains `54` tables and `26` figures
  - evidence ledger is now `9` claims and `27` evidence sources with `0` missing sources
  - readiness audit is now `26` checks, `24` blocking checks, `0` blocking failures, and status `paper_draft_ready`
  - new ledger claim: `C9_schema_literal_targets_v5_is_negative_evidence`
- Verification:
  - `uv run pytest tests/test_prompt_contracts.py tests/test_visual_hard_slice_probe_packet.py -q`
  - `uv run python scripts/run_visual_hard_slice_probe_packet.py --run-group-id 20260509T_visual_hard_slice_v5_dry_run --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets`
  - `uv run python scripts/run_visual_hard_slice_probe_packet.py --run-group-id 20260509T_visual_hard_slice_executor_equivalence_v1 --execute`
  - `uv run python scripts/compare_tool_directive_probes.py results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets --output-dir results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
  - `uv run pytest tests/test_prompt_contracts.py tests/test_visual_hard_slice_probe_packet.py tests/test_mlx_tool_contract_report.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py tests/test_runtime_cli.py::test_runtime_cli_packet_json_inspects_visual_hard_slice_probe_packet -q`
  - `uv run moonie-agent packet --kind visual-hard-slice-probe --packet-id 20260509T_visual_hard_slice_executor_equivalence_v1 --json`

## 2026-05-09 - Visual Hard-Slice Exactness Versus Executor Target Diagnostic

- A new diagnostic script now separates strict benchmark-canonical visual argument exactness from executor-visible target success:
  - script: [`scripts/analyze_visual_hard_slice_exactness.py`](../scripts/analyze_visual_hard_slice_exactness.py)
  - artifact: [`results/reports/visual_hard_slice_exactness_diagnostic`](../results/reports/visual_hard_slice_exactness_diagnostic)
  - seed packet: [`20260509T_visual_hard_slice_executor_equivalence_v1`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1)
- Result:
  - v4 schema-field hints: exact `6 / 8`, executable `8 / 8`, non-exact executor successes `2`, benchmark-label artifact candidates `2`, true harness failures `0`
  - v5 schema target literals: exact `5 / 8`, executable `7 / 8`, non-exact executor successes `2`, benchmark-label artifact candidates `2`, true harness failures `1`
  - v4 exact gap `visual_metric_panel_vs_table_selector`: expected target `hard-metric-1001`, actual target `hard-metric-1001`
  - v4 exact gap `visual_callout_warning_with_user_decoy`: expected target `hard-callout-decoy-1102`, actual target `hard-callout-decoy-1102`
  - v5 adds a true executor failure on `visual_form_error_with_prior_selection_decoy` by choosing stale `refine_selection(selection_id="sel-stale")` instead of current-image `extract_layout`
- Interpretation:
  - The two v4 hard-slice exact misses are not current evidence of failed visual targeting. They are better classified as executor-success selector aliases under the local deterministic visual executor.
  - This strengthens the paper framing: strict correctness, exact protocol fidelity, and executor-visible success are distinct metrics.
  - This led directly to first-class executor-equivalence scoring beside strict exactness, instead of another target-query wording profile.
- Reporting updates:
  - generated MLX tool-contract report now has `56` tables and `26` figures
  - evidence ledger now has `10` claims, `31` evidence sources, `0` missing sources
  - readiness audit now has `28` checks, `26` blocking checks, `0` blocking failures, status `paper_draft_ready`
  - new ledger claim: `C10_v4_exact_misses_are_executor_success_aliases`
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_exactness_diagnostic.py -q`
  - `uv run python scripts/analyze_visual_hard_slice_exactness.py --json`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`

## 2026-05-09 - First-Class Executor-Equivalence Hard-Slice Metric

- Executor-visible visual target success is now measured directly in the tool-directive probe and hard-slice packet:
  - probe implementation: [`src/gemma4_capability_map/runtime/tool_directive_probe.py`](../src/gemma4_capability_map/runtime/tool_directive_probe.py)
  - packet runner: [`scripts/run_visual_hard_slice_probe_packet.py`](../scripts/run_visual_hard_slice_probe_packet.py)
  - CLI packet renderer: [`src/gemma4_capability_map/runtime/research_packets.py`](../src/gemma4_capability_map/runtime/research_packets.py)
  - executed packet: [`20260509T_visual_hard_slice_executor_equivalence_v1`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1)
  - v5-vs-v4 comparison: [`schema_literal_targets_vs_schema_field_hints`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints)
- Result:
  - contracted MLX: strict `8 / 8`, executable `8 / 8`, executor-equivalent `8 / 8`
  - no-directive MLX: strict `1 / 8`, executable `1 / 8`, executor-equivalent `1 / 8`
  - `visual_role_catalog_schema_field_hints_v4`: strict `6 / 8`, executable `8 / 8`, executor-equivalent `8 / 8`
  - `visual_role_catalog_schema_literal_targets_v5`: strict `5 / 8`, executable `7 / 8`, executor-equivalent `7 / 8`
  - direct v5-vs-v4 comparison now reports executor-equivalence delta `-0.125`, matching the executable regression and making the stale-selection wrong-tool failure measurable without relying only on strict JSON labels.
- Interpretation:
  - This turns the previous exactness diagnostic into a first-class benchmark channel. The v4 "misses" still matter for strict protocol fidelity, but they are no longer scored as visual target failures when the deterministic executor reaches the expected local element.
  - The current paper framing becomes stronger: strict exactness, recovered/executable operation, and executor-equivalent target success are distinct endpoints. Harness improvements should declare which endpoint they improve.
  - The next H1 move should build a packaged visual workflow around executor-visible success while retaining strict exactness as a separate protocol-fidelity measure.
- Reporting updates to regenerate after this slice:
  - MLX tool-contract report should use [`20260509T_visual_hard_slice_executor_equivalence_v1`](../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1) as the latest visual hard-slice packet.
  - publication evidence ledger claims C8-C10 should point at the executor-equivalence packet.
  - publication readiness audit should require the executor-equivalence packet and its v5-vs-v4 comparison.
- Verification:
  - `uv run pytest tests/test_tool_directive_probe.py tests/test_visual_hard_slice.py tests/test_visual_hard_slice_probe_packet.py tests/test_runtime_cli.py::test_runtime_cli_packet_json_inspects_visual_hard_slice_probe_packet -q`
  - `uv run python scripts/run_visual_hard_slice_probe_packet.py --run-group-id 20260509T_visual_hard_slice_executor_equivalence_v1 --execute`
  - `uv run python scripts/analyze_visual_hard_slice_exactness.py --json`
  - `uv run python scripts/compare_tool_directive_probes.py results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets --output-dir results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints`

## 2026-05-09 - H1l Visual Executor-Equivalence Live Scaffold

- A new packaged-workflow H1 slice now carries the executor-equivalence result into the CLI-first live harness:
  - config: [`configs/knowledge_work_h1l_slice.yaml`](../configs/knowledge_work_h1l_slice.yaml)
  - brief: [`docs/continuity/h1l-slice.md`](continuity/h1l-slice.md)
  - candidate packet id: `mlx_visual_executor_equivalence_candidates`
  - executed candidate packet: [`20260509T_h1l_visual_executor_equivalence_candidates_v1`](../results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet)
  - helper packet id: `mlx_visual_executor_equivalence_helper_ablation`
- Packet shape:
  - five visual live workflows: dashboard review, dashboard referent review, job visual constraint override, finance invoice lock review, and finance invoice revision
  - six candidate rows: contracted MLX, no-directive MLX, role catalog v1, argument hints v2, schema-field hints v4, and schema target literals v5
  - five helper rows for controller repair, controller fallback, and argument repair attribution on the same workflow set
- Interpretation:
  - H1l is not another prompt wording patch. It is the attribution surface for asking whether v4's `8 / 8` executor-equivalent hard-slice behavior survives packaged workflows.
  - The executed candidate packet is negative as a discriminator: all six rows tie at readiness `0.90406`, strict `0.85`, recovered `0.8`, raw clean `1.0`, and repair/fallback/argument repair `0.0 / 0.0 / 0.0`.
  - Current packaged visual workflows are too staged to preserve the hard-slice distinction. Defer the H1l helper packet until a visual live surface separates at least one candidate row.
- Verification:
  - `uv run pytest tests/test_knowledge_work_h1.py::test_h1l_slice_config_maps_to_visual_executor_equivalence_packet -q`
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1l_slice.yaml --packet-id mlx_visual_executor_equivalence_candidates --run-group-id 20260509T_h1l_visual_executor_equivalence_candidates_dry_run_v1 --dry-run`
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1l_slice.yaml --packet-id mlx_visual_executor_equivalence_candidates --run-group-id 20260509T_h1l_visual_executor_equivalence_candidates_v1`
  - `uv run python scripts/summarize_h1_tool_contract.py results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet`

## 2026-05-09 - Replay-Shaped Visual Hard-Slice CLI-Live Result

- The live replay operator can now preserve visual hard-slice cases directly instead of only replaying the original legacy tool-directive cases:
  - runtime loader: [`src/gemma4_capability_map/runtime/tool_probe_replay_live.py`](../src/gemma4_capability_map/runtime/tool_probe_replay_live.py)
  - converter: [`scripts/build_visual_hard_slice_replay_packet.py`](../scripts/build_visual_hard_slice_replay_packet.py)
  - source packet: [`20260509T_visual_hard_slice_no_directive_replay_dry_run_v1`](../results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1)
  - no-directive live packet: [`20260509T_visual_hard_slice_no_directive_hard_replay_live_execute_v2`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_no_directive_hard_replay_live_execute_v2)
  - contracted live packet: [`20260509T_visual_hard_slice_contracted_hard_replay_live_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_contracted_hard_replay_live_execute_v1)
  - role-catalog live packet: [`20260509T_visual_hard_slice_role_catalog_hard_replay_live_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_role_catalog_hard_replay_live_execute_v1)
  - argument-hints live packet: [`20260509T_visual_hard_slice_argument_hints_hard_replay_live_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_argument_hints_hard_replay_live_execute_v1)
  - schema-field live packet: [`20260509T_visual_hard_slice_schema_field_hints_hard_replay_live_execute_v2`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_schema_field_hints_hard_replay_live_execute_v2)
  - schema-target-literal live packet: [`20260509T_visual_hard_slice_schema_literal_targets_hard_replay_live_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_schema_literal_targets_hard_replay_live_execute_v1)
  - comparison matrix:
    - [`20260509T_visual_hard_slice_contracted_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_contracted_vs_no_directive_live_v1)
    - [`20260509T_visual_hard_slice_role_catalog_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_role_catalog_vs_no_directive_live_v1)
    - [`20260509T_visual_hard_slice_argument_hints_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_argument_hints_vs_no_directive_live_v1)
    - [`20260509T_visual_hard_slice_schema_field_hints_vs_no_directive_live_v2`](../results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_schema_field_hints_vs_no_directive_live_v2)
    - [`20260509T_visual_hard_slice_schema_literal_targets_vs_no_directive_live_v1`](../results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_schema_literal_targets_vs_no_directive_live_v1)
- Result:
  - no-directive replay on the two preserved hard-slice failures: strict `0 / 2`, executable `0 / 2`, executor-equivalent `0 / 2`
  - contracted MLX is the upper bound: strict/executable/executor-equivalent `2 / 2`
  - role catalog v1 and argument hints v2 each recover only the stale-selection decoy: strict/executable/executor-equivalent `1 / 2`
  - schema-field hints v4 is the strongest no-directive row: strict `1 / 2`, executable/executor-equivalent `2 / 2`
  - schema target literals v5 remains negative on strict fidelity: strict `0 / 2`, executable/executor-equivalent `1 / 2`
  - v4 makes `visual_form_error_with_prior_selection_decoy` exact and keeps `visual_metric_panel_vs_table_selector` executor-equivalent through a selector alias
  - v5 makes the stale-selection decoy worse by turning it into a wrong-tool failure
- Interpretation:
  - H1l's packaged-workflow saturation was not the final word on the v4 result. It showed that the current packaged visual workflows are too staged. The replay-shaped CLI-live result shows the same hard-slice signal survives in the operator path when the raw case shape is preserved.
  - This is now the best evidence that v4 is a real harnessing improvement for local MLX visual execution, but the improvement target is executor-equivalent visual grounding, not full strict protocol fidelity.
  - The next experiment should not be another broad prompt wording patch. The follow-up stress slice now repeats the executor-alias and stale-selection cases with fresh decoys; the remaining useful extension is more alias/decoy repetition before any H1m packaged workflow.
- Reporting updates:
  - live replay packets now carry executor-equivalence counts/rates beside strict exactness and executable match
  - live replay comparisons now carry per-case executor-equivalence deltas
  - generated MLX tool-contract report now has `59` tables and `28` figures
  - new report section: `Visual Hard-Slice CLI-Live Replay`
  - new report figure: [`visual_hard_slice_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_live_replay_gate.svg)
- Verification:
  - `uv run pytest tests/test_tool_probe_replay_live.py tests/test_tool_probe_replay_live_comparison.py tests/test_runtime_cli.py::test_runtime_cli_packet_json_inspects_tool_probe_replay_live_packet tests/test_runtime_cli.py::test_runtime_cli_packet_json_inspects_tool_probe_replay_live_comparison -q`
  - `uv run pytest tests/test_visual_hard_slice_replay_packet.py tests/test_tool_probe_replay_live.py -q`
  - `uv run python scripts/build_visual_hard_slice_replay_packet.py --run-group-id 20260509T_visual_hard_slice_no_directive_replay_dry_run_v1 --failure-mode argument_mismatch`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_no_directive_replay_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_no_directive_hard_replay_live_execute_v2 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_no_directive_replay_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_contracted_hard_replay_live_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_no_directive_replay_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_role_catalog_hard_replay_live_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_no_directive_replay_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_argument_hints_hard_replay_live_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_no_directive_replay_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_schema_field_hints_hard_replay_live_execute_v2 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_no_directive_replay_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_schema_literal_targets_hard_replay_live_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets --execute --json`
  - `uv run python scripts/compare_tool_probe_replay_live_packets.py results/tool_probe_replay_live/20260509T_visual_hard_slice_no_directive_hard_replay_live_execute_v2 results/tool_probe_replay_live/20260509T_visual_hard_slice_schema_field_hints_hard_replay_live_execute_v2 --output-dir results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_schema_field_hints_vs_no_directive_live_v2`
  - `uv run python scripts/compare_tool_probe_replay_live_packets.py results/tool_probe_replay_live/20260509T_visual_hard_slice_no_directive_hard_replay_live_execute_v2 results/tool_probe_replay_live/20260509T_visual_hard_slice_schema_literal_targets_hard_replay_live_execute_v1 --output-dir results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_schema_literal_targets_vs_no_directive_live_v1`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run pytest tests/test_mlx_tool_contract_report.py tests/test_tool_probe_replay_live_comparison.py -q`

## 2026-05-09 - Visual Hard-Slice Stress CLI-Live Matrix

- Built the next replay-shaped visual stress packet from the two mechanisms exposed by the preserved hard-slice replay:
  - builder: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
  - source packet: [`20260509T_visual_hard_slice_live_stress_dry_run_v1`](../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1)
  - no-directive live baseline: [`20260509T_visual_hard_slice_live_stress_no_directive_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_no_directive_execute_v1)
  - contracted live upper bound: [`20260509T_visual_hard_slice_live_stress_contracted_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_contracted_execute_v1)
  - report table: [`visual_hard_slice_stress_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_stress_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_stress_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_stress_live_replay_gate.svg)
- Result:
  - no-directive MLX: strict `2 / 4`, executable/executor-equivalent `3 / 4`
  - contracted MLX: strict/executable/executor-equivalent `4 / 4`
  - role catalog v1: strict `1 / 4`, executor-equivalent `2 / 4`; this row regresses relative to no-directive on the warning stale-selection decoy
  - argument hints v2: strict `2 / 4`, executor-equivalent `3 / 4`; tied with no-directive on this stress packet
  - schema-field hints v4: strict `2 / 4`, executor-equivalent `4 / 4`
  - schema target literals v5: strict `2 / 4`, executor-equivalent `4 / 4`
- Interpretation:
  - The stress slice is deliberately harder than the preserved two-case slice but less decisive than the original hard-slice probe. It shows no-directive MLX can solve the stale-selection cases without help, while the remaining difference concentrates on a metric-panel alias/decoy case.
  - The main finding survives in a more nuanced form: schema-local catalog hints do not improve strict JSON fidelity over no-directive on this stress packet, but they do recover executor-visible target success on the hardest metric-panel case.
  - This strengthens the paper framing that strict protocol fidelity and executor target success must be reported separately. It also makes the next move clearer: add more metric-panel/callout alias cases and repeats before promoting anything to H1m packaged workflows.
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py tests/test_tool_probe_replay_live.py::test_tool_probe_replay_live_loads_packet_serialized_custom_cases -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260509T_visual_hard_slice_live_stress_dry_run_v1`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_no_directive_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_schema_field_hints_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints --execute --json`
  - `uv run python scripts/build_mlx_tool_contract_report.py`

## 2026-05-09 - Visual Hard-Slice Alias-Repeat Matrix

- Added an eight-case alias-repeat stress suite to repeat the metric-panel and callout selector-alias mechanisms before promoting anything into a new packaged H1 workflow:
  - builder: [`scripts/build_visual_hard_slice_live_stress_packet.py`](../scripts/build_visual_hard_slice_live_stress_packet.py)
  - suite flag: `--suite alias_repeat_v2`
  - source packet: [`20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1`](../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1)
  - no-directive live packet: [`20260509T_visual_hard_slice_live_stress_alias_repeat_no_directive_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_no_directive_execute_v1)
  - contracted live packet: [`20260509T_visual_hard_slice_live_stress_alias_repeat_contracted_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_contracted_execute_v1)
  - role-catalog live packet: [`20260509T_visual_hard_slice_live_stress_alias_repeat_role_catalog_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_role_catalog_execute_v1)
  - argument-hints live packet: [`20260509T_visual_hard_slice_live_stress_alias_repeat_argument_hints_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_argument_hints_execute_v1)
  - schema-field live packet: [`20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_execute_v1)
  - schema-target-literal live packet: [`20260509T_visual_hard_slice_live_stress_alias_repeat_schema_literal_targets_execute_v1`](../results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_schema_literal_targets_execute_v1)
  - report table: [`visual_hard_slice_alias_repeat_live_replay_summary.csv`](../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_repeat_live_replay_summary.csv)
  - report figure: [`visual_hard_slice_alias_repeat_live_replay_gate.svg`](../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_repeat_live_replay_gate.svg)
  - diagnostic: [`results/reports/visual_alias_repeat_diagnostic`](../results/reports/visual_alias_repeat_diagnostic)
- Result:
  - no-directive MLX: strict `2 / 8`, executable/executor-equivalent `5 / 8`
  - contracted MLX: strict `7 / 8`, executable/executor-equivalent `8 / 8`
  - role catalog v1: strict `1 / 8`, executable/executor-equivalent `6 / 8`
  - argument hints v2: strict `2 / 8`, executable/executor-equivalent `6 / 8`
  - schema-field hints v4: strict `2 / 8`, executable/executor-equivalent `7 / 8`
  - schema target literals v5: strict `3 / 8`, executable/executor-equivalent `8 / 8`
  - schema-field improved cases:
    - `stress_callout_warning_person_table_decoy`: no tool call becomes executor-equivalent
    - `stress_metric_panel_with_chart_table_decoys`: argument mismatch becomes executor-equivalent
  - schema target literals additionally makes `stress_callout_warning_risk_note_decoy` exact and recovers full executor-equivalence on the chart/table metric-panel decoy
- Interpretation:
  - The four-case stress result was not a one-off from a single metric-panel example. When alias/decoy pressure is repeated, schema-local profiles still improve executor-visible grounding, while only the full contracted profile approaches strict canonical-label fidelity.
  - This strengthens the central research answer: local MLX Gemma harnessing gains are showing up as executor-grounding improvements under visual alias pressure, not as full strict protocol-copy recovery.
  - The v5 schema-target-literal profile is no longer simply negative on repeated alias pressure: it remains weaker than contracted strict fidelity, but it is the strongest no-directive row on this packet by executor-equivalence and has a small strict gain over no-directive.
  - The next empirical move is to repeat the alias-repeat packet or package only the surviving metric-panel/callout mechanisms into a non-saturated H1m workflow.
- Reporting updates:
  - generated MLX tool-contract report now has `63` tables and `30` figures
  - alias-repeat diagnostic classifies strict gains, executor-only gains, and regressions across the completed five-row matrix
  - publication evidence claim `C13_visual_live_stress_separates_executor_grounding_from_strict_fidelity` now includes the completed alias-repeat matrix
  - publication readiness audit now requires the alias-repeat packet, full comparison set, and generated summary table
- Verification:
  - `uv run pytest tests/test_visual_hard_slice_live_stress_packet.py -q`
  - `uv run python scripts/build_visual_hard_slice_live_stress_packet.py --run-group-id 20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1 --suite alias_repeat_v2`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_no_directive_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive --execute --json`
  - `uv run moonie-agent replay-live --packet-id 20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1 --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints --execute --json`
  - `uv run python scripts/compare_tool_probe_replay_live_packets.py results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_no_directive_execute_v1 results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_execute_v1 --output-dir results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_vs_no_directive_v1`
  - `uv run python scripts/analyze_visual_live_stress_matrix.py --matrix alias-repeat`

## 2026-05-09 - H1m Visual Alias-Repeat Packaged Result

- Added and executed the packaged-workflow promotion target for the completed alias-repeat replay matrix:
  - config: [`configs/knowledge_work_h1m_slice.yaml`](../configs/knowledge_work_h1m_slice.yaml)
  - brief: [`docs/continuity/h1m-slice.md`](continuity/h1m-slice.md)
  - packaged workflow registry: [`configs/packaged_workflows.yaml`](../configs/packaged_workflows.yaml)
  - executed packet: [`20260509T_h1m_visual_alias_repeat_candidates_v1`](../results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet)
  - generated table: [`h1m_visual_alias_repeat_candidate_metrics.csv`](../results/reports/mlx_tool_contract_harnessing/tables/h1m_visual_alias_repeat_candidate_metrics.csv)
  - generated figure: [`h1m_visual_alias_repeat_burden.svg`](../results/reports/mlx_tool_contract_harnessing/figures/h1m_visual_alias_repeat_burden.svg)
- New packaged workflows:
  - `executive_visual_dashboard_revision`
  - `jobs_visual_latest_issue_review`
  - `finance_visual_invoice_hold_review`
- Packet plan:
  - candidate packet: `mlx_visual_alias_repeat_packaged_candidates`
  - helper packet: `mlx_visual_alias_repeat_helper_ablation`
- Result:
  - contracted MLX, no-directive MLX, role catalog v1, argument hints v2, schema-field hints v4, and schema-target-literal v5 all tie
  - readiness: `0.87783`
  - strict/recovered: `0.75 / 0.667`
  - repair/fallback/argument repair: `0.0 / 0.0 / 0.0`
  - raw clean: `1.0`
- Interpretation:
  - H1m is evidence now, and it is negative evidence about the current packaged live surface.
  - The replay-shaped alias-repeat matrix still separates strict protocol fidelity from executor-equivalent visual grounding, but these three packaged workflows wash out that discrimination.
  - Do not run the H1m helper packet yet. There is no row separation for repair/fallback/argument-repair attribution to explain.
  - The next useful visual move is either repeated alias replay, stochastic repeats, or a less staged non-packaged CLI live task that preserves alias/decoy pressure more faithfully than packaged workflows.
- Reporting updates:
  - generated MLX tool-contract report now has `64` tables and `31` figures
  - publication evidence claim `C14_h1m_packaged_alias_repeat_saturates` records the negative packaged result
  - publication readiness audit now requires the H1m packet and generated report table
- Verification:
  - `uv run pytest tests/test_knowledge_work_h1.py::test_h1m_slice_config_maps_to_visual_alias_repeat_packet tests/test_runtime_cli.py::test_runtime_cli_lists_workflows -q`
  - `uv run moonie-agent workflows --lane live_web_stress --workflow-id executive_visual_dashboard_revision --validate`
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1m_slice.yaml --packet-id mlx_visual_alias_repeat_packaged_candidates --run-group-id 20260509T_h1m_visual_alias_repeat_candidates_dry_run_v1 --dry-run`
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1m_slice.yaml --packet-id mlx_visual_alias_repeat_packaged_candidates --run-group-id 20260509T_h1m_visual_alias_repeat_candidates_v1`
  - `uv run python scripts/summarize_h1_tool_contract.py results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet`
  - `uv run python scripts/build_mlx_tool_contract_report.py`
  - `uv run python scripts/build_publication_evidence_ledger.py`
  - `uv run python scripts/audit_publication_readiness.py`
