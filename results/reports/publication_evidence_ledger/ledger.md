# Publication Evidence Ledger

This ledger maps paper-level claims to packet-backed evidence and known limitations.

## Manifest

- generated_at: `2026-05-09T11:05:44.150380+00:00`
- claim_count: `8`
- evidence_source_count: `25`
- missing_source_count: `0`

## Claims

| Claim ID | Status | Evidence | Primary Metric | Limitation | Next Test |
| --- | --- | --- | --- | --- | --- |
| C1_controller_dependence_hidden_by_readiness | supported_current_packets | strong_internal | H1h/H1i no-directive readiness parity with high repair/fallback burden and low raw-clean rate. | Current support is internal to Moonie's knowledge-work harness and local MLX runtime. | Run the same helper-ablation structure on a harder H1 slice selected from raw replay failures. |
| C2_final_tool_directive_causal_for_protocol | supported_current_packets | strong_internal | Contracted exact replay is 7/8 while no-directive exact replay is 0/8 on the same cases. | The replay suite is intentionally focused on eight observed no-directive failures, not a population estimate. | Expand the replay suite with independently authored hard cases and repeated seeds. |
| C3_packaged_workflows_can_saturate | supported_current_packets | moderate_internal | H1j/H1k packaged packets saturated while exact replay still showed no-directive failures. | The packaged workflow scaffolds may make the task easier than the one-turn replay contract. | Build a harder packaged workflow slice that preserves one-turn parallel and visual follow-on pressure. |
| C4_visual_catalog_role_routing_is_real | supported_current_packets | moderate_internal | visual_role_catalog_v1 moves latest-filter from wrong/no-call behavior to refine_selection argument mismatch. | The intervention improves routing more than exact literal fidelity. | Test catalog-role profiles across a larger visual follow-on set with fresh UI states. |
| C5_visual_argument_hints_improve_exactness_but_not_executability | supported_current_packets | moderate_internal | v2 reaches 2/3 focused visual live exactness but loses executable form-target recovery. | The improvement is focused on three visual replay cases and has a known form-target regression. | Search for a split selector intervention that preserves v2 filter exactness and v1 form-target executability. |
| C6_split_selector_wording_is_negative_evidence | negative_result_current_packets | moderate_internal | v3 raw exact falls to 1/8 versus v2 at 2/8 and readback regresses through tool_name/name mismatch. | This is one candidate profile; it does not rule out all field-specific selector interventions. | Try an executor-grounded schema annotation or few-shot-free field contract that does not add broad behavioral prose. |
| C7_schema_field_hints_tie_exactness_without_executability | negative_result_current_packets | moderate_internal | v4 ties v2 at 2/8 raw exact and preserves readback, but stays 0/1 executable and over-prefers refine_selection on form-target. | This tests one schema-field annotation profile on the focused eight-case replay-derived probe. | Create a fresh visual hard slice or constrain refine_selection preference only when a real selection_id is present. |
| C8_visual_hard_slice_targets_remaining_uncertainty | supported_current_packets | moderate_internal | Contracted MLX reaches 8/8 exact and executable; no-directive falls to 1/8; schema-field hints reach 6/8 exact and 8/8 executable. | The packet is eight independently authored visual cases, so it is stronger than design-only evidence but still not a population estimate. | Inspect the two schema-field exact misses, then repeat or extend the hard slice before promoting to a packaged H1 workflow. |

## Evidence Sources

| Claim ID | Type | Exists | Path | Purpose |
| --- | --- | ---: | --- | --- |
| C1_controller_dependence_hidden_by_readiness | h1_ablation_packet | True | results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1 | Full no-directive replication showing controller burden behind readiness parity. |
| C1_controller_dependence_hidden_by_readiness | h1_ablation_packet | True | results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1 | Fast worst-family loop preserving the H1h causal ordering. |
| C1_controller_dependence_hidden_by_readiness | report_table | True | results/reports/mlx_tool_contract_harnessing/tables/packet_summary.csv | Cross-packet readiness, repair, fallback, argument-repair, and raw-clean summary. |
| C2_final_tool_directive_causal_for_protocol | probe_replay_comparison | True | results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1 | A/B replay comparison for the exact same failed no-directive probe cases. |
| C2_final_tool_directive_causal_for_protocol | live_replay_comparison | True | results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1 | Operator-visible live replay of canonical argument failures. |
| C2_final_tool_directive_causal_for_protocol | live_replay_comparison | True | results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1 | Operator-visible live replay of visual no-call failures. |
| C2_final_tool_directive_causal_for_protocol | live_replay_comparison | True | results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1 | Operator-visible live replay of the parallel two-call failure. |
| C3_packaged_workflows_can_saturate | h1_ablation_packet | True | results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet | Probe-derived packaged workflows that saturated across candidate rows. |
| C3_packaged_workflows_can_saturate | h1_ablation_packet | True | results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet | Packaged parallel-audit workflow showing safe scaffold but easier behavior than raw replay. |
| C3_packaged_workflows_can_saturate | report_table | True | results/reports/mlx_tool_contract_harnessing/tables/live_parallel_replay_case_deltas.csv | Live exact-replay evidence that the raw parallel two-call shape still fails without the directive. |
| C4_visual_catalog_role_routing_is_real | diagnostic_packet | True | results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1 | Expected-vs-actual visual tool-choice diagnostic for wave3, wave4, and catalog profile. |
| C4_visual_catalog_role_routing_is_real | catalog_probe_packet | True | results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe | Raw catalog-profile probe showing routing and executable visual-form recovery. |
| C4_visual_catalog_role_routing_is_real | live_replay_comparison | True | results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1 | Focused live comparison showing catalog profile changes wrong-tool/no-call into argument mismatch. |
| C5_visual_argument_hints_improve_exactness_but_not_executability | catalog_probe_packet | True | results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe | Raw v2 catalog probe showing latest-filter exactness and form-target executable regression. |
| C5_visual_argument_hints_improve_exactness_but_not_executability | live_replay_comparison | True | results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1 | Focused live comparison showing v2 matches contracted exactness but loses executable recovery. |
| C5_visual_argument_hints_improve_exactness_but_not_executability | live_replay_comparison | True | results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1 | Focused live comparison showing v2 exact gain versus v1 and executable regression. |
| C6_split_selector_wording_is_negative_evidence | catalog_probe_packet | True | results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe | Raw v3 probe packet and case outputs. |
| C6_split_selector_wording_is_negative_evidence | catalog_probe_comparison | True | results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2 | Direct v3-vs-v2 comparison showing exact regression. |
| C6_split_selector_wording_is_negative_evidence | live_replay_decision | True | results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1 | Promotion decision packet explaining why v3 did not spend live replay budget. |
| C7_schema_field_hints_tie_exactness_without_executability | catalog_probe_packet | True | results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe | Raw v4 probe packet showing schema-field exactness tie and form-target wrong-tool regression. |
| C7_schema_field_hints_tie_exactness_without_executability | catalog_probe_comparison | True | results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2 | Direct v4-vs-v2 comparison showing no exact gain over the current best visual candidate. |
| C7_schema_field_hints_tie_exactness_without_executability | live_replay_decision | True | results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1 | Promotion decision packet explaining why v4 did not spend live replay budget. |
| C8_visual_hard_slice_targets_remaining_uncertainty | visual_hard_slice_probe_packet | True | results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_execute_v1 | Executed fresh visual hard-slice packet across contracted, no-directive, and visual catalog profile rows. |
| C8_visual_hard_slice_targets_remaining_uncertainty | design_packet | True | results/reports/visual_hard_slice_design | Fresh visual hard-slice design packet derived from v1/v2/v3/v4 failure analysis. |
| C8_visual_hard_slice_targets_remaining_uncertainty | live_replay_decision | True | results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1 | Negative v4 promotion decision motivating a fresh visual hard-slice rather than another live replay. |
