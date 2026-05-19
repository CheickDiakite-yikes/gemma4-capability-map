# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_route_arbitration_residual_exactness_visual_stale_selection_gate_visual_value_bearing_target_query_synthesis_visual_contextual_surface_alias_routing_visual_composed_route_gating_visual_negation_guard_visual_semantic_target_preservation`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_h2z_negated_component_target_preservation`
- Baseline exact rate: `0.75`
- Candidate exact rate: `0.75`
- Delta exact rate: `0.0`
- Baseline executable rate: `0.75`
- Candidate executable rate: `0.75`
- Delta executable rate: `0.0`
- Baseline executor-equivalence rate: `0.75`
- Candidate executor-equivalence rate: `0.75`
- Delta executor-equivalence rate: `0.0`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h3_clinical_triage_priority_chip_memo_decoy | h3_mixed_workflow_instructional_negation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_finance_approval_banner_remembered_selection | h3_finance_stale_selection_paraphrase | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_finance_forecast_tile_leftover_evidence | h3_finance_stale_selection_paraphrase | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h3_finance_invoice_lock_archived_selector | h3_finance_stale_selection_paraphrase | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_finance_renewal_lane_retired_view | h3_finance_stale_selection_paraphrase | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h3_legal_contract_risk_badge_note_decoy | h3_mixed_workflow_instructional_negation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_ops_deployment_gate_panel_log_decoy | h3_mixed_workflow_instructional_negation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_policy_expired_policy_banner_value_first | h3_policy_label_order_inversion | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_policy_missing_evidence_field_value_first | h3_policy_label_order_inversion | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_policy_paused_state_pill_value_first | h3_policy_label_order_inversion | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_policy_rejected_approval_chip_value_first | h3_policy_label_order_inversion | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_recruiting_candidate_stage_tile_table_decoy | h3_mixed_workflow_instructional_negation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_research_ablation_tile_remembered_selection | h3_research_stale_selection_paraphrase | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_research_claim_panel_carryover_selection | h3_research_stale_selection_paraphrase | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h3_research_evidence_badge_retired_selection | h3_research_stale_selection_paraphrase | False | False | False | False | False | False | 1 | 1 | 0 | wrong_tool |
| h3_research_method_card_shadow_selection | h3_research_stale_selection_paraphrase | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_support_disabled_escalation_toggle | h3_support_negated_component_syntax | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_support_inactive_alert_banner | h3_support_negated_component_syntax | False | False | False | False | False | False | 1 | 1 | 0 | argument_mismatch |
| h3_support_unassigned_owner_field | h3_support_negated_component_syntax | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_support_unresolved_ticket_badge | h3_support_negated_component_syntax | True | True | True | True | True | True | 1 | 1 | 0 | exact |
