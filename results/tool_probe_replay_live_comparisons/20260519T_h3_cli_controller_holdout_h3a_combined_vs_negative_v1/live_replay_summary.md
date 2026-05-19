# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_h3a_negative_value_target_preservation`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined`
- Baseline exact rate: `0.8`
- Candidate exact rate: `1.0`
- Delta exact rate: `0.19999999999999996`
- Baseline executable rate: `0.8`
- Candidate executable rate: `1.0`
- Delta executable rate: `0.19999999999999996`
- Baseline executor-equivalence rate: `0.8`
- Candidate executor-equivalence rate: `1.0`
- Delta executor-equivalence rate: `0.19999999999999996`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline executor eq | candidate executor eq | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h3_clinical_triage_priority_chip_memo_decoy | h3_mixed_workflow_instructional_negation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_finance_approval_banner_remembered_selection | h3_finance_stale_selection_paraphrase | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_finance_forecast_tile_leftover_evidence | h3_finance_stale_selection_paraphrase | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h3_finance_invoice_lock_archived_selector | h3_finance_stale_selection_paraphrase | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_finance_renewal_lane_retired_view | h3_finance_stale_selection_paraphrase | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h3_legal_contract_risk_badge_note_decoy | h3_mixed_workflow_instructional_negation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_ops_deployment_gate_panel_log_decoy | h3_mixed_workflow_instructional_negation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_policy_expired_policy_banner_value_first | h3_policy_label_order_inversion | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_policy_missing_evidence_field_value_first | h3_policy_label_order_inversion | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_policy_paused_state_pill_value_first | h3_policy_label_order_inversion | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_policy_rejected_approval_chip_value_first | h3_policy_label_order_inversion | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_recruiting_candidate_stage_tile_table_decoy | h3_mixed_workflow_instructional_negation | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_research_ablation_tile_remembered_selection | h3_research_stale_selection_paraphrase | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_research_claim_panel_carryover_selection | h3_research_stale_selection_paraphrase | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h3_research_evidence_badge_retired_selection | h3_research_stale_selection_paraphrase | False | True | False | True | False | True | 1 | 1 | 0 | exact |
| h3_research_method_card_shadow_selection | h3_research_stale_selection_paraphrase | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_support_disabled_escalation_toggle | h3_support_negated_component_syntax | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_support_inactive_alert_banner | h3_support_negated_component_syntax | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_support_unassigned_owner_field | h3_support_negated_component_syntax | True | True | True | True | True | True | 1 | 1 | 0 | exact |
| h3_support_unresolved_ticket_badge | h3_support_negated_component_syntax | True | True | True | True | True | True | 1 | 1 | 0 | exact |
