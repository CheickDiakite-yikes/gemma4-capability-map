# Publication Readiness Audit

- readiness_level: `paper_draft_ready`
- blocking_passed: `True`
- blocking_failed_count: `0`
- recommended_failed_count: `0`

| Check | Severity | Passed | Detail | Path |
| --- | --- | ---: | --- | --- |
| ledger_manifest_exists | blocking | True | Publication evidence ledger manifest exists. | results/reports/publication_evidence_ledger/manifest.json |
| ledger_has_no_missing_sources | blocking | True | missing_source_count=0 |  |
| ledger_has_claims | blocking | True | claim_count=6 |  |
| ledger_includes_negative_results | blocking | True | At least one claim is explicitly labeled as negative-result evidence. |  |
| tool_contract_report_has_current_tables | blocking | True | table_count=45 |  |
| tool_contract_report_has_current_figures | blocking | True | figure_count=25 |  |
| v3_negative_probe_packet_exists | blocking | True | Negative v3 catalog-profile probe is preserved. | results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe |
| v3_skipped_live_decision_exists | blocking | True | Skipped-live decision is preserved as an auditable packet. | results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1/decision.md |
| current_state_doc_exists | blocking | True | Continuity current-state doc exists. | docs/continuity/current-state.md |
| next_steps_doc_exists | blocking | True | Continuity next-steps doc exists. | docs/continuity/next-steps.md |
| research_log_exists | blocking | True | Research log exists. | docs/research-log.md |
| paper_outline_exists | recommended | True | Paper outline exists for publication drafting. | docs/paper/moonie-gemma-harnessing-paper-outline.md |
| methodology_doc_exists | recommended | True | Methodology doc exists. | docs/methodology.md |
| script_build_mlx_tool_contract_report.py_exists | blocking | True | Reproduction script `build_mlx_tool_contract_report.py` exists. | scripts/build_mlx_tool_contract_report.py |
| script_build_publication_evidence_ledger.py_exists | blocking | True | Reproduction script `build_publication_evidence_ledger.py` exists. | scripts/build_publication_evidence_ledger.py |
| script_audit_publication_readiness.py_exists | blocking | True | Reproduction script `audit_publication_readiness.py` exists. | scripts/audit_publication_readiness.py |
| script_run_tool_catalog_profile_probe_packet.py_exists | blocking | True | Reproduction script `run_tool_catalog_profile_probe_packet.py` exists. | scripts/run_tool_catalog_profile_probe_packet.py |
| script_compare_tool_directive_probes.py_exists | blocking | True | Reproduction script `compare_tool_directive_probes.py` exists. | scripts/compare_tool_directive_probes.py |
