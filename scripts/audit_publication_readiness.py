from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_DIR = ROOT / "results" / "reports" / "publication_evidence_ledger"
DEFAULT_TOOL_CONTRACT_REPORT_DIR = ROOT / "results" / "reports" / "mlx_tool_contract_harnessing"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "publication_readiness_audit"


def audit_publication_readiness(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ledger_dir: str | Path = DEFAULT_LEDGER_DIR,
    tool_contract_report_dir: str | Path = DEFAULT_TOOL_CONTRACT_REPORT_DIR,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ledger_dir = Path(ledger_dir)
    tool_contract_report_dir = Path(tool_contract_report_dir)
    ledger_payload = _read_json(ledger_dir / "ledger.json")
    ledger_manifest = ledger_payload.get("manifest", {}) if isinstance(ledger_payload, dict) else {}
    report_manifest = _read_json(tool_contract_report_dir / "manifest.json")

    checks = [
        _check_path(
            check_id="ledger_manifest_exists",
            severity="blocking",
            path=ledger_dir / "manifest.json",
            detail="Publication evidence ledger manifest exists.",
        ),
        _check_bool(
            check_id="ledger_has_no_missing_sources",
            severity="blocking",
            passed=int(ledger_manifest.get("missing_source_count", 1) or 0) == 0,
            detail=f"missing_source_count={ledger_manifest.get('missing_source_count', '')}",
        ),
        _check_bool(
            check_id="ledger_has_claims",
            severity="blocking",
            passed=int(ledger_manifest.get("claim_count", 0) or 0) >= 6,
            detail=f"claim_count={ledger_manifest.get('claim_count', '')}",
        ),
        _check_bool(
            check_id="ledger_includes_negative_results",
            severity="blocking",
            passed=_has_negative_result_claim(ledger_payload),
            detail="At least one claim is explicitly labeled as negative-result evidence.",
        ),
        _check_bool(
            check_id="tool_contract_report_has_current_tables",
            severity="blocking",
            passed=int(report_manifest.get("table_count", 0) or 0) >= 82,
            detail=f"table_count={report_manifest.get('table_count', '')}",
        ),
        _check_bool(
            check_id="tool_contract_report_has_current_figures",
            severity="blocking",
            passed=int(report_manifest.get("figure_count", 0) or 0) >= 39,
            detail=f"figure_count={report_manifest.get('figure_count', '')}",
        ),
        _check_path(
            check_id="v3_negative_probe_packet_exists",
            severity="blocking",
            path=ROOT / "results" / "tool_catalog_profile_probe_packets" / "20260508T_visual_role_catalog_split_selector_hints_v3_probe",
            detail="Negative v3 catalog-profile probe is preserved.",
        ),
        _check_path(
            check_id="v3_skipped_live_decision_exists",
            severity="blocking",
            path=ROOT / "results" / "tool_probe_replay_live" / "20260508T_visual_split_selector_hints_live_replay_skipped_v1" / "decision.md",
            detail="Skipped-live decision is preserved as an auditable packet.",
        ),
        _check_path(
            check_id="v4_negative_probe_packet_exists",
            severity="blocking",
            path=ROOT / "results" / "tool_catalog_profile_probe_packets" / "20260509T_visual_role_catalog_schema_field_hints_v4_probe",
            detail="Negative v4 schema-field probe is preserved.",
        ),
        _check_path(
            check_id="v4_skipped_live_decision_exists",
            severity="blocking",
            path=ROOT / "results" / "tool_probe_replay_live" / "20260509T_visual_schema_field_hints_live_replay_skipped_v1" / "decision.md",
            detail="Skipped-live decision is preserved as an auditable packet.",
        ),
        _check_path(
            check_id="visual_hard_slice_design_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_hard_slice_design" / "design.json",
            detail="Fresh visual hard-slice design packet exists before new benchmark execution.",
        ),
        _check_path(
            check_id="visual_hard_slice_execute_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "visual_hard_slice_probe_packets"
            / "20260509T_visual_hard_slice_executor_equivalence_v1"
            / "candidate_gate_summary.md",
            detail="Latest executed visual hard-slice packet exists with strict, executable, and executor-equivalence gate summary.",
        ),
        _check_path(
            check_id="visual_hard_slice_v5_vs_v4_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "visual_hard_slice_probe_packets"
            / "20260509T_visual_hard_slice_executor_equivalence_v1"
            / "schema_literal_targets_vs_schema_field_hints"
            / "probe_comparison.json",
            detail="Direct v5-vs-v4 comparison exists to preserve the negative target-literal result with executor-equivalence metrics.",
        ),
        _check_path(
            check_id="visual_hard_slice_exactness_diagnostic_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "visual_hard_slice_exactness_diagnostic"
            / "exactness_diagnostic.json",
            detail="Exactness-vs-executor diagnostic exists for interpreting v4 paraphrases and v5 regression.",
        ),
        _check_path(
            check_id="h1l_visual_executor_equivalence_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "knowledge_work_h1_slice"
            / "20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet"
            / "tool_contract_system_deltas.csv",
            detail="Executed H1l visual executor-equivalence packaged-workflow packet exists with system deltas.",
        ),
        _check_path(
            check_id="h1m_visual_alias_repeat_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "knowledge_work_h1_slice"
            / "20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet"
            / "tool_contract_system_deltas.csv",
            detail="Executed H1m visual alias-repeat packaged-workflow packet exists with system deltas.",
        ),
        _check_path(
            check_id="visual_hard_slice_live_replay_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_schema_field_hints_vs_no_directive_live_v2"
            / "live_replay_comparison.json",
            detail="Replay-shaped visual hard-slice CLI-live comparison exists with executor-equivalence deltas.",
        ),
        _check_path(
            check_id="visual_hard_slice_live_contracted_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_contracted_vs_no_directive_live_v1"
            / "live_replay_comparison.json",
            detail="Replay-shaped visual hard-slice CLI-live contracted upper-bound comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_live_role_catalog_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_role_catalog_vs_no_directive_live_v1"
            / "live_replay_comparison.json",
            detail="Replay-shaped visual hard-slice CLI-live role-catalog comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_live_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_argument_hints_vs_no_directive_live_v1"
            / "live_replay_comparison.json",
            detail="Replay-shaped visual hard-slice CLI-live argument-hints comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_live_schema_literals_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_schema_literal_targets_vs_no_directive_live_v1"
            / "live_replay_comparison.json",
            detail="Replay-shaped visual hard-slice CLI-live schema-target-literal comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_stress_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260509T_visual_hard_slice_live_stress_dry_run_v1"
            / "replay_cases.json",
            detail="Designed visual hard-slice stress replay packet exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_stress_schema_field_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_schema_field_hints_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Visual hard-slice stress schema-field live replay comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_stress_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "visual_hard_slice_stress_live_replay_summary.csv",
            detail="Visual hard-slice stress live replay summary table exists in the main report.",
        ),
        _check_path(
            check_id="visual_live_stress_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_live_stress_diagnostic" / "diagnostic.md",
            detail="Visual live stress diagnostic report exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_repeat_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1"
            / "replay_cases.json",
            detail="Eight-case alias-repeat visual stress replay packet exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_repeat_schema_field_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Alias-repeat schema-field live replay comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_repeat_contracted_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_alias_repeat_contracted_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Alias-repeat contracted live replay comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_repeat_role_catalog_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_alias_repeat_role_catalog_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Alias-repeat role-catalog live replay comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_repeat_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_alias_repeat_argument_hints_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Alias-repeat argument-hints live replay comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_repeat_schema_literals_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_alias_repeat_schema_literal_targets_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Alias-repeat schema-target-literal live replay comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_repeat_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "visual_hard_slice_alias_repeat_live_replay_summary.csv",
            detail="Alias-repeat live replay summary table exists in the main report.",
        ),
        _check_path(
            check_id="visual_alias_repeat_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_alias_repeat_diagnostic" / "diagnostic.md",
            detail="Visual alias-repeat diagnostic report exists.",
        ),
        _check_path(
            check_id="h1m_visual_alias_repeat_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h1m_visual_alias_repeat_candidate_metrics.csv",
            detail="H1m visual alias-repeat candidate metrics table exists in the main report.",
        ),
        _check_path(
            check_id="packaged_replay_gap_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "packaged_replay_gap_diagnostic" / "diagnostic.md",
            detail="Packaged replay gap diagnostic exists to compare replay gains with H1l/H1m packaged saturation.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_transfer_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1"
            / "replay_cases.json",
            detail="Six-case alias-transfer visual stress replay packet exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_transfer_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_alias_transfer_argument_hints_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Alias-transfer argument-hints live replay comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_transfer_contracted_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_alias_transfer_contracted_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Alias-transfer contracted live replay comparison exists.",
        ),
        _check_path(
            check_id="visual_hard_slice_alias_transfer_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "visual_hard_slice_alias_transfer_live_replay_summary.csv",
            detail="Alias-transfer live replay summary table exists in the main report.",
        ),
        _check_path(
            check_id="visual_alias_transfer_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_alias_transfer_diagnostic" / "diagnostic.md",
            detail="Visual alias-transfer diagnostic report exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_contract_split_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1n_alias_transfer_contract_split" / "diagnostic.md",
            detail="H1n contract-split diagnostic exists to separate planner exactness from executor-target success.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_oracle_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2"
            / "replay_cases.json",
            detail="Rebuilt H1n alias-transfer packet exists with oracle expected calls.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_oracle_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_argument_hints_vs_no_directive_v2"
            / "live_replay_comparison.json",
            detail="Oracle H1n argument-hints live replay comparison exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_oracle_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_alias_transfer_oracle_diagnostic" / "diagnostic.md",
            detail="Oracle H1n alias-transfer diagnostic report exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_oracle_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv",
            detail="Oracle H1n alias-transfer summary table exists in the main report.",
        ),
        _check_path(
            check_id="h1n_oracle_helper_ablation_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1n_oracle_helper_ablation" / "diagnostic.md",
            detail="H1n oracle argument-hints helper-ablation diagnostic exists.",
        ),
        _check_path(
            check_id="h1n_oracle_helper_ablation_no_repair_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_h1n_oracle_argument_hints_no_controller_repair_vs_argument_hints_v1"
            / "live_replay_comparison.json",
            detail="H1n oracle argument-hints no-controller-repair comparison exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_repeat_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1"
            / "replay_cases.json",
            detail="Fresh H1n oracle alias-transfer repeat packet exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_repeat_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_alias_transfer_repeat_diagnostic" / "diagnostic.md",
            detail="Fresh H1n oracle alias-transfer repeat diagnostic exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_repeat_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_h1n_oracle_repeat_argument_hints_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Fresh H1n repeat argument-hints comparison exists.",
        ),
        _check_path(
            check_id="h1n_oracle_transfer_synthesis_report_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1n_oracle_transfer_synthesis" / "report.md",
            detail="Two-packet H1n oracle-transfer synthesis report exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_oblique_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1"
            / "replay_cases.json",
            detail="Held-out H1n oblique-label oracle packet exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_oblique_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_alias_transfer_oblique_diagnostic" / "diagnostic.md",
            detail="Held-out H1n oblique-label diagnostic exists.",
        ),
        _check_path(
            check_id="h1n_alias_transfer_oblique_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_h1n_oracle_oblique_argument_hints_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Held-out H1n oblique-label argument-hints comparison exists.",
        ),
        _check_path(
            check_id="h1n_oblique_miss_analysis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1n_oblique_miss_analysis" / "diagnostic.md",
            detail="H1n oblique-label miss analysis exists.",
        ),
        _check_path(
            check_id="h1n_oblique_code_hints_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260509T_h1n_oracle_oblique_code_hints_execute_v1"
            / "summary.json",
            detail="H1n oblique-code-hints replay packet exists.",
        ),
        _check_path(
            check_id="h1n_oblique_code_hints_vs_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1"
            / "live_replay_comparison.json",
            detail="H1n oblique-code-hints comparison against argument hints exists.",
        ),
        _check_path(
            check_id="h1n_oblique_code_hints_delta_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1n_oblique_code_hints_delta" / "diagnostic.md",
            detail="H1n oblique-code-hints gain/regression diagnostic exists.",
        ),
        _check_path(
            check_id="h1n_code_hints_transfer_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_oracle_code_hints_transfer_execute_v1"
            / "summary.json",
            detail="Oblique-code profile replay on the earlier oracle packet exists.",
        ),
        _check_path(
            check_id="h1n_code_hints_repeat_transfer_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_oracle_repeat_code_hints_transfer_execute_v1"
            / "summary.json",
            detail="Oblique-code profile replay on the repeat oracle packet exists.",
        ),
        _check_path(
            check_id="h1n_code_hints_transfer_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1n_code_hints_transfer_synthesis" / "report.md",
            detail="Three-packet H1n oblique-code transfer synthesis exists.",
        ),
        _check_path(
            check_id="h1n_oblique_code_guard_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_oracle_oblique_code_guard_execute_v1"
            / "summary.json",
            detail="H1n oblique-code-guard replay packet exists.",
        ),
        _check_path(
            check_id="h1n_oblique_code_guard_vs_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h1n_oracle_oblique_code_guard_vs_argument_hints_v1"
            / "live_replay_comparison.json",
            detail="H1n oblique-code-guard comparison against argument hints exists.",
        ),
        _check_path(
            check_id="h1n_oblique_code_guard_vs_code_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h1n_oracle_oblique_code_guard_vs_code_hints_v1"
            / "live_replay_comparison.json",
            detail="H1n oblique-code-guard comparison against v6 code hints exists.",
        ),
        _check_path(
            check_id="h1n_code_guard_transfer_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_oracle_code_guard_transfer_execute_v1"
            / "summary.json",
            detail="Code-guard transfer replay on the earlier oracle packet exists.",
        ),
        _check_path(
            check_id="h1n_code_guard_repeat_transfer_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_oracle_repeat_code_guard_transfer_execute_v1"
            / "summary.json",
            detail="Code-guard transfer replay on the repeat oracle packet exists.",
        ),
        _check_path(
            check_id="h1n_code_guard_transfer_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1n_code_guard_transfer_synthesis" / "report.md",
            detail="Three-packet H1n code-guard transfer synthesis exists.",
        ),
        _check_path(
            check_id="h1n_post_repair_holdout_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1"
            / "summary.json",
            detail="Fresh H1n post-repair holdout packet exists.",
        ),
        _check_path(
            check_id="h1n_post_repair_code_guard_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_post_repair_code_guard_execute_v1"
            / "summary.json",
            detail="Code-guard execution on the post-repair holdout exists.",
        ),
        _check_path(
            check_id="h1n_post_repair_code_guard_vs_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h1n_post_repair_code_guard_vs_argument_hints_v1"
            / "live_replay_comparison.json",
            detail="Post-repair code-guard comparison against argument hints exists.",
        ),
        _check_path(
            check_id="h1n_post_repair_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_alias_transfer_post_repair_diagnostic" / "diagnostic.md",
            detail="Post-repair visual alias-transfer diagnostic exists.",
        ),
        _check_path(
            check_id="h1n_post_repair_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "visual_hard_slice_post_repair_live_replay_summary.csv",
            detail="Paper-facing post-repair holdout summary table exists.",
        ),
        _check_path(
            check_id="h1n_residual_holdout_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1"
            / "summary.json",
            detail="Fresh H1n residual holdout packet exists.",
        ),
        _check_path(
            check_id="h1n_residual_hybrid_label_guard_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_residual_hybrid_label_guard_execute_v1"
            / "summary.json",
            detail="Hybrid label guard execution on the residual holdout exists.",
        ),
        _check_path(
            check_id="h1n_residual_hybrid_vs_code_guard_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h1n_residual_hybrid_label_guard_vs_code_guard_v1"
            / "live_replay_comparison.json",
            detail="Residual hybrid label guard comparison against v7 code guard exists.",
        ),
        _check_path(
            check_id="h1n_residual_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_alias_transfer_residual_diagnostic" / "diagnostic.md",
            detail="Residual visual alias-transfer diagnostic exists.",
        ),
        _check_path(
            check_id="h1n_residual_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "visual_hard_slice_residual_live_replay_summary.csv",
            detail="Paper-facing residual holdout summary table exists.",
        ),
        _check_path(
            check_id="h1n_component_value_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260510T_visual_hard_slice_component_value_oracle_dry_run_v1"
            / "summary.json",
            detail="Component-role/value holdout packet exists.",
        ),
        _check_path(
            check_id="h1n_component_value_v9_live_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_component_value_component_value_guard_execute_v1"
            / "summary.json",
            detail="Component-value v9 live replay packet exists.",
        ),
        _check_path(
            check_id="h1n_component_value_v10_live_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1n_component_value_no_call_control_rescue_execute_v1"
            / "summary.json",
            detail="Component-value v10 no-call control rescue live replay packet exists.",
        ),
        _check_path(
            check_id="h1n_component_value_v10_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h1n_component_value_no_call_control_rescue_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="Component-value v10 no-call control rescue comparison against no-directive exists.",
        ),
        _check_path(
            check_id="h1n_component_value_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_component_value_diagnostic" / "diagnostic.md",
            detail="Component-value visual diagnostic exists.",
        ),
        _check_path(
            check_id="h1n_component_value_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "visual_hard_slice_component_value_live_replay_summary.csv",
            detail="Paper-facing component-value holdout summary table exists.",
        ),
        _check_path(
            check_id="h1n_no_call_rescue_transfer_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1n_no_call_rescue_transfer_synthesis" / "report.md",
            detail="No-call rescue transfer synthesis report exists.",
        ),
        _check_path(
            check_id="h1o_control_factorial_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260510T_h1o_control_factorial_oracle_dry_run_v1"
            / "summary.json",
            detail="H1o control-factorial oracle packet exists.",
        ),
        _check_path(
            check_id="h1o_control_factorial_argument_hints_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1o_control_factorial_argument_hints_execute_v1"
            / "summary.json",
            detail="H1o argument-hints live replay exists.",
        ),
        _check_path(
            check_id="h1o_control_factorial_component_value_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1o_control_factorial_component_value_guard_execute_v1"
            / "summary.json",
            detail="H1o component-value-guard live replay exists.",
        ),
        _check_path(
            check_id="h1o_control_factorial_argument_hints_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h1o_control_factorial_argument_hints_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="H1o argument-hints comparison against no-directive exists.",
        ),
        _check_path(
            check_id="h1o_control_factorial_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_h1o_control_factorial_diagnostic" / "diagnostic.md",
            detail="H1o control-factorial matrix diagnostic exists.",
        ),
        _check_path(
            check_id="h1o_control_factorial_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1o_control_factorial_synthesis" / "report.md",
            detail="H1o mechanism-family synthesis report exists.",
        ),
        _check_path(
            check_id="h1p_component_value_holdout_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260510T_h1p_component_value_holdout_oracle_dry_run_v1"
            / "summary.json",
            detail="H1p component-value holdout oracle packet exists.",
        ),
        _check_path(
            check_id="h1p_component_value_no_directive_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1p_component_value_no_directive_execute_v1"
            / "summary.json",
            detail="H1p no-directive live replay exists.",
        ),
        _check_path(
            check_id="h1p_component_value_component_guard_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1p_component_value_component_value_guard_execute_v1"
            / "summary.json",
            detail="H1p component-value-guard live replay exists.",
        ),
        _check_path(
            check_id="h1p_component_value_component_guard_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h1p_component_value_component_value_guard_vs_no_directive_v1"
            / "live_replay_comparison.json",
            detail="H1p component-value-guard comparison against no-directive exists.",
        ),
        _check_path(
            check_id="h1p_component_value_diagnostic_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "visual_h1p_component_value_diagnostic" / "diagnostic.md",
            detail="H1p component-value matrix diagnostic exists.",
        ),
        _check_path(
            check_id="h1p_component_value_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "visual_hard_slice_h1p_live_replay_summary.csv",
            detail="Paper-facing H1p summary table exists.",
        ),
        _check_path(
            check_id="h1q_component_label_guard_h1n_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1"
            / "summary.json",
            detail="H1q component-label guard H1n live replay exists.",
        ),
        _check_path(
            check_id="h1q_component_label_guard_h1o_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1"
            / "summary.json",
            detail="H1q component-label guard H1o live replay exists.",
        ),
        _check_path(
            check_id="h1q_component_label_guard_h1p_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1"
            / "summary.json",
            detail="H1q component-label guard H1p live replay exists.",
        ),
        _check_path(
            check_id="h1q_component_label_guard_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1q_component_label_guard_transfer_synthesis" / "report.md",
            detail="H1q transfer synthesis report exists.",
        ),
        _check_path(
            check_id="h1q_component_label_guard_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h1q_component_label_guard_aggregate_summary.csv",
            detail="Paper-facing H1q aggregate table exists in the main report.",
        ),
        _check_path(
            check_id="h1s_component_residual_h1n_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1"
            / "summary.json",
            detail="H1s component-residual guard H1n transfer live replay exists.",
        ),
        _check_path(
            check_id="h1s_component_residual_h1o_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1"
            / "summary.json",
            detail="H1s component-residual guard H1o transfer live replay exists.",
        ),
        _check_path(
            check_id="h1s_component_residual_h1p_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1"
            / "summary.json",
            detail="H1s component-residual guard H1p transfer live replay exists.",
        ),
        _check_path(
            check_id="h1s_component_residual_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1s_component_residual_transfer_synthesis" / "report.md",
            detail="H1s transfer-gate synthesis report exists.",
        ),
        _check_path(
            check_id="h1s_component_residual_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h1s_component_residual_transfer_aggregate.csv",
            detail="Paper-facing H1s aggregate table exists in the main report.",
        ),
        _check_path(
            check_id="h1x_v11_breaker_no_directive_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1x_v11_breaker_no_directive_execute_v1"
            / "summary.json",
            detail="H1x no-directive live replay exists.",
        ),
        _check_path(
            check_id="h1x_v11_breaker_v11_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1x_v11_breaker_component_label_guard_execute_v1"
            / "summary.json",
            detail="H1x v11 component-label guard live replay exists.",
        ),
        _check_path(
            check_id="h1x_v11_breaker_v12_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1x_v11_breaker_component_residual_guard_execute_v1"
            / "summary.json",
            detail="H1x v12 component-residual guard live replay exists.",
        ),
        _check_path(
            check_id="h1x_v11_breaker_v15_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1x_v11_breaker_code_label_exact_guard_execute_v1"
            / "summary.json",
            detail="H1x v15 code-label exact guard live replay exists.",
        ),
        _check_path(
            check_id="h1x_v11_breaker_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1x_v11_breaker_synthesis" / "report.md",
            detail="H1x v11-breaker synthesis report exists.",
        ),
        _check_path(
            check_id="h1x_v11_breaker_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h1x_v11_breaker_packet_summary.csv",
            detail="Paper-facing H1x packet table exists in the main report.",
        ),
        _check_path(
            check_id="h1x_v11_breaker_report_figure_exists",
            severity="recommended",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "figures"
            / "h1x_v11_breaker_gate.svg",
            detail="Paper-facing H1x replay gate figure exists in the main report.",
        ),
        _check_path(
            check_id="h1y_routed_residual_no_directive_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1y_routed_residual_no_directive_execute_v1"
            / "summary.json",
            detail="H1y no-directive live replay exists.",
        ),
        _check_path(
            check_id="h1y_routed_residual_v12_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h1y_routed_residual_component_residual_guard_execute_v1"
            / "summary.json",
            detail="H1y v12 component-residual guard live replay exists.",
        ),
        _check_path(
            check_id="h2a_visual_stale_selection_gate_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2a_visual_stale_selection_gate_on_h1y_execute_v1"
            / "summary.json",
            detail="H2a controller stale-selection gate live replay exists.",
        ),
        _check_path(
            check_id="h2a_visual_stale_selection_gate_vs_v11_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h2a_visual_stale_selection_gate_vs_component_label_guard_on_h1y_v1"
            / "live_replay_comparison.json",
            detail="H2a-vs-v11 live comparison exists.",
        ),
        _check_path(
            check_id="h2a_visual_stale_selection_gate_vs_v12_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h2a_visual_stale_selection_gate_vs_component_residual_guard_on_h1y_v1"
            / "live_replay_comparison.json",
            detail="H2a-vs-v12 live comparison exists.",
        ),
        _check_path(
            check_id="h1y_routed_residual_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h1y_routed_residual_synthesis" / "report.md",
            detail="H1y/H2a routed-residual synthesis report exists.",
        ),
        _check_path(
            check_id="h1y_routed_residual_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h1y_routed_residual_packet_summary.csv",
            detail="Paper-facing H1y/H2a packet table exists in the main report.",
        ),
        _check_path(
            check_id="h1y_routed_residual_report_figure_exists",
            severity="recommended",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "figures"
            / "h1y_routed_residual_gate.svg",
            detail="Paper-facing H1y/H2a replay gate figure exists in the main report.",
        ),
        _check_path(
            check_id="h2a_transfer_h1n_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2a_visual_stale_selection_gate_on_h1n_component_value_execute_v1"
            / "summary.json",
            detail="H2a transfer replay exists for H1n component-value residual.",
        ),
        _check_path(
            check_id="h2a_transfer_h1o_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2a_visual_stale_selection_gate_on_h1o_execute_v1"
            / "summary.json",
            detail="H2a transfer replay exists for H1o control-factorial.",
        ),
        _check_path(
            check_id="h2a_transfer_h1p_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2a_visual_stale_selection_gate_on_h1p_execute_v1"
            / "summary.json",
            detail="H2a transfer replay exists for H1p component-value holdout.",
        ),
        _check_path(
            check_id="h2a_transfer_h1x_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1"
            / "summary.json",
            detail="H2a transfer replay exists for H1x v11-breaker.",
        ),
        _check_path(
            check_id="h2a_stale_selection_transfer_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h2a_stale_selection_transfer_synthesis" / "report.md",
            detail="H2a stale-selection transfer synthesis report exists.",
        ),
        _check_path(
            check_id="h2a_stale_selection_transfer_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h2a_stale_selection_transfer_aggregate_summary.csv",
            detail="Paper-facing H2a transfer aggregate table exists in the main report.",
        ),
        _check_path(
            check_id="h2a_stale_selection_transfer_residual_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h2a_stale_selection_transfer_residual_rows.csv",
            detail="Paper-facing H2a residual table exists in the main report.",
        ),
        _check_path(
            check_id="h2a_stale_selection_transfer_report_figure_exists",
            severity="recommended",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "figures"
            / "h2a_stale_selection_transfer_gate.svg",
            detail="Paper-facing H2a transfer gate figure exists in the main report.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260510T_h2b_residual_exactness_dry_run_v1"
            / "manifest.json",
            detail="H2b residual exactness dry-run packet exists.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_no_directive_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2b_residual_exactness_no_directive_execute_v1"
            / "summary.json",
            detail="H2b no-directive live replay exists.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_v11_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2b_residual_exactness_component_label_guard_execute_v1"
            / "summary.json",
            detail="H2b v11 component-label live replay exists.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_v12_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2b_residual_exactness_component_residual_guard_execute_v1"
            / "summary.json",
            detail="H2b v12 component-residual live replay exists.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_v15_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2b_residual_exactness_code_label_exact_guard_execute_v1"
            / "summary.json",
            detail="H2b v15 code-label live replay exists.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_h2a_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2b_residual_exactness_h2a_execute_v1"
            / "summary.json",
            detail="H2b H2a stale-selection live replay exists.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_v9_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2b_residual_exactness_component_value_guard_execute_v1"
            / "summary.json",
            detail="H2b v9 component-value live replay exists.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h2b_residual_exactness_synthesis" / "report.md",
            detail="H2b residual exactness synthesis report exists.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h2b_residual_exactness_packet_summary.csv",
            detail="Paper-facing H2b packet table exists in the main report.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_case_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h2b_residual_exactness_case_matrix.csv",
            detail="Paper-facing H2b case matrix exists in the main report.",
        ),
        _check_path(
            check_id="h2b_residual_exactness_report_figure_exists",
            severity="recommended",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "figures"
            / "h2b_residual_exactness_gate.svg",
            detail="Paper-facing H2b residual exactness gate figure exists in the main report.",
        ),
        _check_path(
            check_id="h2c_scoped_residual_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1"
            / "summary.json",
            detail="H2c scoped residual live replay exists on H2b.",
        ),
        _check_path(
            check_id="h2c_scoped_residual_vs_v12_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h2c_scoped_residual_gate_vs_component_residual_guard_on_h2b_v1"
            / "live_replay_comparison.json",
            detail="H2c-vs-v12 H2b comparison exists.",
        ),
        _check_path(
            check_id="h2c_scoped_residual_vs_h2a_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260510T_h2c_scoped_residual_gate_vs_h2a_on_h2b_v1"
            / "live_replay_comparison.json",
            detail="H2c-vs-H2a H2b comparison exists.",
        ),
        _check_path(
            check_id="h2c_scoped_residual_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h2c_scoped_residual_synthesis" / "report.md",
            detail="H2c scoped residual synthesis report exists.",
        ),
        _check_path(
            check_id="h2c_scoped_residual_report_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h2c_scoped_residual_packet_summary.csv",
            detail="Paper-facing H2c packet table exists in the main report.",
        ),
        _check_path(
            check_id="h2c_scoped_residual_comparison_table_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "tables"
            / "h2c_scoped_residual_comparison_summary.csv",
            detail="Paper-facing H2c comparison table exists in the main report.",
        ),
        _check_path(
            check_id="h2c_scoped_residual_report_figure_exists",
            severity="recommended",
            path=ROOT
            / "results"
            / "reports"
            / "mlx_tool_contract_harnessing"
            / "figures"
            / "h2c_scoped_residual_gate.svg",
            detail="Paper-facing H2c scoped residual gate figure exists in the main report.",
        ),
        _check_path(
            check_id="h2l_target_normalization_overreach_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260512T_h2l_target_normalization_overreach_dry_run_v1"
            / "replay_cases.json",
            detail="H2l target-normalization overreach dry-run packet exists.",
        ),
        _check_path(
            check_id="h2l_target_normalization_overreach_h2e_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2l_target_normalization_overreach_h2e_execute_v1"
            / "summary.json",
            detail="H2l H2e live replay exists.",
        ),
        _check_path(
            check_id="h2l_target_normalization_overreach_h2j_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2l_target_normalization_overreach_h2j_execute_v1"
            / "summary.json",
            detail="H2l full-H2j live replay exists.",
        ),
        _check_path(
            check_id="h2l_target_normalization_overreach_no_stale_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2l_target_normalization_overreach_h2j_no_stale_gate_execute_v1"
            / "summary.json",
            detail="H2l H2j without stale-selection gate live replay exists.",
        ),
        _check_path(
            check_id="h2l_target_normalization_overreach_h2j_vs_h2e_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2l_target_normalization_overreach_h2j_vs_h2e_v1"
            / "live_replay_comparison.json",
            detail="H2l full-H2j versus H2e comparison exists.",
        ),
        _check_path(
            check_id="h2l_target_normalization_overreach_h2j_vs_no_stale_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2l_target_normalization_overreach_h2j_vs_no_stale_gate_v1"
            / "live_replay_comparison.json",
            detail="H2l full-H2j versus no-stale-gate comparison exists.",
        ),
        _check_path(
            check_id="h2l_target_normalization_overreach_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h2l_target_normalization_overreach_synthesis" / "report.md",
            detail="H2l target-normalization overreach synthesis report exists.",
        ),
        _check_path(
            check_id="h2l_target_normalization_overreach_report_figure_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "h2l_target_normalization_overreach_synthesis"
            / "figures"
            / "h2l_target_normalization_overreach_gate.svg",
            detail="H2l target-normalization overreach figure exists.",
        ),
        _check_path(
            check_id="h2m_less_direct_overreach_packet_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_packets"
            / "20260512T_h2m_less_direct_target_normalization_overreach_dry_run_v1"
            / "replay_cases.json",
            detail="H2m less-direct target-normalization overreach dry-run packet exists.",
        ),
        _check_path(
            check_id="h2m_less_direct_overreach_h2e_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2m_less_direct_target_normalization_overreach_h2e_execute_v1"
            / "summary.json",
            detail="H2m H2e live replay exists.",
        ),
        _check_path(
            check_id="h2m_less_direct_overreach_h2j_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_execute_v1"
            / "summary.json",
            detail="H2m full-H2j live replay exists.",
        ),
        _check_path(
            check_id="h2m_less_direct_overreach_no_stale_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_no_stale_gate_execute_v1"
            / "summary.json",
            detail="H2m H2j without stale-selection gate live replay exists.",
        ),
        _check_path(
            check_id="h2m_less_direct_overreach_h2j_vs_h2e_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_h2e_v1"
            / "live_replay_comparison.json",
            detail="H2m full-H2j versus H2e comparison exists.",
        ),
        _check_path(
            check_id="h2m_less_direct_overreach_h2j_vs_no_stale_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_no_stale_gate_v1"
            / "live_replay_comparison.json",
            detail="H2m full-H2j versus no-stale-gate comparison exists.",
        ),
        _check_path(
            check_id="h2m_less_direct_overreach_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h2m_less_direct_overreach_synthesis" / "report.md",
            detail="H2m less-direct target-normalization overreach synthesis report exists.",
        ),
        _check_path(
            check_id="h2m_less_direct_overreach_report_figure_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "h2m_less_direct_overreach_synthesis"
            / "figures"
            / "h2m_less_direct_overreach_gate.svg",
            detail="H2m less-direct target-normalization overreach figure exists.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_h2m_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2n_scoped_target_normalization_on_h2m_execute_v1"
            / "summary.json",
            detail="H2n scoped target-normalization live replay exists on H2m.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_h2k_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2n_scoped_target_normalization_on_h2k_execute_v1"
            / "summary.json",
            detail="H2n scoped target-normalization transfer live replay exists on H2k.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_h2l_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2n_scoped_target_normalization_on_h2l_execute_v1"
            / "summary.json",
            detail="H2n scoped target-normalization transfer live replay exists on H2l.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_h2f_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2n_scoped_target_normalization_on_h2f_execute_v1"
            / "summary.json",
            detail="H2n scoped target-normalization transfer live replay exists on H2f.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_vs_h2j_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2n versus H2j comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_vs_h2e_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2n_scoped_target_normalization_vs_h2e_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2n versus H2e comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_vs_h2j_h2k_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2k_v1"
            / "live_replay_comparison.json",
            detail="H2n versus H2j transfer comparison exists on H2k.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_vs_h2j_h2l_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2l_v1"
            / "live_replay_comparison.json",
            detail="H2n versus H2j transfer comparison exists on H2l.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_vs_h2j_h2f_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2f_v1"
            / "live_replay_comparison.json",
            detail="H2n versus H2j transfer comparison exists on H2f.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_synthesis_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h2n_scoped_target_normalization_synthesis" / "report.md",
            detail="H2n scoped target-normalization synthesis report exists.",
        ),
        _check_path(
            check_id="h2n_scoped_target_normalization_report_figure_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "h2n_scoped_target_normalization_synthesis"
            / "figures"
            / "h2n_scoped_target_normalization_gate.svg",
            detail="H2n scoped target-normalization figure exists.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_h2m_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2o_value_bearing_target_synthesis_on_h2m_execute_v1"
            / "summary.json",
            detail="H2o value-bearing target synthesis live replay exists on H2m.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_h2k_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2o_value_bearing_target_synthesis_on_h2k_execute_v1"
            / "summary.json",
            detail="H2o value-bearing target synthesis transfer live replay exists on H2k.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_h2l_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2o_value_bearing_target_synthesis_on_h2l_execute_v1"
            / "summary.json",
            detail="H2o value-bearing target synthesis transfer live replay exists on H2l.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_h2f_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2o_value_bearing_target_synthesis_on_h2f_execute_v1"
            / "summary.json",
            detail="H2o value-bearing target synthesis transfer live replay exists on H2f.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_vs_h2n_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2o_value_bearing_target_synthesis_vs_h2n_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2o versus H2n comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_vs_h2j_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2o versus H2j comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_vs_h2e_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2o_value_bearing_target_synthesis_vs_h2e_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2o versus H2e comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_vs_h2j_h2k_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2k_v1"
            / "live_replay_comparison.json",
            detail="H2o versus H2j transfer comparison exists on H2k.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_vs_h2j_h2l_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2l_v1"
            / "live_replay_comparison.json",
            detail="H2o versus H2j transfer comparison exists on H2l.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_vs_h2j_h2f_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2f_v1"
            / "live_replay_comparison.json",
            detail="H2o versus H2j transfer comparison exists on H2f.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_report_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h2o_value_bearing_target_synthesis" / "report.md",
            detail="H2o value-bearing target synthesis report exists.",
        ),
        _check_path(
            check_id="h2o_value_bearing_target_synthesis_report_figure_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "h2o_value_bearing_target_synthesis"
            / "figures"
            / "h2o_value_bearing_target_synthesis_gate.svg",
            detail="H2o value-bearing target synthesis figure exists.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_h2m_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2p_contextual_surface_alias_routing_on_h2m_execute_v1"
            / "summary.json",
            detail="H2p contextual surface-alias routing live replay exists on H2m.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_h2k_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2p_contextual_surface_alias_routing_on_h2k_execute_v1"
            / "summary.json",
            detail="H2p contextual surface-alias routing transfer live replay exists on H2k.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_h2l_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2p_contextual_surface_alias_routing_on_h2l_execute_v1"
            / "summary.json",
            detail="H2p contextual surface-alias routing transfer live replay exists on H2l.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_h2f_live_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live"
            / "20260512T_h2p_contextual_surface_alias_routing_on_h2f_execute_v1"
            / "summary.json",
            detail="H2p contextual surface-alias routing transfer live replay exists on H2f.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_vs_h2o_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2p versus H2o comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_vs_h2n_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2p_contextual_surface_alias_routing_vs_h2n_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2p versus H2n comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_vs_h2j_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2p_contextual_surface_alias_routing_vs_h2j_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2p versus H2j comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_vs_h2e_h2m_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2p_contextual_surface_alias_routing_vs_h2e_on_h2m_v1"
            / "live_replay_comparison.json",
            detail="H2p versus H2e comparison exists on H2m.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_vs_h2o_h2k_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2k_v1"
            / "live_replay_comparison.json",
            detail="H2p versus H2o transfer comparison exists on H2k.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_vs_h2o_h2l_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2l_v1"
            / "live_replay_comparison.json",
            detail="H2p versus H2o transfer comparison exists on H2l.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_vs_h2o_h2f_comparison_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "tool_probe_replay_live_comparisons"
            / "20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2f_v1"
            / "live_replay_comparison.json",
            detail="H2p versus H2o transfer comparison exists on H2f.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_report_exists",
            severity="blocking",
            path=ROOT / "results" / "reports" / "h2p_contextual_surface_alias_routing_synthesis" / "report.md",
            detail="H2p contextual surface-alias routing report exists.",
        ),
        _check_path(
            check_id="h2p_contextual_surface_alias_routing_report_figure_exists",
            severity="blocking",
            path=ROOT
            / "results"
            / "reports"
            / "h2p_contextual_surface_alias_routing_synthesis"
            / "figures"
            / "h2p_contextual_surface_alias_routing_gate.svg",
            detail="H2p contextual surface-alias routing figure exists.",
        ),
        _check_path(
            check_id="current_state_doc_exists",
            severity="blocking",
            path=ROOT / "docs" / "continuity" / "current-state.md",
            detail="Continuity current-state doc exists.",
        ),
        _check_path(
            check_id="next_steps_doc_exists",
            severity="blocking",
            path=ROOT / "docs" / "continuity" / "next-steps.md",
            detail="Continuity next-steps doc exists.",
        ),
        _check_path(
            check_id="research_log_exists",
            severity="blocking",
            path=ROOT / "docs" / "research-log.md",
            detail="Research log exists.",
        ),
        _check_path(
            check_id="paper_outline_exists",
            severity="recommended",
            path=ROOT / "docs" / "paper" / "moonie-gemma-harnessing-paper-outline.md",
            detail="Paper outline exists for publication drafting.",
        ),
        _check_path(
            check_id="methodology_doc_exists",
            severity="recommended",
            path=ROOT / "docs" / "methodology.md",
            detail="Methodology doc exists.",
        ),
    ]
    for script_name in [
        "build_mlx_tool_contract_report.py",
        "build_publication_evidence_ledger.py",
        "audit_publication_readiness.py",
        "run_tool_catalog_profile_probe_packet.py",
        "run_visual_hard_slice_probe.py",
        "run_visual_hard_slice_probe_packet.py",
        "analyze_visual_hard_slice_exactness.py",
        "analyze_packaged_replay_gap.py",
        "analyze_h1n_alias_transfer_contract_split.py",
        "analyze_h1n_oracle_helper_ablation.py",
        "analyze_h1n_oblique_misses.py",
        "analyze_h1n_oblique_code_hints_delta.py",
        "build_h1n_code_hints_transfer_synthesis.py",
        "build_h1n_code_guard_transfer_synthesis.py",
        "build_h1n_no_call_rescue_transfer_synthesis.py",
        "build_h1o_control_factorial_synthesis.py",
        "build_h1q_component_label_guard_transfer_synthesis.py",
        "build_h1s_component_residual_transfer_synthesis.py",
        "build_h2a_stale_selection_transfer_synthesis.py",
        "build_h2b_residual_exactness_packet.py",
        "build_h2b_residual_exactness_synthesis.py",
        "build_h2c_scoped_residual_synthesis.py",
        "build_h1n_oracle_transfer_synthesis.py",
        "build_h2k_target_decoy_overlap_synthesis.py",
        "build_h2l_target_normalization_overreach_synthesis.py",
        "build_h2m_less_direct_overreach_synthesis.py",
        "build_h2n_scoped_target_normalization_synthesis.py",
        "build_h2o_value_bearing_target_synthesis.py",
        "build_h2p_contextual_surface_alias_routing_synthesis.py",
        "compare_tool_directive_probes.py",
        "build_visual_hard_slice_design.py",
        "build_visual_hard_slice_replay_packet.py",
    ]:
        checks.append(
            _check_path(
                check_id=f"script_{script_name}_exists",
                severity="blocking",
                path=ROOT / "scripts" / script_name,
                detail=f"Reproduction script `{script_name}` exists.",
            )
        )

    blocking_checks = [row for row in checks if row["severity"] == "blocking"]
    recommended_checks = [row for row in checks if row["severity"] == "recommended"]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "check_count": len(checks),
        "blocking_check_count": len(blocking_checks),
        "blocking_failed_count": sum(1 for row in blocking_checks if not row["passed"]),
        "recommended_check_count": len(recommended_checks),
        "recommended_failed_count": sum(1 for row in recommended_checks if not row["passed"]),
    }
    summary["blocking_passed"] = summary["blocking_failed_count"] == 0
    summary["readiness_level"] = "paper_draft_ready" if summary["blocking_passed"] else "not_ready"

    _write_csv(tables_dir / "publication_readiness_checks.csv", checks)
    payload = {"summary": summary, "checks": checks}
    (output / "publication_readiness_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "publication_readiness_audit.md").write_text(_markdown(summary, checks), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether Moonie has enough packet-backed evidence for a paper draft.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ledger-dir", default=str(DEFAULT_LEDGER_DIR))
    parser.add_argument("--tool-contract-report-dir", default=str(DEFAULT_TOOL_CONTRACT_REPORT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_publication_readiness(
        output_dir=args.output_dir,
        ledger_dir=args.ledger_dir,
        tool_contract_report_dir=args.tool_contract_report_dir,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _has_negative_result_claim(ledger_payload: dict[str, Any]) -> bool:
    claims = ledger_payload.get("claims", []) if isinstance(ledger_payload, dict) else []
    return any(str(row.get("status", "")).startswith("negative_result") for row in claims if isinstance(row, dict))


def _check_path(*, check_id: str, severity: str, path: Path, detail: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "passed": path.exists(),
        "detail": detail,
        "path": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
    }


def _check_bool(*, check_id: str, severity: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "passed": passed,
        "detail": detail,
        "path": "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any], checks: list[dict[str, Any]]) -> str:
    lines = [
        "# Publication Readiness Audit",
        "",
        f"- readiness_level: `{summary['readiness_level']}`",
        f"- blocking_passed: `{summary['blocking_passed']}`",
        f"- blocking_failed_count: `{summary['blocking_failed_count']}`",
        f"- recommended_failed_count: `{summary['recommended_failed_count']}`",
        "",
        "| Check | Severity | Passed | Detail | Path |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in checks:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["check_id"]),
                    str(row["severity"]),
                    str(row["passed"]),
                    str(row["detail"]),
                    str(row["path"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
