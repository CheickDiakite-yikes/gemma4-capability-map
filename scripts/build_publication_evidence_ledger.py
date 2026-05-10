from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "publication_evidence_ledger"


@dataclass(frozen=True)
class EvidenceSource:
    artifact_type: str
    path: str
    purpose: str


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim: str
    status: str
    evidence_strength: str
    primary_metric: str
    limitation: str
    next_test: str
    sources: tuple[EvidenceSource, ...]


CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="C1_controller_dependence_hidden_by_readiness",
        claim="Top-line readiness parity can hide controller dependence in local MLX Gemma tool-use runs.",
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric="H1h/H1i no-directive readiness parity with high repair/fallback burden and low raw-clean rate.",
        limitation="Current support is internal to Moonie's knowledge-work harness and local MLX runtime.",
        next_test="Run the same helper-ablation structure on a harder H1 slice selected from raw replay failures.",
        sources=(
            EvidenceSource(
                "h1_ablation_packet",
                "results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1",
                "Full no-directive replication showing controller burden behind readiness parity.",
            ),
            EvidenceSource(
                "h1_ablation_packet",
                "results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1",
                "Fast worst-family loop preserving the H1h causal ordering.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/packet_summary.csv",
                "Cross-packet readiness, repair, fallback, argument-repair, and raw-clean summary.",
            ),
        ),
    ),
    Claim(
        claim_id="C2_final_tool_directive_causal_for_protocol",
        claim="The final tool-turn directive is causal for exact raw tool protocol behavior on the focused replay suite.",
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric="Contracted exact replay is 7/8 while no-directive exact replay is 0/8 on the same cases.",
        limitation="The replay suite is intentionally focused on eight observed no-directive failures, not a population estimate.",
        next_test="Expand the replay suite with independently authored hard cases and repeated seeds.",
        sources=(
            EvidenceSource(
                "probe_replay_comparison",
                "results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1",
                "A/B replay comparison for the exact same failed no-directive probe cases.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1",
                "Operator-visible live replay of canonical argument failures.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1",
                "Operator-visible live replay of visual no-call failures.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1",
                "Operator-visible live replay of the parallel two-call failure.",
            ),
        ),
    ),
    Claim(
        claim_id="C3_packaged_workflows_can_saturate",
        claim="Packaged workflow completion can wash out raw one-turn tool-protocol failures.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric="H1j/H1k packaged packets saturated while exact replay still showed no-directive failures.",
        limitation="The packaged workflow scaffolds may make the task easier than the one-turn replay contract.",
        next_test="Build a harder packaged workflow slice that preserves one-turn parallel and visual follow-on pressure.",
        sources=(
            EvidenceSource(
                "h1_ablation_packet",
                "results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet",
                "Probe-derived packaged workflows that saturated across candidate rows.",
            ),
            EvidenceSource(
                "h1_ablation_packet",
                "results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet",
                "Packaged parallel-audit workflow showing safe scaffold but easier behavior than raw replay.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/live_parallel_replay_case_deltas.csv",
                "Live exact-replay evidence that the raw parallel two-call shape still fails without the directive.",
            ),
        ),
    ),
    Claim(
        claim_id="C4_visual_catalog_role_routing_is_real",
        claim="Tool-catalog role presentation changes visual routing behavior even without the exact tool-turn directive.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric="visual_role_catalog_v1 moves latest-filter from wrong/no-call behavior to refine_selection argument mismatch.",
        limitation="The intervention improves routing more than exact literal fidelity.",
        next_test="Test catalog-role profiles across a larger visual follow-on set with fresh UI states.",
        sources=(
            EvidenceSource(
                "diagnostic_packet",
                "results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1",
                "Expected-vs-actual visual tool-choice diagnostic for wave3, wave4, and catalog profile.",
            ),
            EvidenceSource(
                "catalog_probe_packet",
                "results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe",
                "Raw catalog-profile probe showing routing and executable visual-form recovery.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1",
                "Focused live comparison showing catalog profile changes wrong-tool/no-call into argument mismatch.",
            ),
        ),
    ),
    Claim(
        claim_id="C5_visual_argument_hints_improve_exactness_but_not_executability",
        claim="Schema-local visual argument hints improve exact selector fidelity but can hurt executable visual-form recovery.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric="v2 reaches 2/3 focused visual live exactness but loses executable form-target recovery.",
        limitation="The improvement is focused on three visual replay cases and has a known form-target regression.",
        next_test="Search for a split selector intervention that preserves v2 filter exactness and v1 form-target executability.",
        sources=(
            EvidenceSource(
                "catalog_probe_packet",
                "results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe",
                "Raw v2 catalog probe showing latest-filter exactness and form-target executable regression.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1",
                "Focused live comparison showing v2 matches contracted exactness but loses executable recovery.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1",
                "Focused live comparison showing v2 exact gain versus v1 and executable regression.",
            ),
        ),
    ),
    Claim(
        claim_id="C6_split_selector_wording_is_negative_evidence",
        claim="Adding broader split-selector wording did not recover the missing visual-form behavior and introduced a protocol-shape regression.",
        status="negative_result_current_packets",
        evidence_strength="moderate_internal",
        primary_metric="v3 raw exact falls to 1/8 versus v2 at 2/8 and readback regresses through tool_name/name mismatch.",
        limitation="This is one candidate profile; it does not rule out all field-specific selector interventions.",
        next_test="Try an executor-grounded schema annotation or few-shot-free field contract that does not add broad behavioral prose.",
        sources=(
            EvidenceSource(
                "catalog_probe_packet",
                "results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe",
                "Raw v3 probe packet and case outputs.",
            ),
            EvidenceSource(
                "catalog_probe_comparison",
                "results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2",
                "Direct v3-vs-v2 comparison showing exact regression.",
            ),
            EvidenceSource(
                "live_replay_decision",
                "results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1",
                "Promotion decision packet explaining why v3 did not spend live replay budget.",
            ),
        ),
    ),
    Claim(
        claim_id="C7_schema_field_hints_tie_exactness_without_executability",
        claim="Schema-local visual field hints are cleaner than broad selector prose but still do not recover executable form targeting.",
        status="negative_result_current_packets",
        evidence_strength="moderate_internal",
        primary_metric="v4 ties v2 at 2/8 raw exact and preserves readback, but stays 0/1 executable and over-prefers refine_selection on form-target.",
        limitation="This tests one schema-field annotation profile on the focused eight-case replay-derived probe.",
        next_test="Create a fresh visual hard slice or constrain refine_selection preference only when a real selection_id is present.",
        sources=(
            EvidenceSource(
                "catalog_probe_packet",
                "results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe",
                "Raw v4 probe packet showing schema-field exactness tie and form-target wrong-tool regression.",
            ),
            EvidenceSource(
                "catalog_probe_comparison",
                "results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2",
                "Direct v4-vs-v2 comparison showing no exact gain over the current best visual candidate.",
            ),
            EvidenceSource(
                "live_replay_decision",
                "results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1",
                "Promotion decision packet explaining why v4 did not spend live replay budget.",
            ),
        ),
    ),
    Claim(
        claim_id="C8_visual_hard_slice_targets_remaining_uncertainty",
        claim="A fresh visual hard slice breaks the earlier saturation and shows schema-field catalog hints can recover executable visual behavior without the exact directive.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "Contracted MLX reaches 8/8 strict, executable, and executor-equivalent; "
            "no-directive falls to 1/8; schema-field hints reach 6/8 strict and 8/8 executor-equivalent."
        ),
        limitation="The packet is eight independently authored visual cases, so it is stronger than design-only evidence but still not a population estimate.",
        next_test="Promote v4 only after building a packaged H1 visual workflow that tests executor-visible success directly.",
        sources=(
            EvidenceSource(
                "visual_hard_slice_probe_packet",
                "results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1",
                "Executed fresh visual hard-slice packet with first-class strict, executable, and executor-equivalence metrics.",
            ),
            EvidenceSource(
                "design_packet",
                "results/reports/visual_hard_slice_design",
                "Fresh visual hard-slice design packet derived from v1/v2/v3/v4 failure analysis.",
            ),
            EvidenceSource(
                "live_replay_decision",
                "results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1",
                "Negative v4 promotion decision motivating a fresh visual hard-slice rather than another live replay.",
            ),
            EvidenceSource(
                "visual_hard_slice_exactness_diagnostic",
                "results/reports/visual_hard_slice_exactness_diagnostic",
                "Exactness-vs-executor diagnostic showing v4's two non-exact rows still hit the expected local visual regions.",
            ),
        ),
    ),
    Claim(
        claim_id="C9_schema_literal_targets_v5_is_negative_evidence",
        claim="A narrow schema-target-literal repair did not improve hard-slice exactness and introduced a routing regression.",
        status="negative_result_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "v5 reaches 5/8 strict and 7/8 executor-equivalent versus v4 at 6/8 strict "
            "and 8/8 executor-equivalent; v5 adds one wrong-tool failure on the stale-selection decoy."
        ),
        limitation="This is still one eight-case hard-slice packet; it rejects the current wording, not all possible target-query exactness interventions.",
        next_test="Do not iterate target-literal wording again until the stale-selection routing failure is isolated separately.",
        sources=(
            EvidenceSource(
                "visual_hard_slice_probe_packet",
                "results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1",
                "Executed v5 hard-slice packet showing the schema-target-literal profile underperforms schema-field hints.",
            ),
            EvidenceSource(
                "visual_hard_slice_profile_comparison",
                "results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints",
                "Direct v5-vs-v4 comparison showing strict, executable, and executor-equivalence regressions concentrated in visual tool routing.",
            ),
            EvidenceSource(
                "visual_hard_slice_exactness_diagnostic",
                "results/reports/visual_hard_slice_exactness_diagnostic",
                "Diagnostic showing v5 preserves the same two label-artifact candidates as v4 while adding one true wrong-tool failure.",
            ),
        ),
    ),
    Claim(
        claim_id="C10_v4_exact_misses_are_executor_success_aliases",
        claim="The remaining v4 visual hard-slice exact misses are executor-success selector aliases, not current evidence of failed visual targeting.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "v4 has 2/8 non-exact rows, both executor-target matches; first-class executor-equivalence "
            "scores v4 at 8/8 while benchmark-label artifact candidate count is 2 and true harness failure count is 0."
        ),
        limitation="This does not prove every visual selector paraphrase is acceptable; it only classifies the current hard-slice v4 misses under the local deterministic executor.",
        next_test="Use the executor-equivalence score to design a packaged H1 visual workflow that separates executor success from strict protocol fidelity.",
        sources=(
            EvidenceSource(
                "visual_hard_slice_exactness_diagnostic",
                "results/reports/visual_hard_slice_exactness_diagnostic",
                "System and gap tables separating canonical argument exactness from executor-visible target success.",
            ),
            EvidenceSource(
                "visual_hard_slice_probe_packet",
                "results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1",
                "Underlying executed hard-slice packet with raw expected/actual calls, deterministic execution outputs, and executor-equivalence scores.",
            ),
        ),
    ),
    Claim(
        claim_id="C11_h1l_packaged_visual_workflows_remain_saturated",
        claim="Promoting the visual executor-equivalence result into current packaged visual workflows does not yet preserve the hard-slice discrimination.",
        status="negative_result_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "H1l candidate rows tie at readiness 0.90406, strict interface 0.85, recovered execution 0.8, "
            "raw clean 1.0, and zero controller repair/fallback/argument repair."
        ),
        limitation="H1l uses existing staged packaged workflows, so it can reject this packaged surface without rejecting the hard-slice executor-equivalence signal.",
        next_test="Preserve the hard-slice or exact-replay shape more faithfully in live operator execution before spending H1l helper-ablation budget.",
        sources=(
            EvidenceSource(
                "h1_ablation_packet",
                "results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet",
                "Executed H1l visual executor-equivalence candidate packet showing all visual catalog rows saturate on packaged workflows.",
            ),
            EvidenceSource(
                "visual_hard_slice_probe_packet",
                "results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1",
                "Underlying hard-slice packet that motivated H1l by separating strict exactness from executor-equivalent target success.",
            ),
        ),
    ),
    Claim(
        claim_id="C12_replay_shaped_live_preserves_visual_hard_slice_signal",
        claim="Replay-shaped CLI-live execution preserves the visual hard-slice executor-equivalence signal that packaged H1l workflows washed out.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "On the two preserved no-directive visual hard-slice failures, no-directive is 0/2 strict "
            "and 0/2 executor-equivalent; contracted MLX is 2/2 strict and executor-equivalent; schema-field "
            "hints is the strongest no-directive row at 1/2 strict and 2/2 executor-equivalent."
        ),
        limitation="The live replay matrix covers the two preserved no-directive hard-slice failures, not the full eight-case hard-slice candidate matrix.",
        next_test="Build a second replay-shaped live slice that repeats the executor-alias and stale-selection cases under harder decoys before returning to packaged H1 workflows.",
        sources=(
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1",
                "Replay source packet preserving visual hard-slice cases for CLI-live execution.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_contracted_vs_no_directive_live_v1",
                "Replay-shaped CLI-live comparison showing contracted MLX is the 2/2 strict and executor-equivalent upper bound.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_role_catalog_vs_no_directive_live_v1",
                "Replay-shaped CLI-live comparison showing role catalog v1 recovers one strict/executor-equivalent case over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_argument_hints_vs_no_directive_live_v1",
                "Replay-shaped CLI-live comparison showing argument hints v2 matches role catalog v1 on this preserved failure slice.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_schema_field_hints_vs_no_directive_live_v2",
                "Replay-shaped CLI-live comparison showing schema-field hints recovers executor-equivalent target success over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_schema_literal_targets_vs_no_directive_live_v1",
                "Replay-shaped CLI-live comparison showing schema target literals v5 remain negative on strict exactness and introduce a wrong-tool stale-selection miss.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_summary.csv",
                "Report table summarizing exact, executable, and executor-equivalence live replay deltas.",
            ),
        ),
    ),
    Claim(
        claim_id="C13_visual_live_stress_separates_executor_grounding_from_strict_fidelity",
        claim="A harder visual live stress replay preserves the executor-grounding advantage of schema-local catalog hints without improving strict protocol fidelity.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "On four fresh stress cases, no-directive MLX is 2/4 strict and 3/4 executor-equivalent; "
            "contracted MLX is 4/4 strict and executor-equivalent; schema-field hints and schema target literals are "
            "2/4 strict but 4/4 executor-equivalent. On the 8-case alias-repeat follow-up, schema-field hints "
            "matches no-directive at 2/8 strict but improves executor-equivalence from 5/8 to 7/8, while schema "
            "target literals reach 3/8 strict and 8/8 executor-equivalent."
        ),
        limitation="The alias-repeat matrix is still one deterministic replay-shaped live packet, not a repeated stochastic estimate or a broad packaged workflow.",
        next_test="Repeat the alias-repeat packet across seeds or promote only the surviving metric-panel/callout mechanisms into a non-saturated H1m packaged workflow.",
        sources=(
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1",
                "Designed stress replay packet with fresh metric-panel alias and stale-selection decoy cases.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_no_directive_execute_v1",
                "No-directive MLX stress replay baseline.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_contracted_vs_no_directive_v1",
                "Stress replay comparison showing contracted MLX remains the strict and executor-equivalent upper bound.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_schema_field_hints_vs_no_directive_v1",
                "Stress replay comparison showing schema-field hints recover executor-equivalence without strict exactness gain.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_schema_literal_targets_vs_no_directive_v1",
                "Stress replay comparison showing schema target literals match the executor-equivalence gain but not strict fidelity.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_stress_live_replay_summary.csv",
                "Report table summarizing exact, executable, and executor-equivalence stress replay deltas.",
            ),
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1",
                "Eight-case alias-repeat follow-up with additional metric-panel and callout decoys.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_repeat_schema_field_hints_vs_no_directive_v1",
                "Alias-repeat comparison showing schema-field hints improve executor-equivalence without strict exactness gain.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_repeat_contracted_vs_no_directive_v1",
                "Alias-repeat comparison showing contracted MLX remains the strict upper bound.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_repeat_role_catalog_vs_no_directive_v1",
                "Alias-repeat comparison showing role catalog is partial and loses strict exactness versus no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_repeat_argument_hints_vs_no_directive_v1",
                "Alias-repeat comparison showing argument hints improve executor-equivalence but not strict exactness.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_repeat_schema_literal_targets_vs_no_directive_v1",
                "Alias-repeat comparison showing schema target literals reach full executor-equivalence with a small strict gain.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_repeat_live_replay_summary.csv",
                "Report table summarizing the completed alias-repeat replay matrix.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_repeat_diagnostic/diagnostic.md",
                "Diagnostic report classifying strict gains, executor-only gains, and regressions on the alias-repeat matrix.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_live_stress_diagnostic/diagnostic.md",
                "Diagnostic report explaining strict gains, executor-only gains, and regressions across the stress matrix.",
            ),
        ),
    ),
    Claim(
        claim_id="C14_h1m_packaged_alias_repeat_saturates",
        claim="Promoting the alias-repeat visual replay result into current packaged workflows washes out the replay discrimination.",
        status="negative_result_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "H1m candidate rows tie at readiness 0.87783, strict interface 0.75, recovered execution 0.667, "
            "raw clean 1.0, and zero controller repair/fallback/argument repair across all six visual contract rows."
        ),
        limitation=(
            "H1m rejects this staged packaged-workflow surface; it does not reject the alias-repeat replay result, "
            "which still separates strict fidelity from executor-equivalent visual grounding."
        ),
        next_test=(
            "Do not run H1m helper ablations until a packaged or non-packaged live visual surface separates; "
            "next preserve replay shape through repeated alias packets or less staged live tasks."
        ),
        sources=(
            EvidenceSource(
                "h1_ablation_packet",
                "results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet",
                "Executed H1m visual alias-repeat candidate packet showing all rows tie on packaged workflows.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h1m_visual_alias_repeat_candidate_metrics.csv",
                "Generated H1m candidate metrics table in the main harnessing report.",
            ),
            EvidenceSource(
                "continuity_brief",
                "docs/continuity/h1m-slice.md",
                "H1m design brief describing the packaged workflow promotion target and helper-ablation gate.",
            ),
        ),
    ),
    Claim(
        claim_id="C15_packaged_visual_surfaces_wash_out_replay_discrimination",
        claim="Current packaged visual workflows can erase replay-shaped visual row separation, so packaging is an experimental variable.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "The packaged replay gap diagnostic finds 2/2 visual promotion surfaces with positive replay gains but "
            "zero packaged readiness and strict-interface span: H1l max replay executor-equivalence delta 1.0, "
            "H1m max replay executor-equivalence delta 0.375, both packaged spans 0.0."
        ),
        limitation=(
            "This is a two-surface internal diagnostic over current visual packets, not a general proof that all "
            "packaged workflows are too easy."
        ),
        next_test=(
            "Design a less staged live visual task or repeated replay packet before returning to packaged helper "
            "ablations."
        ),
        sources=(
            EvidenceSource(
                "diagnostic_report",
                "results/reports/packaged_replay_gap_diagnostic/diagnostic.md",
                "Diagnostic comparing replay-shaped visual gains against H1l/H1m packaged workflow saturation.",
            ),
            EvidenceSource(
                "diagnostic_table",
                "results/reports/packaged_replay_gap_diagnostic/tables/packaged_replay_gap_surfaces.csv",
                "Surface-level table with replay deltas and packaged metric spans.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h1l_visual_executor_equivalence_candidate_metrics.csv",
                "Generated H1l packaged metrics table used in the gap diagnostic.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h1m_visual_alias_repeat_candidate_metrics.csv",
                "Generated H1m packaged metrics table used in the gap diagnostic.",
            ),
        ),
    ),
    Claim(
        claim_id="C16_visual_alias_transfer_favors_argument_hints_executor_grounding",
        claim="On fresh alias-transfer visual cases, argument-hint cataloging generalizes best for executor-equivalent target success while contracted MLX remains the strict-fidelity upper bound.",
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "H1n alias-transfer replay: no-directive is 0/6 strict and 2/6 executor-equivalent; argument hints v2 is "
            "1/6 strict and 6/6 executor-equivalent; schema target literals v5 is 1/6 strict and 4/6 executor-equivalent; "
            "contracted MLX is 5/6 strict but 1/6 executor-equivalent under the current executor-target scorer."
        ),
        limitation=(
            "This is one deterministic six-case transfer packet, and the contracted strict/executor split needs scorer-level "
            "inspection before being treated as a model-only ranking."
        ),
        next_test=(
            "Inspect the contracted exact-but-not-executor-equivalent rows, then repeat the transfer packet or promote "
            "argument hints into a new non-packaged live helper-ablation slice."
        ),
        sources=(
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1",
                "Designed six-case alias-transfer replay packet with fresh labels and decoys.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_transfer_argument_hints_execute_v1",
                "Argument-hints alias-transfer execution reaching full executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_transfer_argument_hints_vs_no_directive_v1",
                "Comparison showing argument hints improves executor-equivalence by 0.667 over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_transfer_contracted_vs_no_directive_v1",
                "Comparison showing contracted MLX is the strict-fidelity upper bound but regresses executor-equivalence under the current scorer.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_transfer_schema_literal_targets_vs_no_directive_v1",
                "Comparison showing schema-target literals improve executor-equivalence by 0.333 over no-directive.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_live_replay_summary.csv",
                "Generated report summary table for the alias-transfer replay matrix.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_diagnostic/diagnostic.md",
                "Diagnostic report classifying strict gains, executor-only gains, and transfer regressions.",
            ),
        ),
    ),
    Claim(
        claim_id="C17_h1n_strict_exactness_matches_planner_not_oracle",
        claim="H1n strict exactness is partly a planner-contract artifact because most generated expected calls do not satisfy the visual execution oracle.",
        status="benchmark_contract_issue_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "The H1n contract-split diagnostic finds 5/6 generated expected-call contracts fail the packet's "
            "expected_execution oracle; contracted MLX has 4 exact-but-not-executor rows, while argument hints v2 "
            "has 6/6 executor-target successes."
        ),
        limitation=(
            "This diagnoses the current H1n packet contract; it does not invalidate the executor-equivalence result, "
            "but it does require rebuilding H1n with oracle expected calls before using strict exactness as a headline metric."
        ),
        next_test=(
            "Use the rebuilt oracle H1n matrix as the reference packet before any packaged or helper-ablation promotion."
        ),
        sources=(
            EvidenceSource(
                "diagnostic_report",
                "results/reports/h1n_alias_transfer_contract_split/diagnostic.md",
                "Contract-split diagnostic separating heuristic planner-call exactness from executor-target oracle success.",
            ),
            EvidenceSource(
                "diagnostic_table",
                "results/reports/h1n_alias_transfer_contract_split/tables/h1n_expected_call_contract_audit.csv",
                "Per-case audit showing whether generated expected calls satisfy the visual execution oracle.",
            ),
            EvidenceSource(
                "diagnostic_table",
                "results/reports/h1n_alias_transfer_contract_split/tables/h1n_replay_contract_split.csv",
                "Per-run replay table classifying exact/non-exact and executor-target outcomes.",
            ),
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2",
                "Rebuilt H1n dry-run packet whose expected calls are derived from target region labels and execute to the oracle target.",
            ),
        ),
    ),
    Claim(
        claim_id="C18_h1n_oracle_transfer_identifies_argument_hints_as_clean_winner",
        claim=(
            "When H1n alias-transfer replay uses oracle executable expected calls, argument-hint cataloging is the clean "
            "local-Gemma transfer winner and contracted prompting is not a reliable upper bound."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "Oracle H1n alias-transfer replay-live: no-directive is 2/6 exact and executor-equivalent; contracted is "
            "1/6; role catalog is 3/6; argument hints v2 is 5/6 exact and 6/6 executor-equivalent; schema-field hints "
            "is 2/6; schema target literals v5 is 4/6."
        ),
        limitation=(
            "This is still a deterministic six-case transfer packet, so it is causal evidence for this slice rather "
            "than a broad stochastic estimate of all visual tool grounding."
        ),
        next_test=(
            "Repeat the oracle transfer packet or promote argument hints into a non-packaged helper-ablation/live slice "
            "to test whether the effect survives fresh visual families."
        ),
        sources=(
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2",
                "Oracle H1n transfer packet with serialized expected calls that execute to target region labels.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_argument_hints_execute_v2",
                "Argument-hints oracle execution reaching 5/6 exact and 6/6 executor-equivalent target success.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_argument_hints_vs_no_directive_v2",
                "Comparison showing argument hints improves exactness by 0.5 and executor-equivalence by 0.667 over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_schema_literal_targets_vs_no_directive_v2",
                "Comparison showing schema target literals are the second-place oracle transfer mechanism.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md",
                "Diagnostic report summarizing the oracle matrix winner set, regressions, and strict upper bound.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv",
                "Generated main-report table for the oracle alias-transfer replay matrix.",
            ),
        ),
    ),
    Claim(
        claim_id="C19_h1n_argument_hints_gain_is_not_controller_helper_artifact",
        claim=(
            "On the H1n oracle transfer packet, the argument-hints gain persists when controller repair, controller "
            "fallback, or argument repair is disabled one at a time."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "Argument hints remains 5/6 exact and 6/6 executor-equivalent with no controller repair, no controller "
            "fallback, and no argument repair; all three helper-ablation comparisons have 0.0 exact and executor-equivalence deltas."
        ),
        limitation=(
            "This only attributes the six-case oracle replay slice. It does not replace broader controller-helper "
            "ablations on packaged workflows or future visual families."
        ),
        next_test=(
            "Repeat the helper-ablation result on a fresh oracle transfer packet or on a less staged live visual workflow."
        ),
        sources=(
            EvidenceSource(
                "diagnostic_report",
                "results/reports/h1n_oracle_helper_ablation/diagnostic.md",
                "Helper-ablation diagnostic summarizing no observed helper dependence on the H1n oracle argument-hints row.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_argument_hints_no_controller_repair_execute_v1",
                "Argument-hints oracle replay with controller repair disabled.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_argument_hints_no_controller_fallback_execute_v1",
                "Argument-hints oracle replay with controller fallback disabled.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_argument_hints_no_argument_repair_execute_v1",
                "Argument-hints oracle replay with argument repair disabled.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_controller_repair_vs_argument_hints_v1",
                "Comparison showing no exact or executor-equivalence delta when controller repair is disabled.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_controller_fallback_vs_argument_hints_v1",
                "Comparison showing no exact or executor-equivalence delta when controller fallback is disabled.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_argument_hints_no_argument_repair_vs_argument_hints_v1",
                "Comparison showing no exact or executor-equivalence delta when argument repair is disabled.",
            ),
        ),
    ),
    Claim(
        claim_id="C20_h1n_oracle_repeat_confirms_catalog_transfer_not_contracted_upper_bound",
        claim=(
            "A fresh H1n oracle repeat preserves the catalog-profile transfer effect while showing that contracted "
            "prompting is not a reliable visual-transfer upper bound."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "H1n oracle repeat: no-directive is 2/6 exact and executor-equivalent; contracted is 0/6; role catalog "
            "and schema-field hints are 4/6; argument hints v2 and schema target literals v5 are 5/6 exact and "
            "6/6 executor-equivalent."
        ),
        limitation=(
            "This repeat is still deterministic and six-case; it strengthens the transfer claim but does not replace "
            "larger stochastic repeats or less staged live workflows."
        ),
        next_test=(
            "Run a third fresh oracle transfer packet or promote the tied argument-hints/schema-literal profiles into "
            "a less staged live visual workflow."
        ),
        sources=(
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1",
                "Fresh H1n repeat packet with six new alias-transfer labels and decoys.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_repeat_diagnostic/diagnostic.md",
                "Diagnostic report summarizing the repeat matrix winner set and contracted regression.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_repeat_argument_hints_execute_v1",
                "Argument-hints repeat execution reaching full executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_repeat_schema_literal_targets_execute_v1",
                "Schema-literal repeat execution tying argument hints on exactness and executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_repeat_argument_hints_vs_no_directive_v1",
                "Comparison showing argument hints improves exactness by 0.5 and executor-equivalence by 0.667 over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_repeat_schema_literal_targets_vs_no_directive_v1",
                "Comparison showing schema target literals ties argument hints on the repeat packet.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_repeat_contracted_vs_no_directive_v1",
                "Comparison showing contracted prompting regresses below no-directive on the repeat packet.",
            ),
        ),
    ),
    Claim(
        claim_id="C21_h1n_two_packet_oracle_synthesis_narrows_next_visual_question",
        claim=(
            "The two-packet H1n oracle synthesis narrows the next visual-transfer research question to whether "
            "argument hints and schema target literals generalize beyond replay-shaped oracle packets."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "Across two oracle H1n packets, argument hints is executor-equivalent in both packets at 6/6 and 6/6; "
            "schema target literals rises from 4/6 to 6/6 executor-equivalent; contracted is 1/6 then 0/6; helper "
            "ablation preserves argument hints at 5/6 exact and 6/6 executor-equivalent with zero deltas."
        ),
        limitation=(
            "The synthesis combines deterministic replay-shaped oracle packets, so it is a directional finding "
            "for harness design rather than a final population-level estimate."
        ),
        next_test=(
            "Run a third held-out oracle family or a less staged live visual workflow comparing argument hints "
            "against schema target literals without relying on packaged workflow saturation."
        ),
        sources=(
            EvidenceSource(
                "synthesis_report",
                "results/reports/h1n_oracle_transfer_synthesis/report.md",
                "Compact two-packet H1n oracle transfer synthesis with helper-ablation interpretation.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md",
                "First oracle H1n transfer diagnostic showing argument hints as the clean winner.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_repeat_diagnostic/diagnostic.md",
                "Fresh repeat diagnostic showing argument hints and schema target literals tie at full executor-equivalence.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/h1n_oracle_helper_ablation/diagnostic.md",
                "Helper-ablation diagnostic showing the argument-hints gain is not explained by the tested controller helpers.",
            ),
        ),
    ),
    Claim(
        claim_id="C22_h1n_oblique_labels_favor_argument_hints_over_schema_literals",
        claim=(
            "On a held-out H1n oracle packet with code-like visible labels, argument hints generalizes better than "
            "schema target literals and contracted prompting."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "H1n oblique-label oracle replay-live: no-directive is 0/6 exact and executor-equivalent; contracted "
            "is 1/6; role catalog is 2/6; argument hints v2 is 4/6; schema-field hints v4 is 3/6; schema target "
            "literals v5 is 0/6."
        ),
        limitation=(
            "The packet intentionally stresses literal code-like target labels, so it should be interpreted as a "
            "hard held-out mechanism test rather than a representative visual-work population estimate."
        ),
        next_test=(
            "Inspect argument-hints misses on the oblique packet, then compare argument hints and schema-field hints "
            "on a less replay-shaped live visual task."
        ),
        sources=(
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1",
                "Held-out oblique-label oracle packet with code-like visible target labels and semantic decoys.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md",
                "Diagnostic report summarizing the oblique-label live replay matrix.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_argument_hints_execute_v1",
                "Argument-hints execution reaching 4/6 exact and executor-equivalent target success.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_argument_hints_vs_no_directive_v1",
                "Comparison showing argument hints improves exactness and executor-equivalence by 0.667 over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_schema_literal_targets_vs_no_directive_v1",
                "Comparison showing schema target literals do not improve on the oblique-label packet.",
            ),
        ),
    ),
    Claim(
        claim_id="C23_h1n_oblique_argument_hints_misses_are_code_and_negation_errors",
        claim=(
            "The remaining H1n oblique argument-hints failures are specific literal-code and negated-decoy errors, "
            "not generic visual tool-entry collapse."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "Argument hints has two oblique misses: `cell r42` is truncated to `cell`, and `alert p55` is replaced "
            "with the negated decoy `consent toggle`; schema-field hints has three misses spanning semantic broad "
            "selection, code-suffix truncation, and one tool-entry failure."
        ),
        limitation=(
            "This diagnostic classifies deterministic replay outputs from one held-out packet; it is a mechanism "
            "diagnostic, not a new population-level accuracy estimate."
        ),
        next_test=(
            "Try a narrow code-suffix preservation intervention or a negated-decoy guard only if it can be tested "
            "without regressing the four oblique argument-hints wins."
        ),
        sources=(
            EvidenceSource(
                "diagnostic_report",
                "results/reports/h1n_oblique_miss_analysis/diagnostic.md",
                "Miss-analysis report classifying argument-hints and schema-field failures on the oblique packet.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_argument_hints_execute_v1",
                "Underlying argument-hints replay packet with actual calls and execution outputs.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_schema_field_hints_execute_v1",
                "Underlying schema-field replay packet used as the second-place comparison row.",
            ),
        ),
    ),
    Claim(
        claim_id="C24_h1n_oblique_code_hints_repair_two_misses_with_one_regression",
        claim=(
            "A narrow oblique-code catalog profile improves the held-out H1n oblique packet over argument hints by "
            "repairing code-suffix and negated-decoy misses, with one new stale-selection routing regression."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "Oblique code hints reaches 5/6 exact and executor-equivalent versus argument hints at 4/6, improving "
            "by +0.167 on both metrics; it repairs `cell r42` and `alert p55` but loses `field e19` as a wrong-tool case."
        ),
        limitation=(
            "The profile is tuned from observed oblique misses, so this is a successful repair on a held-out packet "
            "but still requires a fresh packet or less staged live task before promotion."
        ),
        next_test=(
            "Run the oblique-code profile on the earlier oracle and repeat packets, or build a fresh post-repair "
            "held-out packet to check whether the `field e19` regression is localized."
        ),
        sources=(
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_code_hints_execute_v1",
                "Oblique-code profile execution reaching 5/6 exact and executor-equivalent target success.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1",
                "Direct comparison showing +0.167 exact and executor-equivalence deltas over argument hints.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/h1n_oblique_code_hints_delta/diagnostic.md",
                "Case-level gain/loss diagnostic showing two repairs, one stale-selection regression, and three preserved wins.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md",
                "Updated oblique diagnostic including the oblique-code profile as the current best row.",
            ),
        ),
    ),
    Claim(
        claim_id="C25_h1n_oblique_code_hints_is_localized_not_general",
        claim=(
            "The oblique-code profile is a localized repair, not a general replacement for argument hints across "
            "H1n oracle transfer packets."
        ),
        status="negative_result_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "Across three H1n oracle packets, argument hints has 14/18 exact and 16/18 executor-equivalent successes, "
            "while oblique code hints has 11/18 exact and 12/18 executor-equivalent successes; code hints improves "
            "only the oblique packet."
        ),
        limitation=(
            "The result compares one targeted profile against argument hints on three replay-shaped oracle packets; "
            "it does not rule out a revised stale-selection guard or a future profile with narrower activation."
        ),
        next_test=(
            "Build a stale-selection guard or activation-gated code-suffix profile, then test on a fresh post-repair "
            "holdout before broad promotion."
        ),
        sources=(
            EvidenceSource(
                "synthesis_report",
                "results/reports/h1n_code_hints_transfer_synthesis/report.md",
                "Three-packet synthesis showing oblique-code gains are localized and transfer losses dominate overall.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_oracle_code_hints_transfer_execute_v1",
                "Oblique-code profile execution on the earlier oracle transfer packet.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_oracle_repeat_code_hints_transfer_execute_v1",
                "Oblique-code profile execution on the repeat oracle transfer packet.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_code_hints_vs_argument_hints_transfer_v1",
                "Direct comparison showing negative transfer on the earlier oracle packet.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_repeat_code_hints_vs_argument_hints_transfer_v1",
                "Direct comparison showing negative transfer on the repeat oracle packet.",
            ),
        ),
    ),
    Claim(
        claim_id="C26_h1n_oblique_code_guard_fixes_v6_regression",
        claim=(
            "An activation-gated oblique-code profile repairs the v6 stale-selection regression and saturates the "
            "held-out H1n oblique oracle packet."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "Oblique code guard v7 reaches 6/6 exact and 6/6 executor-equivalent on the oblique packet, improving "
            "over argument hints by +0.333 and over v6 code hints by +0.167 on both metrics."
        ),
        limitation=(
            "This is a scoped oblique-packet result. The earlier v6 transfer loss shows that this profile must be "
            "transfer-tested before promotion beyond code-like oblique labels."
        ),
        next_test=(
            "Run the code-guard profile on the earlier oracle and repeat packets, then build a fresh post-repair "
            "held-out packet if transfer is not negative."
        ),
        sources=(
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_oracle_oblique_code_guard_execute_v1",
                "Code-guard execution reaching 6/6 exact and executor-equivalent target success on the oblique packet.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_argument_hints_v1",
                "Direct comparison showing +0.333 exact and executor-equivalence deltas over argument hints.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_oblique_code_guard_vs_code_hints_v1",
                "Direct comparison showing the code guard repairs the v6 field-e19 regression.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md",
                "Updated oblique matrix diagnostic including code guard as the strict and executor-equivalence upper bound.",
            ),
        ),
    ),
    Claim(
        claim_id="C27_h1n_code_guard_improves_v6_but_not_argument_hints",
        claim=(
            "The activation-gated code guard is a better scoped repair than v6 code hints, but still does not "
            "replace argument hints across the three H1n oracle packets."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "Across the three H1n oracle packets, code guard reaches 14/18 exact and 15/18 executor-equivalent "
            "successes versus v6 at 11/18 and 12/18, while argument hints remains 14/18 exact and 16/18 "
            "executor-equivalent."
        ),
        limitation=(
            "The comparison is still replay-shaped and packet-conditioned; it should be followed by a fresh "
            "post-repair holdout before claiming a general visual catalog profile."
        ),
        next_test=(
            "Build a fresh post-repair holdout with code-like labels, stale-selection mentions, and non-code "
            "transfer labels, then compare argument hints and code guard."
        ),
        sources=(
            EvidenceSource(
                "synthesis_report",
                "results/reports/h1n_code_guard_transfer_synthesis/report.md",
                "Three-packet synthesis showing code guard improves over v6 but still trails argument hints on executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_oracle_code_guard_transfer_execute_v1",
                "Code-guard profile execution on the earlier oracle packet.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_oracle_repeat_code_guard_transfer_execute_v1",
                "Code-guard profile execution on the repeat oracle packet.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_code_guard_vs_argument_hints_transfer_v1",
                "Direct comparison against argument hints on the earlier oracle packet.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_oracle_repeat_code_guard_vs_argument_hints_transfer_v1",
                "Direct comparison against argument hints on the repeat oracle packet.",
            ),
        ),
    ),
)


def build_ledger(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    claim_rows = [_claim_row(claim) for claim in CLAIMS]
    source_rows = [_source_row(claim, source) for claim in CLAIMS for source in claim.sources]
    missing_sources = [row for row in source_rows if row["exists"] is False]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "claim_count": len(claim_rows),
        "evidence_source_count": len(source_rows),
        "missing_source_count": len(missing_sources),
        "claim_ids": [claim.claim_id for claim in CLAIMS],
    }

    _write_csv(tables_dir / "claim_ledger.csv", claim_rows)
    _write_csv(tables_dir / "evidence_sources.csv", source_rows)
    (output / "ledger.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "claims": claim_rows,
                "evidence_sources": source_rows,
                "missing_sources": missing_sources,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "ledger.md").write_text(_markdown_report(manifest, claim_rows, source_rows), encoding="utf-8")
    return {"manifest": manifest, "claims": claim_rows, "evidence_sources": source_rows, "missing_sources": missing_sources}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-facing claim/evidence ledger for Moonie research.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_ledger(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


def _claim_row(claim: Claim) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "claim": claim.claim,
        "status": claim.status,
        "evidence_strength": claim.evidence_strength,
        "primary_metric": claim.primary_metric,
        "limitation": claim.limitation,
        "next_test": claim.next_test,
        "source_count": len(claim.sources),
    }


def _source_row(claim: Claim, source: EvidenceSource) -> dict[str, Any]:
    path = ROOT / source.path
    return {
        "claim_id": claim.claim_id,
        "artifact_type": source.artifact_type,
        "path": source.path,
        "purpose": source.purpose,
        "exists": path.exists(),
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


def _markdown_report(manifest: dict[str, Any], claim_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Publication Evidence Ledger",
        "",
        "This ledger maps paper-level claims to packet-backed evidence and known limitations.",
        "",
        "## Manifest",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- claim_count: `{manifest['claim_count']}`",
        f"- evidence_source_count: `{manifest['evidence_source_count']}`",
        f"- missing_source_count: `{manifest['missing_source_count']}`",
        "",
        "## Claims",
        "",
        "| Claim ID | Status | Evidence | Primary Metric | Limitation | Next Test |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in claim_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["claim_id"]),
                    str(row["status"]),
                    str(row["evidence_strength"]),
                    str(row["primary_metric"]),
                    str(row["limitation"]),
                    str(row["next_test"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Evidence Sources",
            "",
            "| Claim ID | Type | Exists | Path | Purpose |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for row in source_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["claim_id"]),
                    str(row["artifact_type"]),
                    str(row["exists"]),
                    str(row["path"]),
                    str(row["purpose"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
