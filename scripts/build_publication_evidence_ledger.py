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
    Claim(
        claim_id="C28_h1n_post_repair_holdout_favors_code_guard",
        claim=(
            "A fresh post-repair H1n holdout favors the activation-gated code guard over no-directive, "
            "contracted/default MLX, argument hints, and v6 code hints."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "On the eight-case post-repair holdout, code guard reaches 6/8 exact and executor-equivalent "
            "successes, versus no-directive at 2/8, contracted/default at 3/8, argument hints at 5/8, "
            "and v6 code hints at 5/8."
        ),
        limitation=(
            "The packet is fresh relative to the oblique repair but remains replay-shaped and small; the "
            "remaining misses on `chip l90` and `status pill` need a follow-up micro-slice before promotion."
        ),
        next_test=(
            "Build a focused follow-up around the two residual misses and test whether a hybrid activation "
            "profile can preserve argument-hints non-code behavior while keeping the code-guard gains."
        ),
        sources=(
            EvidenceSource(
                "replay_packet",
                "results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1",
                "Fresh eight-case post-repair holdout with code-like labels, stale-selection mentions, and non-code labels.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_post_repair_code_guard_execute_v1",
                "Code-guard execution reaching 6/8 exact and executor-equivalent successes on the holdout.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_post_repair_code_guard_vs_no_directive_v1",
                "Comparison showing +0.50 exact and executor-equivalence deltas over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_post_repair_code_guard_vs_argument_hints_v1",
                "Comparison showing +0.125 exact and executor-equivalence deltas over argument hints.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_post_repair_code_guard_vs_code_hints_v1",
                "Comparison showing +0.125 exact and executor-equivalence deltas over v6 code hints.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_post_repair_diagnostic/diagnostic.md",
                "Matrix diagnostic identifying code guard as the post-repair strict upper bound and recording regressions.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_post_repair_live_replay_summary.csv",
                "Paper-facing table summarizing post-repair candidate rates.",
            ),
        ),
    ),
    Claim(
        claim_id="C29_h1n_residual_holdout_favors_hybrid_label_guard",
        claim=(
            "A residual H1n holdout favors the hybrid label guard as the current best strict selector profile "
            "for local MLX Gemma visual tool use."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "On the eight-case residual holdout, v8 hybrid label guard reaches 7/8 exact and executor-equivalent "
            "successes, versus contracted/default at 2/8, no-directive at 4/8, argument hints at 5/8 exact "
            "and 7/8 executor-equivalent, and v6/v7 code profiles at 6/8 exact."
        ),
        limitation=(
            "The improvement is strict-selector fidelity on a small replay-shaped packet, not a broad workflow "
            "readiness estimate; `state pill` remains unresolved across the tested profiles."
        ),
        next_test=(
            "Build a component-role/value disambiguation micro-slice around pill/tile/state labels and then "
            "promote only if the profile survives packaged workflow execution without new regressions."
        ),
        sources=(
            EvidenceSource(
                "replay_packet",
                "results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1",
                "Fresh eight-case residual holdout targeting the post-repair chip, pill, and stale-selection misses.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_residual_hybrid_label_guard_execute_v1",
                "Hybrid label guard execution reaching 7/8 strict and executor-equivalent successes.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_residual_hybrid_label_guard_vs_no_directive_v1",
                "Comparison showing +0.375 exact and executor-equivalence deltas over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_residual_hybrid_label_guard_vs_argument_hints_v1",
                "Comparison showing +0.25 exact delta over argument hints with tied executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_residual_hybrid_label_guard_vs_code_guard_v1",
                "Comparison showing +0.125 exact delta over v7 code guard with tied executor-equivalence.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_residual_diagnostic/diagnostic.md",
                "Matrix diagnostic identifying hybrid label guard as the residual strict upper bound and `state pill` as the remaining miss.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_residual_live_replay_summary.csv",
                "Paper-facing table summarizing residual candidate rates.",
            ),
        ),
    ),
    Claim(
        claim_id="C30_component_value_guard_is_negative_evidence",
        claim=(
            "A focused component-role/value holdout rejects the broad v9 component-value guard as a local MLX "
            "Gemma visual prompt intervention."
        ),
        status="negative_result_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "On the eight-case component-value holdout, argument hints v2 and hybrid label guard v8 both reach "
            "6/8 exact and 7/8 executor-equivalent successes; no-directive reaches 5/8 exact and 6/8 "
            "executor-equivalent; the v9 component-value guard falls to 4/8 exact and 4/8 executor-equivalent."
        ),
        limitation=(
            "The packet is intentionally narrow and replay-shaped; the result rejects this broad prompt-contract "
            "wording, not all possible component-role/value runtime interventions."
        ),
        next_test=(
            "Promote only interventions that preserve argument fidelity on the component-value holdout and validate "
            "their transfer on fresh residual/component packets."
        ),
        sources=(
            EvidenceSource(
                "replay_packet",
                "results/tool_probe_replay_packets/20260510T_visual_hard_slice_component_value_oracle_dry_run_v1",
                "Fresh eight-case component-role/value holdout with pill, badge, chip, field, and stale-selection decoys.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_component_value_component_value_guard_execute_v1",
                "v9 component-value guard execution reaching only 4/8 exact and executor-equivalent successes.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_component_value_argument_hints_execute_v1",
                "Argument-hints execution reaching 6/8 exact and 7/8 executor-equivalent successes.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_component_value_hybrid_label_guard_execute_v1",
                "Hybrid label guard execution matching argument hints at 6/8 exact and 7/8 executor-equivalent successes.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_component_value_component_value_guard_vs_no_directive_v1",
                "Comparison showing the v9 component-value guard regresses below no-directive by -0.125 exact and -0.25 executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_component_value_argument_hints_vs_no_directive_v1",
                "Comparison showing +0.125 exact and executor-equivalence deltas for argument hints over no-directive.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_component_value_diagnostic/diagnostic.md",
                "Matrix diagnostic recording v9 regressions and separating v9 from the later v10 no-call rescue.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_component_value_live_replay_summary.csv",
                "Paper-facing table summarizing component-value candidate rates.",
            ),
        ),
    ),
    Claim(
        claim_id="C31_no_call_control_rescue_is_current_component_value_upper_bound",
        claim=(
            "A narrow no-call visual-control rescue profile improves the component-value holdout without the broad "
            "component-role/value regressions seen in v9."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "On the eight-case component-value holdout, v10 reaches 7/8 exact and 8/8 executor-equivalent successes, "
            "improving over no-directive by +0.25 exact/+0.25 executor-equivalence and over argument hints/hybrid by "
            "+0.125 exact/+0.125 executor-equivalence."
        ),
        limitation=(
            "The component-value gain transfers unevenly: it ties or partially transfers on some H1n packets but does "
            "not replace specialized code/label guards."
        ),
        next_test=(
            "Build a fresh H1o control-first slice that "
            "separates activation/no-call rescue from selector-value disambiguation."
        ),
        sources=(
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_component_value_no_call_control_rescue_execute_v1",
                "v10 no-call control rescue execution reaching 7/8 exact and 8/8 executor-equivalent successes.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_component_value_no_call_control_rescue_vs_no_directive_v1",
                "Comparison showing +0.25 exact and +0.25 executor-equivalence deltas over no-directive.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_component_value_no_call_control_rescue_vs_argument_hints_v1",
                "Comparison showing v10 improves over argument hints by +0.125 exact and executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_component_value_no_call_control_rescue_vs_hybrid_label_guard_v1",
                "Comparison showing v10 improves over hybrid label guard by +0.125 exact and executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1n_component_value_no_call_control_rescue_vs_component_value_guard_v1",
                "Comparison showing v10 avoids the v9 component-value guard regressions by +0.375 exact and +0.50 executor-equivalence.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_component_value_diagnostic/diagnostic.md",
                "Component-value matrix diagnostic identifying v10 as the strict upper bound and only executor-equivalent full-success row.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_component_value_live_replay_summary.csv",
                "Paper-facing table summarizing v10 against the component-value candidate set.",
            ),
            EvidenceSource(
                "synthesis_report",
                "results/reports/h1n_no_call_rescue_transfer_synthesis/report.md",
                "Cross-packet transfer synthesis showing v10's aggregate lift and incumbent boundary.",
            ),
        ),
    ),
    Claim(
        claim_id="C32_no_call_rescue_is_scoped_not_general",
        claim=(
            "The v10 no-call control rescue is a scoped activation improvement, not a general replacement for "
            "specialized visual label/code guards."
        ),
        status="supported_current_packets",
        evidence_strength="moderate_internal",
        primary_metric=(
            "Across component-value, residual, post-repair, and oblique transfer packets, v10 reaches 22/30 exact "
            "and 25/30 executor-equivalent successes versus no-directive at 11/30 and 12/30, but trails incumbents "
            "at 25/30 exact and 26/30 executor-equivalent."
        ),
        limitation=(
            "The aggregate spans replay-shaped H1n micro-slices, not a broad population estimate or packaged workflow "
            "confirmation."
        ),
        next_test=(
            "Author H1o as a factorial control slice with separate activation/no-call, code-suffix/negation, and "
            "component-value axes."
        ),
        sources=(
            EvidenceSource(
                "synthesis_report",
                "results/reports/h1n_no_call_rescue_transfer_synthesis/report.md",
                "Aggregate synthesis comparing v10 with no-directive and per-packet incumbent profiles.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_residual_no_call_control_rescue_execute_v1",
                "Residual transfer packet where v10 gives executor-only gains over no-directive but trails v8.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_post_repair_no_call_control_rescue_execute_v1",
                "Post-repair transfer packet where v10 ties v7 code guard at 6/8.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1n_oracle_oblique_no_call_control_rescue_execute_v1",
                "Oblique transfer packet where v10 reaches 5/6 but trails v7 code guard.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_residual_diagnostic/diagnostic.md",
                "Residual diagnostic showing v10 is not the strict upper bound.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md",
                "Oblique diagnostic showing v10 below v7 code guard.",
            ),
        ),
    ),
    Claim(
        claim_id="C33_h1o_factorial_identifies_component_value_residue",
        claim=(
            "The H1o control-factorial slice separates remaining visual harness mechanisms: activation/no-call is "
            "already saturated, code/negation is repairable by controller wording, and component/value boundaries "
            "remain the residual hard case."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On the 12-case H1o packet, argument hints v2 and component-value guard v9 tie the strict upper bound "
            "at 9/12 exact; argument hints, hybrid label guard, and component-value guard tie executor-equivalence "
            "at 10/12; no-directive is already 4/4 exact on activation/no-call but 1/4 exact on code/negation and "
            "0/4 exact on component/value."
        ),
        limitation=(
            "H1o is still a synthetic replay-shaped micro-slice. It is mechanism-discriminative, not a population "
            "estimate across real GUIs or packaged workflows."
        ),
        next_test=(
            "Build a fresh H1p component-only holdout with more diverse component/value surfaces, then test whether "
            "component-value guard can beat argument hints without losing exact selector copying."
        ),
        sources=(
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_directive_execute_v1",
                "No-directive H1o baseline reaching 5/12 exact and 6/12 executor-equivalent, with activation already saturated.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1o_control_factorial_argument_hints_execute_v1",
                "Argument-hints H1o execution reaching 9/12 exact and 10/12 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1o_control_factorial_component_value_guard_execute_v1",
                "Component-value-guard H1o execution tying argument hints at 9/12 exact and 10/12 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1o_control_factorial_argument_hints_vs_no_directive_v1",
                "Comparison showing +0.333 exact and executor-equivalence deltas over no-directive for argument hints.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_h1o_control_factorial_diagnostic/diagnostic.md",
                "Matrix diagnostic identifying strict/equivalence upper bounds and the no-call-rescue regression.",
            ),
            EvidenceSource(
                "synthesis_report",
                "results/reports/h1o_control_factorial_synthesis/report.md",
                "Mechanism-family synthesis showing activation saturation, code/negation repairability, and component/value residue.",
            ),
        ),
    ),
    Claim(
        claim_id="C34_h1p_component_holdout_supports_component_value_domain",
        claim=(
            "The H1p component-only holdout shows that component-value-specific guidance has a real activation "
            "domain: it can outperform generic argument hints when the remaining ambiguity is specifically component "
            "label versus displayed value selection."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On the 12-case H1p component-only holdout, component-value guard v9 reaches 10/12 exact and 11/12 "
            "executor-equivalent, versus no-directive at 0/12, argument hints v2 at 6/12, no-call rescue v10 at "
            "6/12, and hybrid label guard v8 at 9/12 exact and 10/12 executor-equivalent."
        ),
        limitation=(
            "H1p is a replay-shaped synthetic component holdout. Because v9 was negative on the earlier H1n "
            "component-value slice and only tied argument hints on H1o, this supports domain specificity rather "
            "than global promotion."
        ),
        next_test=(
            "Split component-value guidance into narrower component-only wording, then transfer-test it against "
            "H1n and H1o to separate durable component disambiguation from over-broad selector prose."
        ),
        sources=(
            EvidenceSource(
                "oracle_packet",
                "results/tool_probe_replay_packets/20260510T_h1p_component_value_holdout_oracle_dry_run_v1",
                "Fresh 12-case component-only holdout spanning compact components, surface labels, and stale-selection decoys.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1p_component_value_no_directive_execute_v1",
                "No-directive H1p baseline collapsing to 0/12 exact and 0/12 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1p_component_value_argument_hints_execute_v1",
                "Argument-hints H1p execution reaching 6/12 exact and 6/12 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1p_component_value_hybrid_label_guard_execute_v1",
                "Hybrid label guard H1p execution reaching 9/12 exact and 10/12 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1p_component_value_component_value_guard_execute_v1",
                "Component-value guard H1p execution reaching 10/12 exact and 11/12 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1p_component_value_component_value_guard_vs_no_directive_v1",
                "Comparison showing +0.833 exact and +0.917 executor-equivalence deltas over no-directive.",
            ),
            EvidenceSource(
                "diagnostic_report",
                "results/reports/visual_h1p_component_value_diagnostic/diagnostic.md",
                "Matrix diagnostic ranking component-value guard above hybrid, argument hints, and no-call rescue on H1p.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1p_live_replay_summary.csv",
                "Paper-facing H1p summary table in the MLX tool-contract report bundle.",
            ),
        ),
    ),
    Claim(
        claim_id="C35_h1q_component_label_guard_is_strongest_transfer_candidate",
        claim=(
            "The H1q transfer synthesis shows that narrow component-label guidance is stronger than broad "
            "component-value prose across the current H1n/H1o/H1p component ambiguity family."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "Across 32 live replay cases, component-label guard v11 reaches 26/32 exact and 29/32 "
            "executor-equivalent successes, versus component-value guard v9 at 23/32 exact and 25/32 "
            "executor-equivalent."
        ),
        limitation=(
            "The synthesis spans three replay-shaped MLX packets, not a broad population estimate. v11 is also "
            "not a global default because it trails v9 by one executor-equivalent case on H1p and retains "
            "owner-field, state-tag, and mode-toggle residual failures."
        ),
        next_test=(
            "Build H1r around the remaining v11 miss families, especially owner-field stale selection, compact "
            "state tags, mode toggles, and exact paraphrases in H1o code/negation rows."
        ),
        sources=(
            EvidenceSource(
                "transfer_synthesis",
                "results/reports/h1q_component_label_guard_transfer_synthesis/report.md",
                "H1q synthesis aggregating H1n, H1o, and H1p component-label guard transfer results.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1",
                "H1n execution where v11 reaches 6/8 exact and 7/8 executor-equivalent successes.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1",
                "H1o execution where v11 reaches 10/12 exact and 12/12 executor-equivalent successes.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1",
                "H1p execution where v11 reaches 10/12 exact and 10/12 executor-equivalent successes.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h1q_component_label_guard_aggregate_summary.csv",
                "Paper-facing aggregate table comparing v11 against no-directive, v2, v8, v9, and v10.",
            ),
        ),
    ),
    Claim(
        claim_id="C36_h1s_residual_guard_is_targeted_not_global",
        claim=(
            "The H1s transfer gate shows that the v12 component-residual guard is a useful targeted patch, "
            "but not a global replacement for the v11 component-label guard."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "Across H1n/H1o/H1p, v12 improves strict exactness from v11's 26/32 to 27/32, but lowers "
            "executor-equivalence from v11's 29/32 to 27/32; H1n is the clearest negative transfer at "
            "-0.125 exact-rate and -0.250 executor-equivalence rate versus v11."
        ),
        limitation=(
            "H1s is still a replay-shaped synthetic transfer gate. It supports conditional routing or prompt-factor "
            "testing, not broad claims about real GUI populations or a final global prompt contract."
        ),
        next_test=(
            "Build a conditional-route or prompt-factorial slice that keeps v11 as the general component-label "
            "profile and applies v12 residual wording only to code-label and nonstandard component-class contexts."
        ),
        sources=(
            EvidenceSource(
                "transfer_synthesis",
                "results/reports/h1s_component_residual_transfer_synthesis/report.md",
                "H1s synthesis aggregating v12 transfer over H1n, H1o, and H1p after the local H1r win.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1",
                "H1n v12 execution showing negative transfer versus v11 at 5/8 exact and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1",
                "H1o v12 execution reaching 11/12 exact and executor-equivalent, below v11 executor-equivalence.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1",
                "H1p v12 execution improving over v11 at 11/12 exact and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1n_vs_component_label_guard_v1",
                "Pairwise H1n comparison showing the strongest negative transfer versus v11.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h1s_component_residual_transfer_aggregate.csv",
                "Paper-facing aggregate table comparing v12 against v11 and no-directive across H1n/H1o/H1p.",
            ),
        ),
    ),
    Claim(
        claim_id="C37_h1x_breaks_v11_saturation_but_supports_routing",
        claim=(
            "The H1x replay gate breaks v11 saturation on oblique stale-field pressure, but supports routed "
            "residual help rather than a global v12 prompt replacement."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H1x, no-directive reaches 2/8 exact and executor-equivalent, v11 reaches 7/8, v12 reaches "
            "8/8, and v15 reaches 6/8 exact with 7/8 executor-equivalent."
        ),
        limitation=(
            "H1x is a focused replay-shaped synthetic packet and should be interpreted together with H1s, which "
            "already shows v12's broader negative transfer when promoted globally."
        ),
        next_test=(
            "Build H1y as a routed residual-helper test that keeps v11 as the default and activates v12-style "
            "residual wording only on oblique stale-field and nonstandard-class contexts."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h1x_v11_breaker_synthesis/report.md",
                "H1x synthesis comparing no-directive, v11, v12, and v15 on the oblique v11-breaker packet.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1",
                "No-directive H1x baseline showing only activation/no-call rows solve without catalog help.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_label_guard_execute_v1",
                "V11 H1x replay showing the remaining oblique stale-field wrong-tool miss.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_residual_guard_execute_v1",
                "V12 H1x replay saturating the local packet.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h1x_v11_breaker_code_label_exact_guard_execute_v1",
                "V15 H1x replay showing lower strict exactness and partial executor-equivalent rescue.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h1x_v11_breaker_packet_summary.csv",
                "Paper-facing H1x packet table in the generated MLX report.",
            ),
            EvidenceSource(
                "report_figure",
                "results/reports/mlx_tool_contract_harnessing/figures/h1x_v11_breaker_gate.svg",
                "Paper-facing H1x replay gate figure in the generated MLX report.",
            ),
        ),
    ),
    Claim(
        claim_id="C38_h2a_controller_stale_selection_gate_is_causal",
        claim=(
            "The H1y/H2a replay gate shows stale visual selection handling is better solved as a controller-side "
            "runtime mediation than as additional catalog prose."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On the same 10-case H1y packet, no-directive reaches 0/10 exact, v11 reaches 5/10, v12 reaches "
            "7/10, v16 and v17 reach 5/10, and H2a reaches 8/10 exact and executor-equivalent."
        ),
        limitation=(
            "This claim is local to the H1y routed-residual packet; the transfer result is tracked separately so "
            "local causality and held-out generalization remain distinct."
        ),
        next_test=(
            "Use the H2a transfer gate to decide whether the helper should be promoted as a scoped controller "
            "mechanism, then isolate the remaining argument-alias/code-label misses without leaking expected labels."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h1y_routed_residual_synthesis/report.md",
                "H1y/H2a synthesis comparing no-directive, v11, v12, v16, v17, and the controller stale-selection gate.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1y_execute_v1",
                "H2a live execution reaching 8/10 exact and executor-equivalent on H1y.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2a_visual_stale_selection_gate_vs_component_label_guard_on_h1y_v1",
                "Direct comparison showing H2a gains three exact successes over v11.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2a_visual_stale_selection_gate_vs_component_residual_guard_on_h1y_v1",
                "Direct comparison showing H2a gains one exact success over v12.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h1y_routed_residual_packet_summary.csv",
                "Paper-facing H1y/H2a packet table in the generated MLX report.",
            ),
            EvidenceSource(
                "report_figure",
                "results/reports/mlx_tool_contract_harnessing/figures/h1y_routed_residual_gate.svg",
                "Paper-facing H1y/H2a replay gate figure in the generated MLX report.",
            ),
        ),
    ),
    Claim(
        claim_id="C39_h2a_stale_selection_gate_transfers_with_better_executor_profile",
        claim=(
            "The H2a stale-selection controller gate transfers beyond its H1y fit packet and gives the cleanest "
            "current visual helper profile when strict exactness and executor-equivalence are reported together."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "Across H1n/H1o/H1p/H1x, H2a reaches 35/40 strict exact and 38/40 executor-equivalent, versus "
            "no-directive at 12/40 and 14/40, v11 at 33/40 and 36/40, and v12 at 35/40 and 35/40."
        ),
        limitation=(
            "H2a still leaves five transfer residual rows, mostly exact alias/code-label disagreements; two H1p "
            "rows are not executor-equivalent and should not be treated as solved."
        ),
        next_test=(
            "Build the next residual packet around exact alias/code-label fidelity: result pill, alert s92, badge "
            "c08, state tag, and mode toggle, with no expected-call or benchmark-answer access."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2a_stale_selection_transfer_synthesis/report.md",
                "H2a transfer synthesis separating local H1y fit from held-out H1n/H1o/H1p/H1x transfer.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1n_component_value_execute_v1",
                "H2a live execution reaching 7/8 exact and 8/8 executor-equivalent on H1n component-value residual.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1o_execute_v1",
                "H2a live execution reaching 10/12 exact and 12/12 executor-equivalent on H1o control-factorial.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1p_execute_v1",
                "H2a live execution reaching 10/12 exact and 10/12 executor-equivalent on H1p component-value holdout.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1x_execute_v1",
                "H2a live execution reaching 8/8 exact and 8/8 executor-equivalent on the H1x v11-breaker packet.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h2a_stale_selection_transfer_aggregate_summary.csv",
                "Paper-facing H2a transfer aggregate table in the generated MLX report.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h2a_stale_selection_transfer_residual_rows.csv",
                "Paper-facing H2a residual table identifying the remaining exact alias/code-label misses.",
            ),
            EvidenceSource(
                "report_figure",
                "results/reports/mlx_tool_contract_harnessing/figures/h2a_stale_selection_transfer_gate.svg",
                "Paper-facing H2a transfer gate figure in the generated MLX report.",
            ),
        ),
    ),
    Claim(
        claim_id="C40_h2b_residual_exactness_favors_scoped_v12_not_global_h2a",
        claim=(
            "The H2b residual-exactness gate shows that the remaining post-H2a visual residuals need a scoped "
            "alias/code-label route, not a global stale-selection controller default."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On the five-row H2b residual packet, v12 reaches 4/5 strict exact and 4/5 executor-equivalent, "
            "v9 reaches 3/5 strict and 4/5 executor-equivalent, v15 reaches 3/5 strict, H2a reaches 0/5 "
            "strict and 3/5 executor-equivalent, and no-directive reaches 1/5 strict and 2/5 executor-equivalent."
        ),
        limitation=(
            "H2b is deliberately selected from the five H2a residual rows, so it supports a scoped H2c routing "
            "hypothesis rather than a broad population estimate; H1s still warns against global v12 promotion."
        ),
        next_test=(
            "Build H2c as a conditional route that applies v12-like residual language only when alias/code-label "
            "exactness is the likely failure mechanism, while preserving H2a for stale-selection repair."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2b_residual_exactness_synthesis/report.md",
                "H2b synthesis comparing residual exactness profiles on the five post-H2a residual cases.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2b_residual_exactness_no_directive_execute_v1",
                "No-directive H2b execution reaching 1/5 strict and 2/5 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_label_guard_execute_v1",
                "v11 component-label guard H2b execution reaching 0/5 strict and 3/5 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_residual_guard_execute_v1",
                "v12 component-residual guard H2b execution reaching 4/5 strict and 4/5 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2b_residual_exactness_component_value_guard_execute_v1",
                "v9 component-value guard H2b execution tying v12 on executor-equivalence but missing strict exactness.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2b_residual_exactness_h2a_execute_v1",
                "H2a controller gate H2b execution showing stale-selection mediation does not solve residual exactness.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h2b_residual_exactness_packet_summary.csv",
                "Paper-facing H2b packet summary table in the generated MLX report.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h2b_residual_exactness_case_matrix.csv",
                "Paper-facing H2b case matrix separating strict exactness from executor-equivalence.",
            ),
            EvidenceSource(
                "report_figure",
                "results/reports/mlx_tool_contract_harnessing/figures/h2b_residual_exactness_gate.svg",
                "Paper-facing H2b residual exactness gate figure in the generated MLX report.",
            ),
        ),
    ),
    Claim(
        claim_id="C41_h2c_scoped_residual_gate_saturates_h2b_but_needs_transfer",
        claim=(
            "The H2c scoped residual gate solves the five-row H2b residual packet locally, but the evidence still "
            "requires a held-out transfer gate before any global or default promotion."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2b, H2c reaches 5/5 strict exact and 5/5 executor-equivalent, versus v12 at 4/5 and 4/5, "
            "v9 at 3/5 and 4/5, H2a at 0/5 and 3/5, and no-directive at 1/5 and 2/5."
        ),
        limitation=(
            "H2c is fit to the same five residual rows selected from H2a transfer, so it is a local mechanism "
            "result. It does not override the earlier H1s warning that residual wording can hurt transfer."
        ),
        next_test=(
            "Run a minimal H2c transfer gate over H1n/H1o/H1p/H1x residual families and compare strict exactness, "
            "executor-equivalence, and stale-selection behavior against H2a and v12."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2c_scoped_residual_synthesis/report.md",
                "H2c synthesis showing local saturation on the H2b residual exactness packet.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2c_scoped_residual_gate_on_h2b_execute_v1",
                "H2c live execution reaching 5/5 strict and executor-equivalent on H2b.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_component_residual_guard_on_h2b_v1",
                "H2c comparison against v12 showing a one-case strict/executor gain.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_h2a_on_h2b_v1",
                "H2c comparison against H2a showing residual exactness remains separate from stale-selection mediation.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2c_scoped_residual_gate_vs_component_value_guard_on_h2b_v1",
                "H2c comparison against v9 showing it beats the executor-tie row on strict exactness.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h2c_scoped_residual_packet_summary.csv",
                "Paper-facing H2c packet summary table in the generated MLX report.",
            ),
            EvidenceSource(
                "report_table",
                "results/reports/mlx_tool_contract_harnessing/tables/h2c_scoped_residual_comparison_summary.csv",
                "Paper-facing H2c comparison summary table in the generated MLX report.",
            ),
            EvidenceSource(
                "report_figure",
                "results/reports/mlx_tool_contract_harnessing/figures/h2c_scoped_residual_gate.svg",
                "Paper-facing H2c scoped residual gate figure in the generated MLX report.",
            ),
        ),
    ),
    Claim(
        claim_id="C42_h2d_class_preserving_route_repairs_h2c_transfer_but_costs_h2b_exactness",
        claim=(
            "The H2d class-preserving route repairs H2c's held-out H1x component-class transfer miss, but it is "
            "not a clean global replacement because it gives back one strict H2b exact row."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "H2d reaches 8/8 strict exact and 8/8 executor-equivalent on H1x versus H2c at 7/8 and 7/8, "
            "but H2d reaches 4/5 strict exact on H2b versus H2c at 5/5 while preserving 5/5 executor-equivalence."
        ),
        limitation=(
            "H2d was designed after observing H2c's H1x class-swap miss, so it supports a targeted mechanism "
            "interpretation rather than a broad promotion decision."
        ),
        next_test=(
            "Build route arbitration that preserves H2c's compact code/value exactness while retaining H2d's "
            "class-preserving behavior on held-out component-class transfer."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2d_transfer_tradeoff_synthesis/report.md",
                "H2d synthesis showing the H2b/H1x tradeoff and the non-equivalent H2c transfer miss.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h2b_execute_v1",
                "H2d live execution reaching 4/5 strict and 5/5 executor-equivalent on H2b.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2d_class_preserving_route_on_h1x_execute_v1",
                "H2d live execution reaching 8/8 strict and executor-equivalent on H1x.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_h2c_on_h2b_v1",
                "Direct H2d-vs-H2c comparison showing the one-row H2b strict exactness cost.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2d_class_preserving_route_vs_h2c_on_h1x_v1",
                "Direct H2d-vs-H2c comparison showing H2d repairs the H1x transfer miss.",
            ),
        ),
    ),
    Claim(
        claim_id="C43_h2e_route_arbitration_reconciles_h2c_h2d_tradeoff",
        claim=(
            "The H2e route-arbitrated residual profile reconciles the observed H2c/H2d tradeoff on the current "
            "H2b and H1x gates while preserving executor-equivalence."
        ),
        status="supported_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "H2e reaches 5/5 strict exact and 5/5 executor-equivalent on H2b, plus 8/8 strict exact and 8/8 "
            "executor-equivalent on H1x; it has zero non-exact rows across those two packets."
        ),
        limitation=(
            "H2e was built from the H2c/H2d failure analysis, so the current result is mechanism evidence on "
            "two gates rather than a fresh-holdout generalization result."
        ),
        next_test=(
            "Promote H2e only into a newly authored H2f route-arbitration holdout with unseen code labels, "
            "component classes, stale-id decoys, and displayed-value distractors."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2e_route_arbitration_synthesis/report.md",
                "H2e synthesis showing simultaneous H2b and H1x saturation plus counterfactual miss coverage.",
            ),
            EvidenceSource(
                "report_figure",
                "results/reports/h2e_route_arbitration_synthesis/figures/h2e_route_arbitration_gate.svg",
                "Paper-facing H2e route-arbitration gate figure.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h2b_execute_v1",
                "H2e live execution reaching 5/5 strict and executor-equivalent on H2b.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2e_route_arbitration_on_h1x_execute_v1",
                "H2e live execution reaching 8/8 strict and executor-equivalent on H1x.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_h2c_on_h1x_v1",
                "Direct H2e-vs-H2c comparison showing transfer repair over H2c.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2e_route_arbitration_vs_h2d_on_h2b_v1",
                "Direct H2e-vs-H2d comparison showing H2e recovers the H2b exact row H2d missed.",
            ),
        ),
    ),
    Claim(
        claim_id="C44_h2f_holdout_breaks_h2e_global_promotion",
        claim=(
            "The fresh H2f route-arbitration holdout breaks H2e's apparent top-line saturation and localizes the "
            "remaining MLX Gemma failure to component-identity query binding rather than missing tool use."
        ),
        status="negative_result_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2f, H2e reaches 6/10 strict exact and 6/10 executor-equivalent, ties H2c at 6/10, but remains "
            "well above the no-directive floor at 1/10. All four H2e non-exact rows call the right tool with a "
            "target_query that substitutes a displayed value or alias for the requested component identity."
        ),
        limitation=(
            "H2f is a fresh authored holdout with ten cases, so it is stronger than replaying saturated rows but "
            "still needs a follow-up H2g mechanism test to show that a component-identity query contract repairs "
            "the failure without regressing stale-selection and activation-panel cases."
        ),
        next_test=(
            "Build H2g around a component-identity query contract: when the user asks for a component class or "
            "visible label, the live target_query must preserve that requested phrase instead of collapsing to "
            "the component value or a nearby alias."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2f_route_arbitration_holdout_synthesis/report.md",
                "H2f synthesis showing the fresh-holdout failure, causal floor, and component-identity diagnosis.",
            ),
            EvidenceSource(
                "report_figure",
                "results/reports/h2f_route_arbitration_holdout_synthesis/figures/h2f_holdout_profile_bars.svg",
                "Paper-facing H2f profile bar figure.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2e_execute_v1",
                "H2e live execution reaching 6/10 strict and executor-equivalent on H2f.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2f_route_arbitration_h2c_execute_v1",
                "H2c live execution tying H2e at 6/10 strict and executor-equivalent on H2f.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2f_route_arbitration_no_directive_execute_v1",
                "No-directive live execution establishing the 1/10 H2f floor.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2f_route_arbitration_h2e_vs_h2c_v1",
                "Direct H2e-vs-H2c comparison showing no H2f lift from route arbitration.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2f_route_arbitration_h2e_vs_no_directive_v1",
                "Direct H2e-vs-no-directive comparison showing the controller stack remains causal.",
            ),
        ),
    ),
    Claim(
        claim_id="C45_h2g_component_identity_contract_is_partial_executor_gain",
        claim=(
            "The H2g component-identity query contract produces a partial executor-equivalence gain on H2f but "
            "does not repair strict component-identity query fidelity."
        ),
        status="negative_result_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2f, H2g stays at 6/10 strict exact versus H2e but improves executor-equivalence from 6/10 to "
            "7/10. The improvement comes from `resolution badge Deferred` being executor-valid, while the "
            "remaining non-exact rows still include `result tile` -> `Blocked`, `state marker` -> `lifecycle "
            "state marker`, and `mode switch` -> `mode toggle`."
        ),
        limitation=(
            "H2g has only been executed on H2f. Because it does not improve strict exactness on the acceptance "
            "holdout, H2b/H1x backtests are lower priority than designing a stronger exact-query contract."
        ),
        next_test=(
            "Build H2h with explicit negative examples for value substitution and alias expansion, then rerun on "
            "H2f before any H2b/H1x regression backtest."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2f_route_arbitration_holdout_synthesis/report.md",
                "Updated H2f synthesis including H2g as a partial executor-equivalence gain with zero strict gain.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2g_component_identity_query_contract_on_h2f_execute_v1",
                "H2g live execution reaching 6/10 strict and 7/10 executor-equivalent on H2f.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2g_component_identity_query_contract_vs_h2e_on_h2f_v1",
                "Direct H2g-vs-H2e comparison showing zero strict lift and +0.1 executor-equivalence lift.",
            ),
        ),
    ),
    Claim(
        claim_id="C46_h2h_negative_examples_repair_h2f_but_fail_global_transfer",
        claim=(
            "The H2h component-identity negative-example contract strongly repairs the fresh H2f holdout, but "
            "fails global promotion because it regresses the prior H2b and H1x transfer gates."
        ),
        status="supported_scoped_negative_global_promotion",
        evidence_strength="strong_internal",
        primary_metric=(
            "H2h improves H2f from H2e/H2g's 6/10 strict exactness to 9/10 strict and executor-equivalent, "
            "but falls to 3/5 on H2b versus H2e's 5/5 and 6/8 on H1x versus H2e's 8/8."
        ),
        limitation=(
            "H2h is evidence for a causal prompt-contract repair on one fresh holdout, not a deployable default. "
            "The transfer regressions show that explicit negative examples can over-constrain nearby component "
            "classes and code-label rows."
        ),
        next_test=(
            "Build a conditional arbitration profile that keeps H2e as the default and activates H2h-style "
            "negative examples only when the prompt explicitly asks for a displayed-value component identity."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2h_component_identity_tradeoff_synthesis/report.md",
                "Dedicated H2h synthesis showing the H2f repair and H2b/H1x transfer regressions.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2h_component_identity_tradeoff_synthesis/figures/h2h_tradeoff_gate.svg",
                "Figure summarizing H2h's scoped improvement and transfer regressions.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2h_component_identity_negative_examples_on_h2f_execute_v1",
                "H2h live execution reaching 9/10 strict and executor-equivalent on H2f.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h2f_v1",
                "Direct H2h-vs-H2e comparison showing +0.3 exact-rate lift on H2f.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h2b_v1",
                "Direct H2h-vs-H2e comparison showing -0.4 exact-rate regression on H2b.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2h_component_identity_negative_examples_vs_h2e_on_h1x_v1",
                "Direct H2h-vs-H2e comparison showing -0.25 exact-rate regression on H1x.",
            ),
        ),
    ),
    Claim(
        claim_id="C47_h2i_conditional_component_arbitration_does_not_preserve_h2f_repair",
        claim=(
            "The H2i conditional component-identity arbitration prompt does not preserve H2h's fresh-H2f repair, "
            "showing that the safe conditionalization problem remains unsolved."
        ),
        status="negative_result_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2f, H2i reaches 6/10 strict and executor-equivalent, tying H2e and trailing H2h's 9/10 by "
            "0.3 exact-rate. It fails through target-query drift: `alert t47` -> `Escalated`, `result tile` -> "
            "`result tile for Blocked`, `resolution badge` -> `resolution badge for Deferred`, and `state marker` "
            "-> `lifecycle state marker`."
        ),
        limitation=(
            "H2i was intentionally stopped at the H2f gate and not backtested on H2b/H1x, because it did not "
            "improve the acceptance holdout."
        ),
        next_test=(
            "Design the next candidate around a more structural route gate or controller-visible query-normalization "
            "hypothesis, rather than simply adding softer conditional prompt prose."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2f_route_arbitration_holdout_synthesis/report.md",
                "Updated H2f synthesis including H2i as a negative conditionalization result.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260510T_h2i_conditional_component_arbitration_on_h2f_execute_v1",
                "H2i live execution tying H2e at 6/10 on H2f.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2h_on_h2f_v1",
                "Direct H2i-vs-H2h comparison showing -0.3 exact-rate regression on H2f.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260510T_h2i_conditional_component_arbitration_vs_h2e_on_h2f_v1",
                "Direct H2i-vs-H2e comparison showing zero exact-rate gain on H2f.",
            ),
        ),
    ),
    Claim(
        claim_id="C48_h2j_target_query_normalization_repairs_h2f_and_preserves_transfer",
        claim=(
            "The H2j controller-visible target-query normalization gate repairs the fresh H2f component-identity "
            "holdout while preserving the H2b and H1x transfer gates that rejected global H2h promotion."
        ),
        status="supported_current_packets_next_harder_holdout",
        evidence_strength="strong_internal",
        primary_metric=(
            "H2j reaches 10/10 strict and executor-equivalent on H2f, improving by +0.4 exact-rate versus H2e "
            "and +0.1 versus H2h. It also reaches 5/5 on H2b and 8/8 on H1x, tying H2e on both transfer gates "
            "while beating H2h by +0.4 on H2b and +0.25 on H1x."
        ),
        limitation=(
            "H2j is validated on H2f plus the existing H2b/H1x transfer gates, not on a new post-H2j holdout. "
            "The normalizer's next risk is prompts where the same visual label appears as both a requested target "
            "and a negated or before-reading decoy."
        ),
        next_test=(
            "Build an H2k harder holdout with adversarial prompt/state label overlap, then ablate the target-query "
            "normalizer and stale-selection gate separately to quantify controller dependence."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2j_target_query_normalization_transfer_synthesis/report.md",
                "Dedicated H2j synthesis showing H2f repair, H2b/H1x transfer preservation, and controller interventions.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2j_target_query_normalization_transfer_synthesis/figures/h2j_transfer_gate.svg",
                "Figure summarizing H2j closure of H2f and preservation of H2b/H1x.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2f_execute_v2",
                "H2j live execution reaching 10/10 strict and executor-equivalent on H2f.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h2b_execute_v2",
                "H2j live execution preserving H2b at 5/5 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2j_target_query_normalization_on_h1x_execute_v1",
                "H2j live execution preserving H1x at 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2e_on_h2f_v2",
                "Direct H2j-vs-H2e comparison showing +0.4 exact-rate lift on H2f.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2h_on_h2f_v2",
                "Direct H2j-vs-H2h comparison showing +0.1 exact-rate lift on H2f.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2e_on_h2b_v2",
                "Direct H2j-vs-H2e comparison showing transfer preservation on H2b.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2h_on_h2b_v2",
                "Direct H2j-vs-H2h comparison showing +0.4 exact-rate lift on H2b.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2e_on_h1x_v1",
                "Direct H2j-vs-H2e comparison showing transfer preservation on H1x.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2j_target_query_normalization_vs_h2h_on_h1x_v1",
                "Direct H2j-vs-H2h comparison showing +0.25 exact-rate lift on H1x.",
            ),
        ),
    ),
    Claim(
        claim_id="C49_h2k_target_decoy_overlap_supports_h2j_structural_normalization",
        claim=(
            "The H2k target/decoy overlap gate gives fresh support for H2j's controller-visible target-query "
            "normalization mechanism, separating it from both H2e route arbitration and H2h prompt-side negative "
            "examples on adversarial label-overlap cases."
        ),
        status="supported_current_packets_helper_ablation_passed",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2k, H2j reaches 8/8 strict and executor-equivalent, H2h reaches 6/8 strict and executor-equivalent, "
            "and H2e reaches 3/8 strict with 6/8 executor-equivalent. H2j improves exact-rate by +0.625 versus H2e "
            "and +0.25 versus H2h, with 5 target-query-normalization interventions and 0 stale-selection interventions. "
            "The matched H2j-without-stale-selection ablation also reaches 8/8, with 0.0 exact and executor-equivalence "
            "deltas versus full H2j."
        ),
        limitation=(
            "H2k v1 is an 8-case replay-shaped holdout designed around target/decoy overlap, not a packaged workflow "
            "population. H2e is the no-target-normalizer control on this slice, so the remaining limitation is broader "
            "transfer and over-normalization pressure, not stale-selection causality on H2k."
        ),
        next_test=(
            "Build the next fresh holdout around target-query-normalization overreach and then backtest the no-stale "
            "ablation on H2f/H2b/H1x only if considering removing the stale-selection gate globally."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2k_target_decoy_overlap_synthesis/report.md",
                "Dedicated H2k synthesis comparing H2e, H2h, and H2j on target/decoy overlap pressure.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2k_target_decoy_overlap_synthesis/figures/h2k_target_decoy_overlap_gate.svg",
                "Figure summarizing H2k exact-rate separation across H2e, H2h, and H2j.",
            ),
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260512T_h2k_target_decoy_overlap_dry_run_v1",
                "The 8-case H2k dry-run packet defining adversarial target/decoy overlap prompts and expected calls.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_execute_v1",
                "H2j live execution reaching 8/8 strict and executor-equivalent on H2k.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2j_no_stale_gate_execute_v1",
                "Matched stale-selection-gate-off H2j ablation reaching 8/8 strict and executor-equivalent on H2k.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2e_execute_v1",
                "H2e live execution reaching 3/8 strict and 6/8 executor-equivalent on H2k.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2k_target_decoy_overlap_h2h_execute_v1",
                "H2h live execution reaching 6/8 strict and executor-equivalent on H2k.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_h2e_v1",
                "Direct H2j-vs-H2e comparison showing +0.625 exact-rate and +0.25 executor-equivalence-rate gains.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_h2h_v1",
                "Direct H2j-vs-H2h comparison showing +0.25 exact-rate and executor-equivalence-rate gains.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2k_target_decoy_overlap_h2j_vs_no_stale_gate_v1",
                "Matched full-H2j-vs-no-stale-gate comparison showing zero exact and executor-equivalence deltas.",
            ),
        ),
    ),
    Claim(
        claim_id="C50_h2l_overreach_holdout_supports_target_normalization_scope",
        claim=(
            "The H2l target-normalization overreach holdout did not expose over-stripping in the current H2j "
            "controller, and instead shows a scoped target-query-normalization repair on one H2e regression guard."
        ),
        status="supported_current_packets_next_harder_holdout",
        evidence_strength="moderate_internal",
        primary_metric=(
            "On H2l, full H2j and H2j without the stale-selection gate both reach 8/8 strict and executor-equivalent, "
            "while H2e reaches 7/8. H2j improves exact-rate and executor-equivalence-rate by +0.125 versus H2e, "
            "ties the no-stale ablation with 0.0 deltas, and records 1 target-query-normalization intervention "
            "(`critical chip` to `status badge`) with 0 stale-selection interventions."
        ),
        limitation=(
            "H2l v1 is an 8-case replay-shaped overreach holdout with explicit target-is wording and deterministic "
            "cases; it is positive control evidence for scope, not a population estimate or proof that all "
            "over-normalization risks are closed."
        ),
        next_test=(
            "Build H2m with less direct target phrasing, ambiguous local context, or repeated seed variants before "
            "treating target-normalization overreach as resolved."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2l_target_normalization_overreach_synthesis/report.md",
                "Dedicated H2l synthesis comparing H2e, H2j, and H2j without stale-selection on overreach pressure.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2l_target_normalization_overreach_synthesis/figures/h2l_target_normalization_overreach_gate.svg",
                "Figure summarizing H2l exact-rate separation and no-stale tie.",
            ),
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260512T_h2l_target_normalization_overreach_dry_run_v1",
                "The 8-case H2l dry-run packet defining value-bearing target and alias-is-target overreach cases.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2j_execute_v1",
                "Full H2j live execution reaching 8/8 strict and executor-equivalent on H2l.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2j_no_stale_gate_execute_v1",
                "Matched stale-selection-gate-off H2j execution reaching 8/8 strict and executor-equivalent on H2l.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2l_target_normalization_overreach_h2e_execute_v1",
                "H2e live execution reaching 7/8 strict and executor-equivalent on H2l.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2l_target_normalization_overreach_h2j_vs_h2e_v1",
                "Direct H2j-vs-H2e comparison showing +0.125 exact and executor-equivalence-rate gains.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2l_target_normalization_overreach_h2j_vs_no_stale_gate_v1",
                "Matched full-H2j-vs-no-stale-gate comparison showing zero exact and executor-equivalence deltas.",
            ),
        ),
    ),
    Claim(
        claim_id="C51_h2m_less_direct_overreach_rejects_current_target_normalization_scope",
        claim=(
            "The H2m less-direct overreach holdout rejects treating the current H2j target-query normalizer as "
            "globally safe: it helps some contextual labels but over-strips value-bearing labels when the prompt "
            "does not use direct target-is wording."
        ),
        status="negative_result_current_packets",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2m, full H2j and H2j without stale-selection both reach 3/8 strict and 3/8 executor-equivalent. "
            "H2e reaches 1/8 strict and 3/8 executor-equivalent, so H2j improves exact-rate by +0.25 but has "
            "0.0 executor-equivalence-rate delta. Full H2j records 5 target-query-normalization interventions, "
            "0 stale-selection interventions, and 3 value-bearing over-strip rows."
        ),
        limitation=(
            "H2m is an 8-case replay-shaped less-direct packet, not a population estimate. It rejects the current "
            "normalization scope under this wording regime, while still preserving evidence that scoped "
            "normalization repaired H2k/H2l-style contextual aliases."
        ),
        next_test=(
            "Build H2n as a scoped target-normalization policy that preserves H2k/H2l contextual-label repairs "
            "but refuses to shorten value-bearing requests such as `result badge Blocked`, `state tag Closed`, "
            "and `priority badge Critical`."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2m_less_direct_overreach_synthesis/report.md",
                "Dedicated H2m synthesis showing exact-rate gain, executor-equivalence tie, and over-strip rows.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2m_less_direct_overreach_synthesis/figures/h2m_less_direct_overreach_gate.svg",
                "Figure summarizing the H2m exact-rate collapse relative to H2l saturation.",
            ),
            EvidenceSource(
                "tool_probe_replay_packet",
                "results/tool_probe_replay_packets/20260512T_h2m_less_direct_target_normalization_overreach_dry_run_v1",
                "The 8-case H2m dry-run packet with less-direct value-bearing, alias, and regression-guard prompts.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2j_execute_v1",
                "Full H2j live execution reaching 3/8 strict and executor-equivalent on H2m.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2j_no_stale_gate_execute_v1",
                "Matched stale-selection-gate-off H2j execution tying full H2j on H2m.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2m_less_direct_target_normalization_overreach_h2e_execute_v1",
                "H2e live execution reaching 1/8 strict and 3/8 executor-equivalent on H2m.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_h2e_v1",
                "Direct H2j-vs-H2e comparison showing +0.25 exact-rate and 0.0 executor-equivalence-rate deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2m_less_direct_target_normalization_overreach_h2j_vs_no_stale_gate_v1",
                "Matched full-H2j-vs-no-stale-gate comparison showing zero exact and executor-equivalence deltas.",
            ),
        ),
    ),
    Claim(
        claim_id="C52_h2n_scoped_target_normalization_improves_executor_equivalence_without_strict_repair",
        claim=(
            "The H2n scoped target-query normalizer is a cleaner controller candidate than H2j on the current "
            "target-normalization line: it blocks value-bearing over-strips and preserves transfer gates, but it "
            "does not yet repair strict exactness on the less-direct H2m slice."
        ),
        status="supported_current_packets_scope_candidate",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2m, H2n ties H2j at 3/8 strict exactness but improves executor-equivalence from 3/8 to 5/8. "
            "Against H2e, H2n improves strict exactness by +0.25 and executor-equivalence by +0.25. It preserves "
            "H2k at 8/8, H2l at 8/8, and H2f at 10/10 with zero exact-rate delta versus H2j on each transfer gate."
        ),
        limitation=(
            "H2n is a no-op/blocking scope policy, not a canonical target-query construction policy. The remaining "
            "H2m strict misses include `result badge Blocked`, `mode toggle Manual`, and `result tile`, so strict "
            "repair still needs a targeted value-bearing target synthesis gate."
        ),
        next_test=(
            "Build H2o as a canonical value-bearing target-query synthesis gate that only fires when a longer "
            "visual label is recoverable from the image-state catalog and the prompt evidence asks for that value."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2n_scoped_target_normalization_synthesis/report.md",
                "Dedicated H2n synthesis showing H2m executor-equivalence gain, strict tie, and transfer preservation.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2n_scoped_target_normalization_synthesis/figures/h2n_scoped_target_normalization_gate.svg",
                "Figure summarizing H2n exact transfer gates and the H2m strict boundary.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2m_execute_v1",
                "H2n live execution on H2m reaching 3/8 strict and 5/8 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2k_execute_v1",
                "H2n live execution on H2k preserving 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2l_execute_v1",
                "H2n live execution on H2l preserving 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2n_scoped_target_normalization_on_h2f_execute_v1",
                "H2n live execution on H2f preserving 10/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2m_v1",
                "Direct H2n-vs-H2j comparison on H2m showing zero strict delta and +0.25 executor-equivalence delta.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2e_on_h2m_v1",
                "Direct H2n-vs-H2e comparison on H2m showing +0.25 strict and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2k_v1",
                "Transfer comparison showing H2n ties H2j on H2k at 8/8.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2l_v1",
                "Transfer comparison showing H2n ties H2j on H2l at 8/8.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2n_scoped_target_normalization_vs_h2j_on_h2f_v1",
                "Transfer comparison showing H2n ties H2j on H2f at 10/10.",
            ),
        ),
    ),
    Claim(
        claim_id="C53_h2o_value_bearing_target_synthesis_repairs_h2m_strict_with_contextual_alias_residue",
        claim=(
            "Selective value-bearing target-query synthesis repairs the strict H2m target-normalization boundary "
            "without regressing the saturated H2k, H2l, and H2f transfer gates."
        ),
        status="supported_current_packets_scope_candidate",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2m, H2o improves strict exactness from H2n's 3/8 to 7/8 and executor-equivalence from 5/8 to "
            "7/8. It improves strict exactness by +0.50 versus H2j and H2n, by +0.75 versus H2e, and preserves "
            "H2k at 8/8, H2l at 8/8, and H2f at 10/10 with zero exact-rate delta versus H2j on transfer packets."
        ),
        limitation=(
            "The remaining H2m miss is not a value-bearing label construction miss. It is a contextual surface-type "
            "alias row where the model keeps `Blocked` instead of targeting `result tile`, so a distinct H2p alias "
            "routing test is still needed before calling target-query control closed."
        ),
        next_test=(
            "Build H2p as a contextual surface-type alias routing slice that separates displayed values from "
            "surface-class aliases such as tile-style result surface -> result tile."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2o_value_bearing_target_synthesis/report.md",
                "Dedicated H2o synthesis showing H2m strict repair, transfer preservation, and one alias residue.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2o_value_bearing_target_synthesis/figures/h2o_value_bearing_target_synthesis_gate.svg",
                "Figure summarizing H2o exact-rate gains and transfer gates.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2o_value_bearing_target_synthesis_on_h2m_execute_v1",
                "H2o live execution on H2m reaching 7/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2o_value_bearing_target_synthesis_on_h2k_execute_v1",
                "H2o live execution on H2k preserving 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2o_value_bearing_target_synthesis_on_h2l_execute_v1",
                "H2o live execution on H2l preserving 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2o_value_bearing_target_synthesis_on_h2f_execute_v1",
                "H2o live execution on H2f preserving 10/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2o_value_bearing_target_synthesis_vs_h2n_on_h2m_v1",
                "Direct H2o-vs-H2n comparison on H2m showing +0.50 strict and +0.25 executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2m_v1",
                "Direct H2o-vs-H2j comparison on H2m showing +0.50 strict and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2o_value_bearing_target_synthesis_vs_h2e_on_h2m_v1",
                "Direct H2o-vs-H2e comparison on H2m showing +0.75 strict and +0.50 executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2k_v1",
                "Transfer comparison showing H2o ties H2j on H2k at 8/8.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2l_v1",
                "Transfer comparison showing H2o ties H2j on H2l at 8/8.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2o_value_bearing_target_synthesis_vs_h2j_on_h2f_v1",
                "Transfer comparison showing H2o ties H2j on H2f at 10/10.",
            ),
        ),
    ),
    Claim(
        claim_id="C54_h2p_contextual_surface_alias_routing_saturates_h2m_without_transfer_regression",
        claim=(
            "A narrow contextual surface-alias router closes the remaining H2m target-normalization boundary "
            "without regressing the H2k, H2l, or H2f transfer gates."
        ),
        status="supported_current_packets_scope_candidate",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2m, H2p improves strict exactness from H2o's 7/8 to 8/8 and executor-equivalence from 7/8 "
            "to 8/8. It adds +0.125 exact and executor-equivalence deltas versus H2o, +0.625 strict versus "
            "H2n and H2j, +0.875 strict versus H2e, and preserves H2k at 8/8, H2l at 8/8, and H2f at 10/10 "
            "with zero exact-rate delta versus H2o on transfer packets."
        ),
        limitation=(
            "H2p is intentionally narrow: it only addresses prompt-evidenced surface-class aliases such as "
            "tile-style result surface -> result tile. It closes the current H2m packet, but the next research "
            "question should be a harder H1/H2 slice that tests whether this target-control stack remains useful "
            "outside replay-shaped visual layout rows."
        ),
        next_test=(
            "Define a harder post-H2p H1/H2 slice that combines surface aliases, value-bearing labels, stale "
            "selection pressure, and packaged workflow attribution so the current top-line saturation breaks again."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2p_contextual_surface_alias_routing_synthesis/report.md",
                "Dedicated H2p synthesis showing H2m saturation, transfer preservation, and the single alias intervention.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2p_contextual_surface_alias_routing_synthesis/figures/h2p_contextual_surface_alias_routing_gate.svg",
                "Figure summarizing H2p exact-rate gains and transfer gates.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2m_execute_v1",
                "H2p live execution on H2m reaching 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2k_execute_v1",
                "H2p live execution on H2k preserving 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2l_execute_v1",
                "H2p live execution on H2l preserving 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2p_contextual_surface_alias_routing_on_h2f_execute_v1",
                "H2p live execution on H2f preserving 10/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2m_v1",
                "Direct H2p-vs-H2o comparison on H2m showing +0.125 strict and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2n_on_h2m_v1",
                "Direct H2p-vs-H2n comparison on H2m showing +0.625 strict and +0.375 executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2j_on_h2m_v1",
                "Direct H2p-vs-H2j comparison on H2m showing +0.625 strict and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2e_on_h2m_v1",
                "Direct H2p-vs-H2e comparison on H2m showing +0.875 strict and +0.625 executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2k_v1",
                "Transfer comparison showing H2p ties H2o on H2k at 8/8.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2l_v1",
                "Transfer comparison showing H2p ties H2o on H2l at 8/8.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2p_contextual_surface_alias_routing_vs_h2o_on_h2f_v1",
                "Transfer comparison showing H2p ties H2o on H2f at 10/10.",
            ),
        ),
    ),
    Claim(
        claim_id="C55_h2q_composed_surface_value_stale_breaks_h2p_saturation",
        claim=(
            "A composed H2q packet breaks the post-H2p saturation by mixing surface aliases, value-bearing labels, "
            "stale-selection hints, and decoy overlap in the same replay-shaped visual states."
        ),
        status="supported_current_packets_boundary",
        evidence_strength="strong_internal",
        primary_metric=(
            "On H2q, H2p remains the strongest current row but reaches only 3/8 strict and 3/8 "
            "executor-equivalent. H2o reaches 2/8, H2n reaches 0/8 strict and 1/8 executor-equivalent, "
            "and H2e reaches 1/8 strict and 2/8 executor-equivalent. H2p adds +0.125 strict over H2o, "
            "+0.375 over H2n, and +0.25 over H2e, but still leaves five non-exact rows."
        ),
        limitation=(
            "H2q is an 8-case replay-shaped synthetic packet, not a broad GUI population estimate. It supports a "
            "new boundary claim and a next mechanistic target, not a production promotion or a solved live operator "
            "policy."
        ),
        next_test=(
            "Build H2r around composed route gating: reject stale refine_selection calls when the latest prompt says "
            "to ignore old selections, and prefer requested surface classes over nearby same-value comments, banners, "
            "controls, and history context."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2q_composed_surface_value_stale_synthesis/report.md",
                "Dedicated H2q synthesis showing the post-H2p boundary and case-family failure structure.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2q_composed_surface_value_stale_synthesis/figures/h2q_composed_surface_value_stale_gate.svg",
                "Figure summarizing H2q exact-rate results across H2e, H2n, H2o, and H2p.",
            ),
            EvidenceSource(
                "design_packet",
                "results/tool_probe_replay_packets/20260512T_h2q_composed_surface_value_stale_dry_run_v1",
                "H2q dry-run packet defining composed surface/value/stale/decoy pressure.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2p_execute_v1",
                "H2p live execution on H2q reaching 3/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2o_execute_v1",
                "H2o live execution on H2q reaching 2/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2n_execute_v1",
                "H2n live execution on H2q reaching 0/8 strict and 1/8 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2e_execute_v1",
                "H2e live execution on H2q reaching 1/8 strict and 2/8 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2q_composed_surface_value_stale_h2p_vs_h2o_v1",
                "Direct H2p-vs-H2o comparison on H2q showing +0.125 strict and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2q_composed_surface_value_stale_h2p_vs_h2n_v1",
                "Direct H2p-vs-H2n comparison on H2q showing +0.375 strict and +0.25 executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2q_composed_surface_value_stale_h2p_vs_h2e_v1",
                "Direct H2p-vs-H2e comparison on H2q showing +0.25 strict and +0.125 executor-equivalence deltas.",
            ),
        ),
    ),
    Claim(
        claim_id="C56_h2r_composed_route_gating_solves_h2q_locally",
        claim=(
            "A narrow composed route-gating controller repairs the H2q post-H2p composition boundary by combining "
            "stale-selection rejection with requested-surface prioritization over same-value decoys."
        ),
        status="supported_current_packets_transfer_backtested",
        evidence_strength="strong_internal_scoped",
        primary_metric=(
            "On H2q, H2r reaches 8/8 strict and 8/8 executor-equivalent versus H2p at 3/8 and 3/8, "
            "a +0.625 exact-rate and executor-equivalence-rate improvement. H2r records 5 composed-route "
            "interventions matching H2p's five H2q misses: 2 stale-selection rewrites and 3 requested-surface "
            "restorations."
        ),
        limitation=(
            "This remains a local 8-case H2q mechanism claim. Transfer is now positive on the current packet set, "
            "but the helper was still designed after seeing H2q's failure structure and needs a fresh H2s holdout "
            "before global-policy language."
        ),
        next_test=(
            "Use the positive transfer backtest to define a fresh H2s composition holdout with unseen stale-selection "
            "and same-value surface decoys."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2r_composed_route_gating_synthesis/report.md",
                "Dedicated H2r synthesis showing the local H2q repair, mechanism split, and transfer caution.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2r_composed_route_gating_synthesis/figures/h2r_composed_route_gating_gate.svg",
                "Figure summarizing H2r exact-rate improvement over H2p on H2q.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h2q_execute_v2",
                "H2r live execution on H2q reaching 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2r_composed_route_gating_vs_h2p_on_h2q_v2",
                "Direct H2r-vs-H2p comparison on H2q showing +0.625 strict and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2q_composed_surface_value_stale_h2p_execute_v1",
                "H2p incumbent live execution on H2q reaching 3/8 strict and executor-equivalent.",
            ),
        ),
    ),
    Claim(
        claim_id="C57_h2r_transfer_backtest_preserves_current_gates",
        claim=(
            "H2r transfers across the current post-H2p/H2j packets and avoids the H2h-style H2b/H1x regression "
            "while also closing older unsaturated H1y/H1o/H1p slices."
        ),
        status="supported_current_packets_transfer_positive_requires_fresh_holdout",
        evidence_strength="strong_internal_transfer",
        primary_metric=(
            "Across nine transfer packets, H2r reaches 81/81 strict and 81/81 executor-equivalent; including the "
            "H2q origin packet it reaches 89/89 strict. It ties H2j/H2e on H2b and H1x, beats H2h by +0.40 "
            "exact-rate on H2b and +0.25 on H1x, improves H1y by +0.20 versus H2a, and improves H1o/H1p by "
            "+0.0833 exact-rate versus H1s."
        ),
        limitation=(
            "This is still an existing-packet transfer backtest, not a fresh independent population estimate. The "
            "next claim needs a new H2s holdout rather than more tuning on H2q/H2m/H2k/H2l/H2f/H2b/H1x/H1y/H1o/H1p."
        ),
        next_test=(
            "Build H2s with unseen composed stale-selection, value-bearing, contextual surface-alias, and same-value "
            "decoy cases; then run H2r and at least H2p/H2o/H2j controls without modifying H2r first."
        ),
        sources=(
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2r_transfer_backtest_synthesis/report.md",
                "Dedicated H2r transfer synthesis showing 81/81 transfer strict/executor-equivalent and 89/89 including H2q.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2r_transfer_backtest_synthesis/figures/h2r_transfer_backtest_gate.svg",
                "Figure summarizing H2r exactness across the origin, transfer, regression, and older unsaturated packets.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h2m_execute_v1",
                "H2r transfer execution on H2m reaching 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h2k_execute_v2",
                "H2r transfer execution on H2k reaching 8/8 strict and executor-equivalent after the negated-label guard.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h2l_execute_v2",
                "H2r transfer execution on H2l reaching 8/8 strict and executor-equivalent after preserving negated decoys.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h2f_execute_v1",
                "H2r transfer execution on H2f reaching 10/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h2b_execute_v1",
                "H2r H2b regression-gate execution reaching 5/5 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2r_composed_route_gating_vs_h2h_on_h2b_v1",
                "H2r versus H2h on H2b showing +0.40 exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h1x_execute_v1",
                "H2r H1x regression-gate execution reaching 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2r_composed_route_gating_vs_h2h_on_h1x_v1",
                "H2r versus H2h on H1x showing +0.25 exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h1y_execute_v1",
                "H2r older-slice execution on H1y reaching 10/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2r_composed_route_gating_vs_h2a_on_h1y_v1",
                "H2r versus H2a on H1y showing +0.20 exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h1o_execute_v1",
                "H2r older-slice execution on H1o reaching 12/12 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2r_composed_route_gating_on_h1p_execute_v1",
                "H2r older-slice execution on H1p reaching 12/12 strict and executor-equivalent.",
            ),
        ),
    ),
    Claim(
        claim_id="C58_h2s_fresh_holdout_confirms_h2r_composed_route_gating",
        claim=(
            "A fresh H2s composed holdout confirms that H2r's composed route-gating policy transfers beyond the "
            "H2q-derived repair rows when the policy is frozen before the first run."
        ),
        status="supported_fresh_holdout_requires_h2t_or_packaged_transfer",
        evidence_strength="strong_internal_fresh_holdout",
        primary_metric=(
            "On H2s, H2r reaches 10/10 strict and 10/10 executor-equivalent; H2p and H2o each reach 3/10, "
            "and H2j reaches 1/10. H2r improves by +0.70 exact-rate and executor-equivalence-rate versus H2p "
            "and H2o, and +0.90 exact-rate versus H2j."
        ),
        limitation=(
            "H2s is still a replay-shaped synthetic visual holdout, not a broad packaged workflow or human GUI "
            "population. The clean result supports the mechanism but should next face H2t or packaged workflow transfer."
        ),
        next_test=(
            "Build H2t with harder wording and overreach controls, or promote the same composed-route pressure into "
            "packaged visual workflows without losing attribution."
        ),
        sources=(
            EvidenceSource(
                "replay_packet",
                "results/tool_probe_replay_packets/20260512T_h2s_fresh_composed_holdout_dry_run_v1",
                "Fresh H2s dry-run packet with 10 unseen composed surface/value/stale/decoy cases.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2r_execute_v1",
                "Frozen H2r live replay on H2s reaching 10/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2p_execute_v1",
                "H2p incumbent live replay on H2s reaching 3/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2o_execute_v1",
                "H2o value-bearing synthesis control on H2s reaching 3/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2s_fresh_composed_holdout_h2j_execute_v1",
                "H2j target-normalization control on H2s reaching 1/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2s_fresh_composed_holdout_h2r_vs_h2p_v1",
                "H2r versus H2p comparison on H2s showing +0.70 exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2s_fresh_composed_holdout_h2r_vs_h2o_v1",
                "H2r versus H2o comparison on H2s showing +0.70 exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2s_fresh_composed_holdout_h2r_vs_h2j_v1",
                "H2r versus H2j comparison on H2s showing +0.90 exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2s_fresh_composed_holdout_synthesis/report.md",
                "Dedicated H2s synthesis summarizing frozen H2r, controls, non-exact rows, and intervention counts.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2s_fresh_composed_holdout_synthesis/figures/h2s_fresh_composed_holdout_gate.svg",
                "Figure summarizing H2s exact-rate separation across H2j/H2o/H2p/H2r.",
            ),
        ),
    ),
    Claim(
        claim_id="C59_h2t_overreach_independence_breaks_h2r_via_negation_scope_normalization",
        claim=(
            "The H2t overreach-independence holdout breaks H2r top-line saturation by exposing a controller-induced "
            "negation-scope target-normalization regression, not a raw local-MLX Gemma failure on the failed rows."
        ),
        status="supported_fresh_holdout_requires_h2u_negation_aware_normalization",
        evidence_strength="strong_internal_fresh_holdout",
        primary_metric=(
            "On H2t, H2r/H2p/H2o/H2j all reach 8/10 strict and 8/10 executor-equivalent; H2e reaches 6/10 "
            "strict but 9/10 executor-equivalent. H2r improves +0.20 strict exact-rate versus H2e but loses "
            "-0.10 executor-equivalence-rate, and the 2 H2r misses are raw-exact outputs rewritten by controller "
            "target-query normalization to note/caption labels."
        ),
        limitation=(
            "H2t is still a replay-shaped synthetic visual holdout. It cleanly identifies a controller tradeoff, "
            "but H2u must prove that negation-aware normalization preserves H2s/H2q/H2m gains before publication "
            "promotion."
        ),
        next_test=(
            "Build H2u as a negation-aware target-query normalization guard, then run transfer gates over H2t, H2s, "
            "H2q, H2m, H2k, H2l, H2f, H2b, and H1x."
        ),
        sources=(
            EvidenceSource(
                "replay_packet",
                "results/tool_probe_replay_packets/20260512T_h2t_overreach_independence_dry_run_v1",
                "Fresh H2t overreach-independence dry-run packet with negation-scope and low-score/value controls.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2r_execute_v1",
                "Frozen H2r live replay on H2t reaching 8/10 strict and executor-equivalent with two bad normalization rows.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2e_execute_v1",
                "H2e route-arbitration control on H2t reaching 6/10 strict and 9/10 executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2j_execute_v1",
                "H2j target-query normalization control tying H2r at 8/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2o_execute_v1",
                "H2o value-bearing synthesis control tying H2r at 8/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260512T_h2t_overreach_independence_h2p_execute_v1",
                "H2p contextual surface alias-routing control tying H2r at 8/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2e_v1",
                "H2r versus H2e comparison showing +0.20 strict exact-rate and -0.10 executor-equivalence-rate.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2p_v1",
                "H2r versus H2p comparison showing zero delta on H2t.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2o_v1",
                "H2r versus H2o comparison showing zero delta on H2t.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260512T_h2t_overreach_independence_h2r_vs_h2j_v1",
                "H2r versus H2j comparison showing zero delta on H2t.",
            ),
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2t_overreach_independence_synthesis/report.md",
                "Dedicated H2t synthesis summarizing controller overreach, bad-normalization rows, and the H2u next test.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2t_overreach_independence_synthesis/figures/h2t_overreach_independence_gate.svg",
                "Figure summarizing H2t strict/executor tradeoff across H2e/H2j/H2o/H2p/H2r.",
            ),
        ),
    ),
    Claim(
        claim_id="C60_h2u_negation_guard_repairs_h2t_without_h2s_h2q_h2m_regression",
        claim=(
            "The H2u negation-aware guard repairs the H2t controller-induced note/caption overreach by guarding both "
            "target-query normalization and composed-route gating, while preserving H2r performance on H2s, H2q, and H2m."
        ),
        status="supported_current_packets_needs_broader_transfer_backtest",
        evidence_strength="strong_internal_transfer_wave",
        primary_metric=(
            "H2u improves H2t from H2r's 8/10 strict and 8/10 executor-equivalent to 10/10 strict and 10/10 "
            "executor-equivalent (+0.20 exact-rate and +0.20 executor-equivalence-rate), fixes the two H2t "
            "negation-scope rows, and preserves 26/26 strict exactness across H2s/H2q/H2m with zero exact-rate "
            "and executor-equivalence-rate deltas versus H2r."
        ),
        limitation=(
            "The transfer wave covers H2s, H2q, and H2m but not yet the broader H2r backtest set, H1x, H2b/H2f/H2k/H2l, "
            "or packaged workflow execution. The guard is structural, but broader transfer is still required before "
            "global promotion."
        ),
        next_test=(
            "Run H2u over the full H2r transfer backtest matrix and then build a harder H2v/H3 packet that separates "
            "quoted negation, instructional negation, and genuine target negation."
        ),
        sources=(
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260513T_h2t_overreach_independence_h2u_execute_v2",
                "H2u live replay on H2t reaching 10/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260513T_h2u_negation_guard_vs_h2r_on_h2t_v1",
                "H2u versus H2r comparison on H2t showing +0.20 exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260513T_h2u_negation_guard_on_h2s_execute_v1",
                "H2u transfer replay on H2s preserving 10/10 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260513T_h2u_negation_guard_vs_h2r_on_h2s_v1",
                "H2u versus H2r comparison on H2s showing zero exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260513T_h2u_negation_guard_on_h2q_execute_v1",
                "H2u transfer replay on H2q preserving 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260513T_h2u_negation_guard_vs_h2r_on_h2q_v1",
                "H2u versus H2r comparison on H2q showing zero exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "live_replay_packet",
                "results/tool_probe_replay_live/20260513T_h2u_negation_guard_on_h2m_execute_v1",
                "H2u transfer replay on H2m preserving 8/8 strict and executor-equivalent.",
            ),
            EvidenceSource(
                "live_replay_comparison",
                "results/tool_probe_replay_live_comparisons/20260513T_h2u_negation_guard_vs_h2r_on_h2m_v1",
                "H2u versus H2r comparison on H2m showing zero exact and executor-equivalence deltas.",
            ),
            EvidenceSource(
                "replay_synthesis",
                "results/reports/h2u_negation_guard_synthesis/report.md",
                "Dedicated H2u synthesis summarizing H2t repair, transfer gates, fixed rows, and blocked guard rows.",
            ),
            EvidenceSource(
                "replay_synthesis_figure",
                "results/reports/h2u_negation_guard_synthesis/figures/h2u_negation_guard_transfer_gate.svg",
                "Figure summarizing H2u versus H2r exactness across H2t/H2s/H2q/H2m.",
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
