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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
