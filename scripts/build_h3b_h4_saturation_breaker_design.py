from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h3b_h4_saturation_breaker_design"


FAMILY_PLAN: tuple[dict[str, Any], ...] = (
    {
        "family_id": "h3b_unseen_stale_origin_paraphrase",
        "planned_cases": 4,
        "pressure_type": "controller_generalization",
        "primary_axis": "stale selection origin language outside H3a marker set",
        "expected_first_gate": "H3a combined versus H2w/H2z rows on the same packet",
        "why_it_breaks_saturation": (
            "H3a repaired archived, retired, remembered, shadow, carry-over, and leftover phrasing. This family "
            "uses unseen provenance language such as historical bookmark, decommissioned pane, orphaned cursor, "
            "and frozen snapshot while the stale selection id is present in state."
        ),
        "paper_role": "tests whether stale-selection repair learned a mechanism or memorized marker phrasing",
    },
    {
        "family_id": "h3b_extended_negative_value_vocabulary",
        "planned_cases": 4,
        "pressure_type": "semantic_generalization",
        "primary_axis": "negative-value target words outside current controller vocabulary",
        "expected_first_gate": "H3a combined versus H3a negative-only and H2z component-only",
        "why_it_breaks_saturation": (
            "H3a covers inactive, disabled, unresolved, unassigned, paused, rejected, missing, and expired. This "
            "family uses values such as suppressed, withheld, revoked, and voided with same-value note/table decoys."
        ),
        "paper_role": "tests whether value preservation is lexical coverage or general target binding",
    },
    {
        "family_id": "h3b_state_order_flip",
        "planned_cases": 4,
        "pressure_type": "order_sensitivity",
        "primary_axis": "decoy comes before the target and carries a higher-overlap value",
        "expected_first_gate": "H3a combined strict/executor with family-level trace counts",
        "why_it_breaks_saturation": (
            "Earlier packets often made the requested component directly recoverable from a label phrase. This family "
            "puts value-bearing decoys first and asks for the target after a state-order reversal."
        ),
        "paper_role": "separates target binding from row-order and first-match bias",
    },
    {
        "family_id": "h3b_current_selection_stepwise_refine",
        "planned_cases": 4,
        "pressure_type": "workflow_state",
        "primary_axis": "must use a current selection_id from prior tool state, not restart extraction",
        "expected_first_gate": "H3a combined versus no-controller-fallback and no-controller-repair",
        "why_it_breaks_saturation": (
            "H2t protected current-selection refinement, while H3 focused on stale selection rejection. This family "
            "forces the controller to preserve the distinction under new current-selection language."
        ),
        "paper_role": "checks that stale-selection repair does not damage valid stateful CLI operation",
    },
    {
        "family_id": "h4_latest_instruction_retention",
        "planned_cases": 4,
        "pressure_type": "instruction_order",
        "primary_axis": "prior tool result plus latest user instruction retargets the visual operation",
        "expected_first_gate": "H3a combined and H3a no-fallback, with strict latest-instruction deltas",
        "why_it_breaks_saturation": (
            "The frontier-model benchmark table the project is aiming toward emphasizes agentic, UI-control, and "
            "long-context behavior. This family turns that into a local replay contract by making the latest "
            "instruction override earlier visual context."
        ),
        "paper_role": "links local harness evidence to direction-following and UI-control benchmark claims",
    },
    {
        "family_id": "h4_approval_stop_boundary",
        "planned_cases": 4,
        "pressure_type": "operator_safety",
        "primary_axis": "must stop or request approval rather than continuing a packaged workflow step",
        "expected_first_gate": "CLI live harness session events and approval artifacts, not a frontend surface",
        "why_it_breaks_saturation": (
            "Moonie is not just a single tool-call benchmark. This family introduces live-operator behavior: approval "
            "state, sandbox posture, and stop behavior must remain attributable to a workflow family."
        ),
        "paper_role": "moves from exact tool-call replay toward publishable live-agent safety evidence",
    },
)


SCORE_CONTRACT: tuple[dict[str, Any], ...] = (
    {
        "metric_id": "strict_exact",
        "required": True,
        "definition": "actual tool call array exactly equals the oracle call array",
        "failure_signal": "argument, tool, or call-count mismatch",
        "reporting_grain": "overall, family, case, comparison",
    },
    {
        "metric_id": "executor_equivalence",
        "required": True,
        "definition": "actual execution reaches the expected target region or session state",
        "failure_signal": "wrong target, invalid call, no call, or policy-state miss",
        "reporting_grain": "overall, family, case, comparison",
    },
    {
        "metric_id": "controller_trace",
        "required": True,
        "definition": "controller helper metadata attached to each repaired or preserved call",
        "failure_signal": "score changes without helper attribution",
        "reporting_grain": "helper kind, case, family, profile",
    },
    {
        "metric_id": "regression_count",
        "required": True,
        "definition": "candidate misses a case that the baseline passed",
        "failure_signal": "candidate strict false where baseline strict true",
        "reporting_grain": "comparison, family, case",
    },
    {
        "metric_id": "helper_overtrigger",
        "required": True,
        "definition": "new helper fires outside its intended family or on transfer rows where it should be silent",
        "failure_signal": "nonzero unrelated helper interventions on transfer/back-compat batteries",
        "reporting_grain": "helper kind, family, transfer packet",
    },
    {
        "metric_id": "live_operator_artifact",
        "required": True,
        "definition": "CLI Rich/live run leaves inspectable manifest, summary, commands, case states, and run outputs",
        "failure_signal": "unattributed live run or missing sandbox/session evidence",
        "reporting_grain": "session, workflow family, replay packet",
    },
)


BASELINE_PLAN: tuple[dict[str, Any], ...] = (
    {
        "baseline_label": "h2w_semantic_target_preservation",
        "system_id": "mlx_gemma4_e2b_reasoner_only_h2w_no_controller_fallback",
        "role": "strong pre-H3a transfer/back-compat reference",
    },
    {
        "baseline_label": "h2z_boundary_combined",
        "system_id": "mlx_gemma4_e2b_reasoner_only_h2z_boundary_combined",
        "role": "shows whether H3a repairs are still needed on the new packet",
    },
    {
        "baseline_label": "h3a_combined",
        "system_id": "mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined",
        "role": "current candidate to break before adding new helpers",
    },
    {
        "baseline_label": "h3a_no_fallback",
        "system_id": "mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined_no_controller_fallback",
        "role": "planned ablation row for fallback dependence once the registry row exists",
    },
    {
        "baseline_label": "gemini_cli_external_reference",
        "system_id": "gemini_cli_external_baseline",
        "role": "non-replacement external operator reference for packaged workflow behavior",
    },
)


EXTERNAL_ALIGNMENT: tuple[dict[str, Any], ...] = (
    {
        "external_benchmark": "Terminal-bench style",
        "benchmark_group": "Coding / terminal agency",
        "moonie_mapping": "CLI packaged workflow runs, replay-live commands, sandbox manifests, and session event traces",
        "claim_boundary": "Moonie reports local Gemma harnessing quality, not Terminal-bench leaderboard parity",
    },
    {
        "external_benchmark": "Toolathlon style",
        "benchmark_group": "Agentic tool use",
        "moonie_mapping": "strict tool-call, executor-equivalence, helper-causal ablations, and regression rows",
        "claim_boundary": "Tool-use claims require helper attribution and no hidden controller-credit collapse",
    },
    {
        "external_benchmark": "OSWorld-Verified style",
        "benchmark_group": "UI control",
        "moonie_mapping": "visual executor target contracts, stale/current selection state, and live CLI operator evidence",
        "claim_boundary": "Local visual replay is a controlled UI-substrate proxy, not full desktop OS control",
    },
    {
        "external_benchmark": "SWE-Bench / Terminal repair style",
        "benchmark_group": "Recovered execution",
        "moonie_mapping": "repaired versus strict correctness, fallback-causal deltas, and output usability records",
        "claim_boundary": "Recovered success is separated from strict success in every table",
    },
    {
        "external_benchmark": "Long-context direction-following style",
        "benchmark_group": "Instruction retention",
        "moonie_mapping": "latest-instruction override cases with prior tool state and stale provenance decoys",
        "claim_boundary": "Claims are limited to packaged workflow/replay contexts until broader long-context runs exist",
    },
)


def build_h3b_h4_saturation_breaker_design(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    family_rows = [dict(row) for row in FAMILY_PLAN]
    score_rows = [dict(row) for row in SCORE_CONTRACT]
    baseline_rows = [dict(row) for row in BASELINE_PLAN]
    alignment_rows = [dict(row) for row in EXTERNAL_ALIGNMENT]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "phase": "h3b_h4_saturation_breaker_design",
        "planned_family_count": len(family_rows),
        "planned_case_count": sum(int(row["planned_cases"]) for row in family_rows),
        "score_metric_count": len(score_rows),
        "baseline_count": len(baseline_rows),
        "external_alignment_count": len(alignment_rows),
        "current_candidate": "mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined",
        "first_execution_packet": "h3b_saturation_breaker_v27",
        "primary_decision_gate": (
            "Score H3a before adding new helpers; promote no new controller unless it fixes a named family, has "
            "clean transfer/back-compat evidence, and leaves helper traces attributable."
        ),
        "publication_standard": (
            "Every top-line score must be paired with controller-dependence, executor-equivalence, regression, "
            "family, and live artifact evidence."
        ),
    }
    payload = {
        "manifest": manifest,
        "family_rows": family_rows,
        "score_rows": score_rows,
        "baseline_rows": baseline_rows,
        "external_alignment_rows": alignment_rows,
    }

    _write_csv(tables_dir / "h3b_h4_family_plan.csv", family_rows)
    _write_csv(tables_dir / "h3b_h4_score_contract.csv", score_rows)
    _write_csv(tables_dir / "h3b_h4_baseline_plan.csv", baseline_rows)
    _write_csv(tables_dir / "h3b_h4_external_benchmark_alignment.csv", alignment_rows)
    _write_svg(figures_dir / "h3b_h4_benchmark_pressure_map.svg", family_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "design.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    family_rows = payload["family_rows"]
    score_rows = payload["score_rows"]
    baseline_rows = payload["baseline_rows"]
    alignment_rows = payload["external_alignment_rows"]
    lines = [
        "# H3b/H4 Saturation-Breaker Design",
        "",
        "## Why this exists",
        "",
        (
            "H3a now closes the fresh H3 packet and preserves the broad H2w-era transfer battery. The next "
            "benchmark should therefore stop proving the same surface is saturated and instead create harder "
            "evidence about controller generalization, stateful operation, approval boundaries, and live CLI "
            "execution."
        ),
        "",
        "The design follows the standard implied by frontier agent benchmark tables: every score is grouped by a named benchmark family, but Moonie keeps an extra attribution layer for controller dependence.",
        "",
        "## Manifest",
        "",
        f"- Planned families: `{manifest['planned_family_count']}`",
        f"- Planned cases: `{manifest['planned_case_count']}`",
        f"- Score metrics: `{manifest['score_metric_count']}`",
        f"- Current candidate: `{manifest['current_candidate']}`",
        f"- First execution packet: `{manifest['first_execution_packet']}`",
        "",
        "## Families",
        "",
        _table(["family_id", "planned_cases", "pressure_type", "paper_role"], family_rows),
        "",
        "## Score Contract",
        "",
        _table(["metric_id", "required", "definition", "reporting_grain"], score_rows),
        "",
        "## Baselines",
        "",
        _table(["baseline_label", "system_id", "role"], baseline_rows),
        "",
        "## External Benchmark Alignment",
        "",
        _table(["external_benchmark", "benchmark_group", "moonie_mapping", "claim_boundary"], alignment_rows),
        "",
        "## Decision Gate",
        "",
        manifest["primary_decision_gate"],
        "",
        manifest["publication_standard"],
        "",
    ]
    return "\n".join(lines)


def _table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "/")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_svg(path: Path, family_rows: list[dict[str, Any]]) -> None:
    width = 1120
    row_height = 44
    top = 88
    height = top + len(family_rows) * row_height + 64
    max_cases = max(int(row["planned_cases"]) for row in family_rows)
    colors = ["#0F766E", "#B45309", "#4338CA", "#047857", "#7C3AED", "#BE123C"]
    bars = []
    for index, row in enumerate(family_rows):
        y = top + index * row_height
        cases = int(row["planned_cases"])
        bar_width = int(380 * cases / max_cases)
        color = colors[index % len(colors)]
        bars.append(
            f'<text x="40" y="{y + 24}" font-size="15" fill="#111827">{_svg(row["family_id"])}</text>'
            f'<rect x="470" y="{y + 6}" width="{bar_width}" height="24" fill="{color}" rx="3" />'
            f'<text x="{470 + bar_width + 12}" y="{y + 24}" font-size="14" fill="#374151">{cases} cases / {_svg(row["pressure_type"])}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#ffffff" />
  <text x="40" y="42" font-size="24" font-weight="700" fill="#111827">H3b/H4 Benchmark Pressure Map</text>
  <text x="40" y="68" font-size="14" fill="#4B5563">Designed to break H3a top-line saturation while preserving family-level attribution.</text>
  {''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _svg(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the H3b/H4 saturation-breaker design report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_h3b_h4_saturation_breaker_design(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
