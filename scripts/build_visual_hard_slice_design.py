from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_hard_slice_design"


@dataclass(frozen=True)
class VisualCaseDesign:
    case_id: str
    family: str
    primary_discriminator: str
    expected_tool: str
    expected_argument_focus: str
    failure_pressure: str
    publishable_reason: str


CASE_DESIGNS: tuple[VisualCaseDesign, ...] = (
    VisualCaseDesign(
        case_id="visual_form_error_vs_message_author",
        family="visual_argument_copying",
        primary_discriminator="target_query_region_class_vs_business_subject",
        expected_tool="extract_layout",
        expected_argument_focus="target_query should name the visible error or warning region, not message author/source.",
        failure_pressure="v2/v4 tend to select recruiter/note/phone/source concepts instead of executable visual regions.",
        publishable_reason="Separates executable visual targeting from task-story nouns.",
    ),
    VisualCaseDesign(
        case_id="visual_form_error_with_prior_selection_decoy",
        family="visual_tool_routing",
        primary_discriminator="extract_layout_vs_refine_selection_when_no_real_selection_id",
        expected_tool="extract_layout",
        expected_argument_focus="image_id is copied from visual state; target_query stays on visible form error class.",
        failure_pressure="v4 over-preferred refine_selection with selection_id=latest on the form-target case.",
        publishable_reason="Tests whether schema hints cause false selection carryover.",
    ),
    VisualCaseDesign(
        case_id="visual_latest_filter_existing_selection",
        family="visual_referent_carryover",
        primary_discriminator="compact_filter_query_after_selection_id",
        expected_tool="refine_selection",
        expected_argument_focus="selection_id copied exactly; filter_query remains the literal token latest.",
        failure_pressure="v1 expanded latest into latest issue; v2/v4 fixed it.",
        publishable_reason="Preserves the current positive result as a regression guard.",
    ),
    VisualCaseDesign(
        case_id="visual_remaining_filter_existing_selection",
        family="visual_referent_carryover",
        primary_discriminator="compact_filter_query_non_latest_token",
        expected_tool="refine_selection",
        expected_argument_focus="filter_query remains remaining without surrounding nouns.",
        failure_pressure="Tests whether the latest-only fix generalizes to other compact selector tokens.",
        publishable_reason="Checks generality rather than overfitting to one literal.",
    ),
    VisualCaseDesign(
        case_id="visual_region_readback_after_layout_result",
        family="visual_region_readback",
        primary_discriminator="read_region_text_json_shape",
        expected_tool="read_region_text",
        expected_argument_focus="top-level call key remains name and region_id is copied as an opaque id.",
        failure_pressure="v3 emitted tool_name instead of name on readback.",
        publishable_reason="Guards protocol shape separately from visual selection semantics.",
    ),
    VisualCaseDesign(
        case_id="visual_metric_panel_vs_table_selector",
        family="visual_argument_copying",
        primary_discriminator="target_query_specific_visible_region_class",
        expected_tool="extract_layout",
        expected_argument_focus="target_query distinguishes metric panel from table without copying business prose.",
        failure_pressure="Tests target_query specificity without relying on validation-error wording.",
        publishable_reason="Adds fresh visual region classes beyond current replay cases.",
    ),
    VisualCaseDesign(
        case_id="visual_callout_warning_with_user_decoy",
        family="visual_argument_copying",
        primary_discriminator="target_query_visible_warning_vs_user_decoy",
        expected_tool="extract_layout",
        expected_argument_focus="target_query uses warning/callout region even when the user mentions a person or ticket.",
        failure_pressure="Targets the same semantic drift as recruiter note without reusing that surface.",
        publishable_reason="Fresh decoy case for form-target executability.",
    ),
    VisualCaseDesign(
        case_id="visual_selection_id_opaque_copy_with_filter",
        family="visual_referent_carryover",
        primary_discriminator="opaque_selection_id_copy",
        expected_tool="refine_selection",
        expected_argument_focus="selection_id is copied exactly from prior tool result and not replaced with latest/open/etc.",
        failure_pressure="v4 produced selection_id=latest on a case without a valid selection id.",
        publishable_reason="Separates selector token copying from opaque id copying.",
    ),
)


def build_design(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = [_case_row(case) for case in CASE_DESIGNS]
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "case_count": len(rows),
        "family_counts": family_counts,
        "purpose": "Design a fresh visual hard slice after v2/v3/v4 catalog-profile results; this is a design packet, not model evidence.",
    }
    _write_csv(tables_dir / "visual_hard_slice_case_designs.csv", rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "design.json").write_text(
        json.dumps({"manifest": manifest, "case_designs": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "design.md").write_text(_markdown(manifest, rows), encoding="utf-8")
    return {"manifest": manifest, "case_designs": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fresh visual hard-slice design packet.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_design(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


def _case_row(case: VisualCaseDesign) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "primary_discriminator": case.primary_discriminator,
        "expected_tool": case.expected_tool,
        "expected_argument_focus": case.expected_argument_focus,
        "failure_pressure": case.failure_pressure,
        "publishable_reason": case.publishable_reason,
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


def _markdown(manifest: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines = [
        "# Visual Hard Slice Design",
        "",
        "This packet designs the next fresh visual hard slice. It is not model evidence yet.",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- case_count: `{manifest['case_count']}`",
        f"- purpose: {manifest['purpose']}",
        "",
        "| Case ID | Family | Discriminator | Expected Tool | Argument Focus | Failure Pressure |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["family"],
                    row["primary_discriminator"],
                    row["expected_tool"],
                    row["expected_argument_focus"],
                    row["failure_pressure"],
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
